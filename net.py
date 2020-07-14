import torch
import torch.nn as nn
import torch.nn.functional as F

# * adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_log_softmax(vector, mask, dim=-1):
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector + (mask + 1e-45).log()
    return F.log_softmax(vector, dim=dim)

# * adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_max(vector, mask, dim, keepdim=False, min_val=-1e7):
    one_minus_mask = (1. - mask).byte()
    replaced_vector = vector.masked_fill(one_minus_mask.bool(), min_val)
    max_value, max_index = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value, max_index


class encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, bidir):
        super(encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.bidir = bidir

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=n_layers, bidirectional=self.bidir, batch_first=True)

    def forward(self, inputs, inputs_length):
        # * have to utilize pack-pad change because of using bidirection LSTM
        packed = nn.utils.rnn.pack_padded_sequence(inputs, inputs_length, batch_first=True)
        outputs, hidden = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden


class attention(nn.Module):
    def __init__(self, hidden_dim):
        super(attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.W1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.W2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.V = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, mask):
        # * encoder_outputs: [batch_size, seq_len, hidden_dim]
        # * decoder_state: [batch_size, hidden_dim]
        encoder_trans = self.W1(encoder_outputs)
        # * encoder_trans: [batch_size, seq_len, hidden_dim]
        decoder_trans = self.W2(decoder_state).unsqueeze(1)
        # * decoder_trans: [batch_size, 1, hidden_dim]
        u_i = self.V(torch.tanh(encoder_trans + decoder_trans)).squeeze(-1)
        # * u_i: [batch_size, seq_len]

        # * it is softmax in origin paper, here is log_softmax
        log_score = masked_log_softmax(u_i, mask, dim=-1)

        return log_score


class pointer_net(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers=1, bidir=True, batch_first=True):
        super(pointer_net, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bidir = bidir
        self.n_directions = 2 if self.bidir else 1
        self.n_layers = n_layers

        self.embedding_layer = nn.Linear(self.input_dim, self.embedding_dim, bias=False)
        self.encoder = encoder(embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim, n_layers=self.n_layers, bidir=self.bidir)
        # * the decoder is implemented by LSTMCell for convinience
        self.decoder = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.attn = attention(hidden_dim=self.hidden_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, input_seq, input_lengths):
        batch_size = input_seq.size(0)
        max_seq_len = input_seq.size(1)

        embedded = self.embedding_layer(input_seq)
        encoder_outputs, encoder_hidden = self.encoder.forward(embedded, input_lengths)

        # * change the size of encoder_outputs to the 0.5 times of origin's because of using bidirection LSTM
        if self.bidir:
            encoder_outputs = encoder_outputs[:, :, : self.hidden_dim] + encoder_outputs[:, :, self.hidden_dim:]

        encoder_hn, encoder_cn = encoder_hidden
        encoder_hn = encoder_hn.view(self.n_layers, self.n_directions, batch_size, self.hidden_dim)
        encoder_cn = encoder_cn.view(self.n_layers, self.n_directions, batch_size, self.hidden_dim)

        # * initialize the decoder input as a zero tensor
        decoder_input = encoder_outputs.new_zeros(torch.Size([batch_size, self.hidden_dim]))
        # * initialize the decoder hidden tensor and change the size of it to fit the LSTMCell function
        decoder_hidden = (encoder_hn[-1, 0, :, :].squeeze(), encoder_cn[-1, 0, :, :].squeeze())

        range_tensor = torch.arange(max_seq_len, dtype=input_lengths.dtype).expand(batch_size, max_seq_len, max_seq_len)
        each_len_tensor = input_lengths.view(-1, 1, 1).expand(batch_size, max_seq_len, max_seq_len)

        row_mask_tensor = (range_tensor < each_len_tensor)
        col_mask_tensor = row_mask_tensor.transpose(1, 2)
        mask_tensor = row_mask_tensor * col_mask_tensor

        pointer_log_scores = []
        pointer_argmaxs = []

        for i in range(max_seq_len):
            sub_mask = mask_tensor[:, i, :].float()

            h_i, c_i = self.decoder(decoder_input, decoder_hidden)

            decoder_hidden = (h_i, c_i)

            log_score = self.attn.forward(h_i, encoder_outputs, sub_mask)
            pointer_log_scores.append(log_score)

            _, masked_argmax = masked_max(log_score, sub_mask, keepdim=True, dim=1)
            pointer_argmaxs.append(masked_argmax)
            index_tensor = masked_argmax.unsqueeze(-1).expand(batch_size, 1, self.hidden_dim)

            # * find the max probability of input element and initialize the next decoder input as the encoder output of it
            decoder_input = torch.gather(encoder_outputs, dim=1, index=index_tensor).squeeze(1)

        pointer_log_scores = torch.stack(pointer_log_scores, 1)
        pointer_argmaxs = torch.cat(pointer_argmaxs, 1)
        return pointer_log_scores, pointer_argmaxs, mask_tensor
