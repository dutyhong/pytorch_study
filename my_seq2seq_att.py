import torch
from torch import nn
from torch.nn.functional import softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_dim):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size)
    def forward(self, input, hidden):
        embed_out = self.embed(input).view(1, 1, -1)
        output, hts = self.gru(embed_out, hidden)
        return output, hts
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(device)

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_dim):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size)
        self.dense = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
    def forward(self, input, hidden):
        embed_out = self.embed(input).view(1, 1, -1)
        output, hts = self.gru(embed_out, hidden)
        dense_out = self.dense(output[0])
        softmax_out = self.log_softmax(dense_out)
        return softmax_out, hidden #新的hidden要作为下一个timestep的隐层输入，所以必须返回
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = torch.nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)

        output = torch.nn.functional.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


