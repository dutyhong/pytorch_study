from datetime import date, datetime

import torch.nn as nn
import torch
from torch.nn import RNN
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from pytorch_study.data_process import get_train_test_data, get_batch_data, get_robot_train_test_data
import numpy as np
# import torchtext
# transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# output = loss(input, target)
# output.backward()
#
# loss = nn.BCEWithLogitsLoss()
# input = torch.randn(3, requires_grad=True)
# target = torch.empty(3).random_(2)
# output = loss(input, target)
# output.backward()
# tf = nn.Transformer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChRnnCfn(nn.Module):
    def __init__(self, hidden_size, vocab_size, embed_dim, num_layers=1, n_categories=2):
        super(ChRnnCfn, self).__init__()
        self.embeds = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, n_categories)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    def forward(self, input_data):
        embed = self.embeds(input_data).permute(1, 0, 2)  #view(max_seq_len, -1, embed_dim)
        output, hidden = self.rnn(embed, self.init_hidden(self.num_layers, len(input_data), self.hidden_size))
        tmp_output = output[-1, :, :]
        linear_out = self.linear(self.relu(tmp_output))
        result = self.softmax(linear_out)
        return result
    def init_hidden(self, num_layers, batch_size, hidden_size):
        return (torch.randn(num_layers, batch_size, hidden_size),
                torch.randn(num_layers, batch_size, hidden_size))

class ChRnnAttCfn(nn.Module):
    def __init__(self, hidden_size, vocab_size, embed_dim, attention_size, seq_length, num_layers=1, bidirectional=False):
        super(ChRnnAttCfn, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.bidirectional = bidiretional
        self.num_directions = 2 if bidirectional else 1
        self.embed_layer = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers, bidirectional=bidirectional, dropout=0.3)
        self.linear = nn.Linear(hidden_size*num_layers*self.num_directions, n_categories)
        self.ws1_att = torch.zeros(hidden_size*num_layers*self.num_directions, attention_size).to(device)
        self.ws2_att = torch.zeros(attention_size,1).to(device)
        self.softmax = nn.Softmax(dim=1)
    def attention_layer(self, lstm_output):
        ##output : (seq_length, batch_size, hidden_size*num_layers)
        lstm_output_tmp = torch.reshape(lstm_output, [-1, self.hidden_size*self.num_layers*self.num_directions]) #（batch_size*seq_length, hidden_size*num_layers）
        att1 = torch.tanh(torch.mm(lstm_output_tmp, self.ws1_att)) # (batch_size*seq_length, attention_size)
        att2 = torch.mm(att1, self.ws2_att) # (batch_size*seq_length, 1)
        exps = torch.Tensor.reshape(torch.exp(att2), [-1, self.seq_length]) #(batch_size, seq_length)
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1]) #(batch_size*seq_length, 1)
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.seq_length, 1])
        #输出向量为 m=a*H
        lstm_output = lstm_output.permute(1, 0, 2)
        att_out = torch.sum(alphas_reshape*lstm_output, 1) #(batch_size, hidden_size*num_layers)
        return att_out
    def forward(self, input_data):
        embed_out = self.embed_layer(input_data) #(batch_size, seq_length, embed_dim)
        embed_out = embed_out.permute(1, 0, 2)
        lstm_out, (ht, ct) = self.rnn(embed_out) #(seq_length, batch_size, hidden_size*num_layers)
        att_out = self.attention_layer(lstm_out) # (batch_size, hidden_size*num_layers)
        dense_out = self.linear(att_out)
        dense_out = self.softmax(dense_out)
        return dense_out


##模型可视化
# a = np.random.randint(1,20, size=(10, 20), dtype=np.long)
# # np.random.ra
# dummy_input = torch.from_numpy(a)  #torch.rand(10, 20, dtype=torch.long)
# with SummaryWriter(comment = "chrnncfnmodel") as w:
#     w.add_graph(model, (dummy_input, ))
# summary(model, input_size=torch.tensor((32, 20), dtype=torch.long))
# for param in model.parameters():
#     print(param)
##获取数据
train_data, test_data, val_data, vocabs = get_train_test_data()  # get_robot_train_test_data() #
vocab_size = len(vocabs)
word_to_index = {}
index_to_word = {}
for i, char in enumerate(vocabs):
    word_to_index[char] = i
    index_to_word[i] = char
##将每行变成ids
def line_to_ids(line, max_seq_len):
    ids = torch.zeros(max_seq_len, dtype=torch.long)
    chars = list(line)
    if len(chars)>=max_seq_len:
        for i in range(max_seq_len):
            ids[i] = word_to_index[chars[i]]
    else:
        for i in range(len(chars)):
            ids[i] = word_to_index[chars[i]]
    return ids

def get_batch_train_tensor(batch_data, max_seq_len):
    batch_size = len(batch_data)
    sample_tensors = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    label_tensors = torch.zeros(batch_size, dtype=torch.long)
    for i, sample in enumerate(batch_data):
        label = int(sample[0])
        text = sample[1]
        line_tensor = line_to_ids(text, max_seq_len)
        label_tensors[i] = label
        sample_tensors[i,:] = line_tensor
    return label_tensors, sample_tensors

#保存模型和加载模型
# torch.save(model.state_dict(), "craf")
# model = ChRnnCfn(hidden_size, vocab_size, embed_dim, num_layers, n_categories)
# model.load_state_dict(torch.load("craf"))

def evaluate(test_data, model):
    model.eval()
    with torch.no_grad():
        test_data_tensors = get_batch_train_tensor(test_data, max_seq_len)
        test_true_labels = test_data_tensors[0].to(device)
        test_samples = test_data_tensors[1].to(device)
        log_probs = model(test_samples)
        # nlogs = log_probs.detach().numpy()
        cnt = 0
        total_cnt = len(test_data)
        for i in range(total_cnt):
            if torch.argmax(log_probs[i])==test_true_labels[i]:
                cnt = cnt + 1
        print("准确率 ：%f"%(cnt/total_cnt))
if __name__== '__main__':
    # device = torch.device("cpu")
    hidden_size = 10
    num_layers = 1
    n_categories = 2
    embed_dim = 64
    max_seq_len = 20
    batch_size = 32
    bidiretional = False
    epochs = 20


    # model = ChRnnCfn(hidden_size, vocab_size, embed_dim, num_layers=num_layers)
    model = ChRnnAttCfn(hidden_size, vocab_size, embed_dim, attention_size=int(max_seq_len / 2), seq_length=max_seq_len,
                        num_layers=num_layers, bidirectional=bidiretional).to(device)

    ##训练过程
    # loss_fn = nn.NLLLoss()
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(model.parameters(),lr=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_batchs = get_batch_data(train_data, batch_size)
    start = datetime.now()
    for i in range(epochs):
        print("\n\nepoch %s*****\n\n" % i)
        for j, batch_data in enumerate(train_batchs):
            # model.zero_grad()
            batch_data = get_batch_train_tensor(batch_data, max_seq_len)
            batch_true_labels = batch_data[0].to(device)
            batch_samples = batch_data[1].to(device)
            # one_sample = one_sample.view(max_seq_len, 1, -1)
            log_probs = model(batch_samples)
            loss = loss_fn(log_probs, batch_true_labels)
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("iteration %s,batch_cnt%s" %(i,j))
    end = datetime.now()
    print("耗时 ：%f秒" % ((end - start).seconds))

    torch.save(model, "test_model.pt")
    torch.save(model.state_dict(), "test_model_params.pt")

    evaluate(test_data, model)
# binary_accuracy