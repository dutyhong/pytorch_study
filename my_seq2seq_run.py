import os
import random

from torch.nn import NLLLoss
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

from pytorch_study.my_seq2seq_att import Encoder
from pytorch_study.my_seq2seq_att import Decoder
from pytorch_study.seq2seq_data_process import ModelInput
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def to_train_tensor(train_pairs):
    train_in = []
    train_out = []
    for in_sen, out_sen in train_pairs:
        in_sen = torch.tensor(in_sen, dtype=torch.long).view(-1, 1).to(device)
        out_sen = torch.tensor(out_sen, dtype=torch.long).view(-1, 1).to(device)
        train_in.append(in_sen)
        train_out.append(out_sen)
    return train_in, train_out
def shuffle_train_tensor(train_in, train_out):
    total_train = []
    random_train_in = []
    random_train_out = []
    for in_sen, out_sen in zip(train_in, train_out):
        total_train.append((in_sen, out_sen))
    random.shuffle(total_train)
    for in_sen, out_sen in total_train:
        random_train_in.append(in_sen)
        random_train_out.append(out_sen)
    return random_train_in, random_train_out
def train_process(encoder:Encoder, decoder:Decoder, train_tensors, model_input:ModelInput):
    loss_fn = torch.nn.NLLLoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    loss = 0
    cnt = 0
    teacher_forcing_ratio = 0.5
    # step_cnt = 0
    for in_tensor, out_tensor in zip(train_tensors[0], train_tensors[1]):
        cnt = cnt + 1
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        init_hidden = encoder.init_hidden()
        encoder_hidden = init_hidden
        for idin in in_tensor:
            encoder_out, encoder_hidden = encoder(idin,encoder_hidden)
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor(model_input.out_word2index["SOS"], dtype=torch.long).to(device)
        decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden) ##输入第一步的SOS
        ##using teacher-forcing
        teacher_force = True if random.random() < teacher_forcing_ratio else False
        if teacher_force:
            for idout in out_tensor:
                decoder_out, decoder_hidden = decoder(idout, decoder_hidden)
                ##计算loss
                loss = loss + loss_fn(decoder_out, idout)
        else:
            for idout in out_tensor:
                decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_out.topk(1)
                decoder_input = topi.squeeze().detach()##取排序最大的index
                loss = loss + loss_fn(decoder_out, idout)
                if decoder_input==model_input.EOS_token:
                    break
        if cnt%256==0:
            print("一个batch训练loss为：%f" % (loss / cnt))
            loss.backward(retain_graph=True)
            decoder_optimizer.step()
            encoder_optimizer.step()
            torch.cuda.empty_cache()
            loss = 0
            cnt = 0
    ##训练完保存模型
    print("save model ing!!!!!!!!")
    torch.save(encoder.state_dict(), "./data/encoder_bin.pt")
    torch.save(decoder.state_dict(), "./data/decoder_bin.pt")
def evaluate(encoder:Encoder, decoder:Decoder, input_sents:list, model_input:ModelInput, max_length:int):
    ##将输入字符串转换为模型输入
    decoder_sen = []
    for input_sent in input_sents:
        words = input_sent.split(" ")
        input_tensor = torch.tensor([model_input.in_word2index[word] for word in words], dtype=torch.long).view(-1,1).to(device)
        # inout_tensors.append(input_tensor)
        encoder_hidden = encoder.init_hidden()
        decoder_words = []
        for inid in input_tensor:
            encoder_output, encoder_hidden = encoder(inid, encoder_hidden)
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor(model_input.SOS_token, dtype=torch.long).to(device)
        for i in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item()==model_input.EOS_token:
                decoder_words.append(model_input.EOS_token)
                break
            else:
                decoder_words.append(model_input.out_index2word[topi.item()])
            decoder_input = topi.squeeze().detach()
        decoder_sen.append(decoder_words)
    return decoder_sen
if __name__ =='__main__':
    model_input = ModelInput("./data/eng-fra2.txt")
    model_input.read_data()
    train_pairs = model_input.train_pairs
    train_tensors = to_train_tensor(train_pairs)
    in_words = model_input.in_words
    out_words = model_input.out_words
    hidden_size = 128
    embed_dim = 256
    encoder = Encoder(in_words, hidden_size, embed_dim)#.to(device)
    decoder = Decoder(out_words, hidden_size, embed_dim)#.to(device)
    if torch.cuda.device_count()>1:
        encoder = torch.nn.DataParallel(encoder)
        decoder = torch.nn.DataParallel(decoder)
    encoder.to(device)
    decoder.to(device)
    if isinstance(encoder, torch.nn.DataParallel):
        encoder = encoder.module
    if isinstance(decoder, torch.nn.DataParallel):
        decoder = decoder.module
    epochs = 10
    # for i in range(epochs):
    #     print("第%d个epoch!!!"%(i))
    #     train_process(encoder, decoder, train_tensors,model_input)
    #     train_tensors = shuffle_train_tensor(train_tensors[0], train_tensors[1])
    #测试
    sents = ["I saw you.", "the moment go","How good are you?"]
    sents = [model_input.normalizeString(sent) for sent in sents]
    encoder.load_state_dict(torch.load("./data/encoder_bin.pt"))
    decoder.load_state_dict(torch.load("./data/decoder_bin.pt"))
    decoder_sent = evaluate(encoder, decoder, sents, model_input, 10)
    print("ddd")

