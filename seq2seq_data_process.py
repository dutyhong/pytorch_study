import random
import re
import unicodedata

class ModelInput(object):
    def __init__(self, path):
        self.path = path
        self.in_word2index = {"SOS":0, "EOS":1}
        self.in_index2word = {0:"SOS", 1:"EOS"}
        self.out_word2index = {"SOS":0, "EOS":1}
        self.out_index2word = {0:"SOS", 1:"EOS"}
        self.in_words = 2
        self.out_words = 2
        self.train_in_data = []
        self.train_out_data = []
        self.train_pairs = []
        self.EOS_token = 1
        self.SOS_token = 0
        self.max_length = 10
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters

    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s
    def read_data(self):
        # path = "./data/eng-fra1.txt"
        path = self.path
        file = open(path, "r")
        in_words_set = set()
        out_words_set = set()
        train_in = []
        train_out = []
        train_pairs = []
        for line in file.readlines():
            columns = line.split("\t")
            if len(columns)<2:
                continue
            eng = self.normalizeString(columns[0])
            fra = self.normalizeString(columns[1])
            eng_chars = eng.split(" ")
            fra_chars = fra.split(" ")
            if len(eng_chars)>10 or len(fra_chars)>10:
                continue
            eng_chars.append("EOS")
            fra_chars.append("EOS")
            train_in.append(eng_chars)
            train_out.append(fra_chars)
            for eng_char in eng_chars:
                in_words_set.add(eng_char)
            for fra_char in fra_chars:
                out_words_set.add(fra_char)
        self.in_words = len(in_words_set)+2
        self.out_words = len(out_words_set)+2
        for i, word in enumerate(in_words_set):
            self.in_word2index[word] = i+2
            self.in_index2word[i+2] = word
        for i, word in enumerate(out_words_set):
            self.out_word2index[word] = i+2
            self.out_index2word[i+2] = word
        for in_sen, out_sen in zip(train_in, train_out):
            in_sen_ids = [self.in_word2index[word] for word in in_sen]
            out_sen_ids = [self.out_word2index[word] for word in out_sen]
            self.train_pairs.append((in_sen_ids, out_sen_ids))
if __name__ == '__main__':
    model_input = ModelInput("./data/eng-fra2.txt")
    model_input.read_data()
    a = [[1,2,3], [4,5,6],[7,8,9]]
    b = [[1,2,3,4,4,5,6]]
    import numpy as np
    c = np.asarray(b)
    print(c.shape)
    print(b.shape)
    random.shuffle(a)
    random.shuffle(b)
    print(a)
    print(b)
    print("ddddd")
