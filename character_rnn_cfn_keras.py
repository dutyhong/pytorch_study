from keras import Input, Model
from keras.layers import Embedding, LSTM, Dense, Softmax
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

from pytorch_study.data_process import get_train_test_data, get_robot_train_test_data


class ChRnnCfnKeras(object):
    def __init__(self, max_seq_len, vocab_size, embed_dim, hidden_size, n_categories, units):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_categories = n_categories
        self.units = units
    def get_model(self):
        input = Input(shape=(self.max_seq_len, ))
        embed_out = Embedding(self.vocab_size, self.embed_dim)(input)
        lstm_out = LSTM(self.units)(embed_out)
        dense_out = Dense(self.n_categories)(lstm_out)
        softmax_out = Softmax()(dense_out)
        model = Model(inputs=input, outputs=softmax_out)
        model.summary()
        model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy',metrics=['accuracy'])
        return model


train_data, test_data, val_data, vocabs = get_robot_train_test_data() #get_train_test_data()
vocab_size = len(vocabs)
word_to_index = {}
index_to_word = {}
for i, char in enumerate(vocabs):
    word_to_index[char] = i
    index_to_word[i] = char

max_seq_len = 20
embed_dim = 32
hidden_size = 1
units = 10
n_categories = 5
ch_rnn_cfn_keras = ChRnnCfnKeras(max_seq_len, vocab_size, embed_dim, hidden_size, n_categories, units)
model = ch_rnn_cfn_keras.get_model()

##处理数据
# x_train = []
# y_train = []
def trans_chars_ids(input_data):
    x = []
    y = []
    for i, sample in enumerate(input_data):
        label = sample[0]
        text = sample[1]
        xx = []
        for ch in text:
            xx.append(word_to_index[ch])
        y.append(label)
        x.append(xx)
    return x, y
x_train, y_train = trans_chars_ids(train_data)
x_test, y_test = trans_chars_ids(test_data)
x_train = pad_sequences(x_train, max_seq_len,padding="post", truncating="post")
y_train = to_categorical(y_train, num_classes=n_categories)
x_test = pad_sequences(x_test, max_seq_len, padding="post", truncating="post")
y_test = to_categorical(y_test, n_categories)

model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
