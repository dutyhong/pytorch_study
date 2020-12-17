

## 读取数据
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel, BertForPreTraining
from transformers.modeling_tf_bert import TFBertForSequenceClassification
from pytorch_study.character_rnn_classification import ChRnnAttCfn, ChRnnCfn
from pytorch_study.data_process import get_train_test_data
device='cuda:0' if torch.cuda.is_available() else 'cpu'
#
# config = BertConfig.from_json_file('/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json')
# model = BertForPreTraining.from_pretrained('/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt.index', from_tf=True, config=config)
# # 将所有模型参数转换为一个列表
# params = list(model.named_parameters())
#
# print('The BERT model has {:} different named parameters.\n'.format(len(params)))
#
# print('==== Embedding Layer ====\n')
#
# for p in params[0:5]:
#     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
#
# print('\n==== First Transformer ====\n')
#
# for p in params[5:21]:
#     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
#
# print('\n==== Output Layer ====\n')
#
# for p in params[-4:]:
#     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
#
# model.save_pretrained("/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/hg_tf/")
# # model = BertForSequenceClassification.from_pretrained("/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt.index", from_pt=True, )
# model = BertForSequenceClassification.from_pretrained(
#     "/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/hg_tf/",
#     num_labels = 2,
#
#     output_attentions = False,
#     output_hidden_states = False
# )
# # 将所有模型参数转换为一个列表
# params = list(model.named_parameters())
#
# print('The BERT model has {:} different named parameters.\n'.format(len(params)))
#
# print('==== Embedding Layer ====\n')
#
# for p in params[0:5]:
#     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
#
# print('\n==== First Transformer ====\n')
#
# for p in params[5:21]:
#     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
#
# print('\n==== Output Layer ====\n')
#
# for p in params[-4:]:
#     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

train_sents, test_sents, val_sents, vocabs = get_train_test_data()
# tokenizer = BertTokenizer.from_pretrained("/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt")
tokenizer = BertTokenizer.from_pretrained("/Data/public/Bert/bert_base_chinese_pytorch/")
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


model = BertForSequenceClassification.from_pretrained(
    # "/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/hg_tf/",
    "/Data/public/Bert/bert_base_chinese_pytorch/",
    # "bert-base-chinese",# Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False# Whether the model returns all hidden-states.
)
model.to(device)
x_train = []
y_train = []
x_test = []
y_test = []
x_val = []
y_val = []
max_seq_length=20
for label, sent in train_sents:
    sent_ids = tokenizer.encode(sent, add_special_tokens=True, max_length=max_seq_length)
    x_train.append(sent_ids)
    y_train.append(label)
for label, sent in test_sents:
    sent_ids = tokenizer.encode(sent,  add_special_tokens=True, max_length=max_seq_length)
    x_test.append(sent_ids)
    y_test.append(label)
for label, sent in val_sents:
    sent_ids = tokenizer.encode(sent,  add_special_tokens=True, max_length=max_seq_length)
    x_val.append(sent_ids)
    y_val.append(label)

# sentences = sentences[:,1:10]

# input_ids = [tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN) for sent in sentences]
# test_input_ids=[tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN) for sent in test_sentences]

from keras.preprocessing.sequence import pad_sequences
print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

train_input_ids = pad_sequences(x_train, maxlen=max_seq_length, dtype="long",
                          value=0, truncating="post", padding="post")

test_input_ids = pad_sequences(x_test, maxlen=max_seq_length, dtype="long",
                          value=0, truncating="post", padding="post")
val_input_ids = pad_sequences(x_test, maxlen=max_seq_length, dtype="long",
                          value=0, truncating="post", padding="post")

# Create attention masks
x_train_attention_masks = []

# For each sentence...
for sent in train_input_ids:
    # Create the attention mask.
    #   - If a token ID is 0, then it's padding, set the mask to 0.
    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
    att_mask = [int(token_id > 0) for token_id in sent]

    # Store the attention mask for this sentence.
    x_train_attention_masks.append(att_mask)

x_test_attention_masks = []

# For each sentence...
for sent in test_input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    x_test_attention_masks.append(att_mask)
import numpy as np
##转换为tensor
train_input_ids = torch.tensor(train_input_ids, dtype=torch.long)
y_train = torch.tensor(y_train,  dtype=torch.long)
test_input_ids = torch.tensor(test_input_ids,dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
val_input_ids = torch.tensor(val_input_ids,dtype=torch.long)
y_val = torch.tensor(y_val,  dtype=torch.long)

x_train_attention_masks = torch.Tensor(np.asarray(x_train_attention_masks))
x_test_attention_masks = torch.Tensor(np.asarray(x_test_attention_masks))


# hidden_size = 10
# num_layers = 1
# n_categories = 2
# embed_dim = 64
# max_seq_len = 20
# batch_size = 32
# bidiretional = False
# vocab_size = tokenizer.vocab_size
# model = ChRnnCfn(hidden_size, vocab_size, embed_dim, num_layers=num_layers)
# model.to(device)
batch_size = 32
# Create the DataLoader for our training set.
train_data = TensorDataset(train_input_ids, x_train_attention_masks, y_train)
# train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, batch_size=batch_size)

# ##开始训练
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
epochs = 5
loss_fn= torch.nn.CrossEntropyLoss()
for i in range(epochs):
    for step, batch in enumerate(train_dataloader):
        input_ids = batch[0].to(device)
        input_masks = batch[1].to(device)
        labels = batch[2].to(device)
        # log_probs = model(input_ids)
        # loss = loss_fn(log_probs, labels)
        outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_masks, labels=labels)
        loss = outputs[0]
        logits = outputs[1]
        print("第%d个epoch，第%d个batch的损失为%f"%(i, step, loss.item()))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        for name, params in model.named_parameters():
            print('-->name:', name, '-->grad_requires:',params.requires_grad, \
		 ' -->grad_value:',params.grad)
# torch.save(model.state_dict(), "BertFinetune.model")

# model = BertForSequenceClassification.from_pretrained(
#     # "/Data/public/Bert/bert_base_chinese_pytorch/", # Use the 12-layer BERT model, with an uncased vocab.
#     "bert-base-chinese",
#     num_labels = 2, # The number of output labels--2 for binary classification.
#                     # You can increase this for multi-class tasks.
#     output_attentions = False, # Whether the model returns attentions weights.
#     output_hidden_states=False)
# model.load_state_dict(torch.load("BertFinetune.model"))
# model.to(device)
##测试结果
def evaluate(test_data, model):
    model.eval()
    with torch.no_grad():
        input_ids = test_data[0].to(device)
        input_masks = test_data[1].to(device)
        labels = test_data[2].to(device)
        total_cnt = len(labels)
        # logits = []
        # for i in range(total_cnt):
        outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_masks)
        logits = outputs[0]
        # logits.append(logit.detach().cpu().numpy())
    cnt = 0
    for i in range(total_cnt):
        if torch.argmax(logits[i])==labels[i]:
            cnt = cnt +1
    print("准确率为：%f"%(cnt/total_cnt))
test_data = (test_input_ids, x_test_attention_masks, y_test)
evaluate(test_data, model)
print("ddd")