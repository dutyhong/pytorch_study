"""
__project_ = 'PycharmProjects'
__file_name__ = 'text_brewer_test'
__author__ = 'duty'
__time__ = '2020/6/18 2:11 PM'
__product_name = PyCharm
"""
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("hfl/rbt3")
model = BertModel.from_pretrained("hfl/rbt3")
params = model.parameters()
for param in params:
    print(type(param.data), param.size())
named_params = model.named_parameters()
for name, param in named_params:
    print(name,"  ", param.size())
print("ddd")