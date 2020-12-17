"""
__project_ = 'PycharmProjects'
__file_name__ = 'electra_model'
__author__ = 'duty'
__time__ = '2020/9/28 11:18 AM'
__product_name = PyCharm
"""
from transformers import ElectraTokenizer, ElectraModel


tokenizer = ElectraTokenizer.from_pretrained("/Users/duty/publicdata/chinese_electra_small_discriminator_pytorch")
electra_model = ElectraModel.from_pretrained("/Users/duty/publicdata/chinese_electra_small_discriminator_pytorch")
inputs = tokenizer("我这边是车易拍啊", return_tensors="pt")
outputs = electra_model(**inputs, output_hidden_states=True)

print("ddd")


