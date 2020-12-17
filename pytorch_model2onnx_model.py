"""
__project_ = 'PycharmProjects'
__file_name__ = 'pytorch_model2onnx_model'
__author__ = 'duty'
__time__ = '2020/10/26 4:03 PM'
__product_name = PyCharm
"""
import torch

from pytorch_study.character_rnn_classification import ChRnnAttCfn

import torch.onnx
import numpy as np
import onnx
import onnxruntime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def create_onnx_model():
    model = torch.load("test_model.pt")
    model.eval()
    batch_size = 1
    input_shape = 20
    export_onnx_file = "test_model.onnx"
    npx = np.random.randint(1, 100, size=(1, input_shape))
    x = torch.from_numpy(npx)
    x = x.to(device)
    # x = torch.randperm(input_shape)
    # x1 = [x]
    # torch.
    torch.onnx.export(model, x, export_onnx_file, opset_version=10, do_constant_folding=True, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input":{0:"batch_size"}, "output":{0:"batch_size"}})

def onnx_model_inference():
    onnx_model = onnx.load("test_model.onnx")
    onnx.checker.check_model(onnx_model)

class ONNXModel(object):
    def __init__(self, onnx_path):
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("intput_names:{}".format(self.input_name))
        print("output_names:{}".format(self.output_name))

    def get_input_name(self, onnx_session):
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self, onnx_session):
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_feed(self, input_name, input_tensor):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = input_tensor
        return input_feed

    def forward(self, input):
        input_seed = self.get_input_feed(self.input_name, input)
        results = self.onnx_session.run(self.output_name, input_seed)

if __name__=="__main__":
    # onnx_model_inference()
    aa = "1"
    bb = "2"
    cc = "3"
    print(aa<bb)
    print(bb<cc)
    create_onnx_model()
    model_session = onnxruntime.InferenceSession("test_model.onnx")
    npx = np.random.randint(1, 100, size=(1, 20))*2
    x = torch.from_numpy(npx)*2
    aa = model_session.get_inputs()
    bb = aa[0].name
    inputs = {model_session.get_inputs()[0].name: npx}
    outputs = model_session.run(None, inputs)
    print("dd")
