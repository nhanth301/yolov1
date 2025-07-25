import torch 
import torch.nn as nn


class Yolov1SmallQAT(nn.Module):
    def __init__(self, model, **kwargs):
        super(Yolov1SmallQAT, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x) 
        x = self.model(x)
        x = self.dequant(x) 
        return x