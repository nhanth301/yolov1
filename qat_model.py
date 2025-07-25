import torch 
import torch.nn as nn
from small_model import CNNBlock

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

def fuse_original_model_for_qat(model):
    
    if hasattr(model, 'darknet') and isinstance(model.darknet, nn.Sequential):
        for i, sub_module in enumerate(model.darknet):
            if isinstance(sub_module, CNNBlock):
                torch.quantization.fuse_modules(sub_module, ['conv', 'batchnorm'], inplace=True)
    return model 