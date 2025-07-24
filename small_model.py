import torch
from torch import nn

#[kernel_size, out_c, stride, padding]
architecture_config_small = [
    (3, 32, 1, 1),  # Convolutional block 1
    "M",            # Max-pooling layer 1
    (3, 64, 1, 1),  # Convolutional block 2
    "M",            # Max-pooling layer 2
    (3, 128, 1, 1), # Convolutional block 3
    "M",            # Max-pooling layer 3
    [(1, 64, 1, 0), (3, 128, 1, 1), 1],  # Convolutional block 4 (repeated 1 time)
    "M",            # Max-pooling layer 4
    (3, 256, 1, 1), # Convolutional block 5
    (3, 256, 2, 1), # Convolutional block 6 (downsample)
    (3, 256, 1, 1), # Convolutional block 7
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1Small(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1Small, self).__init__()
        self.architecture = architecture_config_small
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        
        dummy_input = torch.randn(1, self.in_channels, 448, 448)
        with torch.no_grad():
            darknet_output_size = self.darknet(dummy_input).shape
        
        self.flattened_features_dim = darknet_output_size[1] * darknet_output_size[2] * darknet_output_size[3]
        
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)
    
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_features_dim, 256), 
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(256, S * S * (C + B * 5))
        )
    
if __name__ == '__main__':
    from torchsummary import summary
    model_small = Yolov1Small(split_size=7, num_boxes=2, num_classes=20).to('cuda')
    print(summary(model_small, (3, 448, 448)))
    x = torch.randn(1, 3, 448, 448).to('cuda')
    out = model_small(x)
    print(out.shape)