import torch
import torch.nn as nn

class Basic_block(nn.Module):

    """
    Attempt to rebuild the resnet module for pure practice and grasping the process
    There is no benefit to run the custom class as the Resnet initiate already trained weigths
    resulting into less epoches to acheive convergence
    """

    def __init__(self,ch_in, ch_out, stride=1, downsample = None):
    
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride=stride, padding = 1)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2D(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1)

        self.downsample = downsample

    def forward(self, x):
    
        identity = x

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn(out)

        if self.downsample is not None: 
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Custom_Resnet(nn.module):

    """
    Attempt to rebuild the resnet module for pure practice and grasping the process
    There is no benefit to run the custom class as the Resnet initiate already trained weigths
    resulting into less epoches to acheive convergence
    """

    def __init__(self,):

        super().__init__()
        self.in_channel = 64

        # Input: [B, 3, 224, 224] -> Output: [B, 64, 56, 56]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride = 2, padding = 1)

        self.layer1 = self.build_layer(ch_out = self.in_channel ,blocks = 2 ,stride = 1) # OUT [B,64,56,56]
        self.layer2 = self.build_layer(ch_out = 128 ,blocks = 2 ,stride = 2) # OUT [B,128,28,28]
        self.layer3 = self.build_layer(ch_out = 256 ,blocks = 2 ,stride = 2) # OUT [B,256,14,14]
        self.layer4 = self.build_layer(ch_out = 512 ,blocks = 2 ,stride = 2) # OUT [B,512,7,7]


    def build_layer(self, ch_out, blocks, stride):
        downsample = None

        if stride != 1 or self.in_channel != ch_out :
            downsample = nn.Sequential(
                nn.Conv2D(self.in_channel,ch_out, kernel_size = 1, stride = stride, biais = False),
                nn.BatchNorm2d(ch_out)
            )
        
        layers = []
        layers.append(Basic_block(self.in_channels, ch_out, stride, downsample))
        self.in_channels = ch_out

        for _ in range(1,blocks):
            layers.append(Basic_block(self.in_channel, ch_out,kernel_size = 1,stride=1, downsample=None))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out
