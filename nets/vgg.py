import torch
import torch.nn as nn
from torchvision import models

class VGG16_LargeFOV(nn.Module):
    def __init__(self, num_classes=21, input_size=321, split='train', init_weights=True):
        super(VGG16_LargeFOV, self).__init__()
        self.input_size = input_size
        self.split = split
        self.features = nn.Sequential(
            ### conv1_1 conv1_2 maxpooling
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ### conv2_1 conv2_2 maxpooling
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ### conv3_1 conv3_2 conv3_3 maxpooling
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),


            ### conv4_1 conv4_2 conv4_3 maxpooling(stride=1)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            ### conv5_1 conv5_2 conv5_3 (dilated convolution dilation=2, padding=2)
            ### maxpooling(stride=1)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ### average pooling
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),

            ### fc6 relu6 drop6
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.ReLU(True),
            nn.Dropout2d(0.5),

            ### fc7 relu7 drop7 (kernel_size=1, padding=0)
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Dropout2d(0.5),

            ### fc8
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0)
        )
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        output = self.features(x)
        if self.split == 'test':
            output = nn.functional.interpolate(output, size=(self.input_size, self.input_size), mode='bilinear', align_corners=True)
        return output
    
    def _initialize_weights(self):
        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0] == 'features.38':
                    nn.init.normal_(m[1].weight.data, mean=0, std=0.01)
                    nn.init.constant_(m[1].bias.data, 0.0)
            

if __name__ == "__main__":
    model = VGG16_LargeFOV()
    x = torch.ones([2, 3, 321, 321])
    y = model(x)
    print(y.shape)