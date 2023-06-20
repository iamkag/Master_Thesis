import torch
import torch.nn as nn
import torchvision.models as models

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(self.conv1(out) * x)
        return out

class ResNet50_CBAM(nn.Module):
    def __init__(self,num_classes):
        super(ResNet50_CBAM, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)
        self.num_classes = num_classes
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.cbam1(x)

        x = self.resnet.layer2(x)
        x = self.cbam2(x)

        x = self.resnet.layer3(x)
        x = self.cbam3(x)

        x = self.resnet.layer4(x)
        x = self.cbam4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)

        return x


if __name__ == '__main__':
    model = ResNet50_CBAM()
    print(model)


