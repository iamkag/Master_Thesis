import torchvision.models as models
from model_resnet import ResNet, Bottleneck
import torch
import torch.nn as nn
import collections


def jogn_checkpoint():
    resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], 'test', 1000, 'CBAM')
    checkpoint = torch.load(r'D:\MasterKagioglou\Pedestrian_Dataset\model_with_cbam\CBAM_Jognchan_pretrained_resnet50\RESNET50_CBAM_new_name_wrap.pth')
    new_state_dict = collections.OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove the "module." prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    resnet50.load_state_dict(new_state_dict)
    resnet50.fc = nn.Linear(2048, 101)


def main():
    resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], 'test', 1000, 'CBAM')
    checkpoint = torch.load(r"C:\Users\kmixi\.cache\torch\hub\checkpoints\resnet50-0676ba61.pth")
    for key in resnet50.state_dict():
        if key not in checkpoint:
            checkpoint.update({key: torch.randn(resnet50.state_dict()[key].shape)})
        print(key)

    resnet50.load_state_dict(checkpoint)
    resnet50.fc = nn.Linear(2048, 101)
    print(resnet50.state_dict())


if __name__ == '__main__':
    jogn_checkpoint()
