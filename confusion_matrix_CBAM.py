import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchmetrics import ConfusionMatrix
from torch.utils.data import DataLoader
import torch.nn as nn
import read_data
import os
from model_with_cbam.CBAM_Jognchan_pretrained_resnet50.model_resnet import ResNet, Bottleneck

def CalculateConfusionMatrxis(model_path,save_dir):
    data_dir = r"D:\MasterKagioglou\Data\Pascal_VOC_2012\JPEGImages"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # read the txt file
    test_labels = r"D:\MasterKagioglou\Data\Pascal_VOC_2012\test_seg.txt"
    test_dataset = read_data.PASCAL_VOL2012(data_dir, test_labels, img_trans)

    batch_size = 32
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                              sampler=None)

    model = ResNet(Bottleneck, [3, 4, 6, 3], 'ImageNet', 20, 'CBAM')
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    model.eval()
    model = model.cuda()
    cm_result = torch.zeros(20, 2, 2).cuda()
    confmat = ConfusionMatrix(task="multilabel", num_labels=20).cuda()
    for i, return_dict in enumerate(test_loader):
        target = return_dict['label'].cuda(non_blocking=True)
        input_var = torch.autograd.Variable(return_dict['image']).cuda()
        target_var = torch.autograd.Variable(target)
        #target_var = target_var.float()
        output = model(input_var)
        predictions = torch.sigmoid(output)  # .squeeze()
        # Convert probabilities to binary predictions
        threshold = 0.5
        predicted_labels = (predictions >= threshold)
        cm_result_temp=confmat(predicted_labels,target_var)
        cm_result=torch.add(cm_result,cm_result_temp)

    # Perform the position swap in one line
    cm_result[:, [0, 1], [0, 1]] = cm_result[:, [1, 0], [1, 0]]
    classes_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train',
                      'tv/monitor']
    with open(os.path.join(save_dir, 'confusion_matrix.txt'), 'w') as outFile:
        for i in range(cm_result.size(0)):
            outFile.write(f"Class Label {i} Name {classes_labels[i]} \n")
            for j in range(cm_result.size(1)):
                for k in range(cm_result.size(2)):
                    outFile.write(str(cm_result[i, j, k].item()))
                    outFile.write("\t")
                outFile.write("\n")
    outFile.close()


if __name__ == '__main__':

    # Multi Label
    model_path = r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50_cbam_Jon\models\model_f1_score_ep64.pth"
    save_dir=r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50_cbam_Jon"
    CalculateConfusionMatrxis(model_path,save_dir)