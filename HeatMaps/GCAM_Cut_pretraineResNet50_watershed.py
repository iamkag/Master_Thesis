import torch
from torchvision import models
import numpy as np
import cv2
import torch.nn as nn
import requests
from pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchmetrics import Accuracy, Precision, Recall, F1Score
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import sys

sys.path.append('D:/MasterKagioglou/Pascal_VOC/')
import read_data


def get_last_conv(m, type='CBAM'):
    """
    Get the last conv layer in an Module.
    """
    convs = filter(lambda k: isinstance(k, torch.nn.Conv2d), m.modules())
    if type == 'CBAM':
        return list(convs)[-2]
    else:
        return list(convs)[-1]


def GrabCamModel(model, save_dir, test_loader, batch_size, num_of_classes):
    accuracy_score = Accuracy(task="multilabel", num_labels=num_of_classes).cuda()
    precision_score = Precision(task="multilabel", num_labels=num_of_classes, average='micro').cuda()
    recall_score = Recall(task="multilabel", num_labels=num_of_classes, average='micro').cuda()
    f1_score = F1Score(task="multilabel", num_labels=num_of_classes, average='micro').cuda()
    accuracy = []
    precision = []
    recall = []
    f1score = []
    classes_labels=['aeroplane', 'bicycle','bird', 'boat', 'bottle', 'bus', 'car' , 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
    for i, return_dict in enumerate(test_loader):
        target = return_dict['label'].cuda(non_blocking=True)
        input_var = torch.autograd.Variable(return_dict['image']).cuda()
        target_var = torch.autograd.Variable(target)
        target_var = target_var.float()
        output = model(input_var)
        predictions = torch.sigmoid(output)  # .squeeze()
        # Convert probabilities to binary predictions
        threshold = 0.5
        predicted_labels = (predictions >= threshold).float()
        accuracy.append(accuracy_score(predicted_labels, target_var))
        precision.append(precision_score(predicted_labels, target_var))
        recall.append(recall_score(predicted_labels, target_var))
        f1score.append(f1_score(predicted_labels, target_var))
        for i in range(target_var.size(0)):
            true_label = target_var[i]
            predicted_label = torch.round(predictions[i])
            img_path = return_dict['image_path'][i]
            img_cv2 = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img_cv2 = cv2.resize(img_cv2, (224, 224))
            true_indexes = [k for k, value in enumerate(true_label) if value == 1]
            true_labels=[classes_labels[m] for m in true_indexes]
            pred_index = [j for j, value in enumerate(predicted_label) if value == 1]
            predicted_classes= [classes_labels[l] for l in pred_index]
            for ind in pred_index:
                targets = [ClassifierOutputTarget(ind)]
                target_layers = [get_last_conv(model, None)]
                cam = GradCAM(model=model, target_layers=target_layers)
                image_name = return_dict['image_name'][i]
                grayscale_cams = cam(input_tensor=input_var[i].unsqueeze(0), targets=targets)

                cut_img, sure_fg, sure_bg = watershed_cut(img_path, grayscale_cams, save_dir,  image_name[:-4] + '_ind' + str(ind)+ '.png')
                cut_img = cv2.resize(cut_img, (224, 224))

                cut_img = cv2.cvtColor(cut_img, cv2.COLOR_BGR2RGB)
                img = Image.open(img_path).convert('RGB')
                img = np.array(img)
                img = cv2.resize(img, (224, 224))
                img_float = np.float32(img) / 255
                cam_image = show_cam_on_image(img_float, grayscale_cams[0, :], use_rgb=True)
                cam = np.uint8(255 * grayscale_cams[0, :])
                cam = cv2.merge([cam, cam, cam])
                #####################
                # sure_fg,sure_bg
                # Find the indices where the change occurs
                indices_fg = np.argwhere(np.diff(sure_fg.squeeze()) != 0)
                # Draw the line on the image
                for x, y in indices_fg:
                    img[x, y] = (255, 0, 0)

                indices_bg = np.argwhere(np.diff(sure_bg.squeeze()) != 0)
                # Draw the line on the image
                for x, y in indices_bg:
                    img[x, y] = (0, 0, 255)
                #####################    
                images = np.hstack((img, cam_image, cut_img))
                # images = np.hstack((np.uint8(255 * img)))
                Image.fromarray(images)
                plt.imshow(images)
                plt.title(f"True Labels: {true_labels} \n  Predicted Labels: {predicted_classes}\n current index cut {ind}:{classes_labels[ind]}")
                plt.savefig(os.path.join(save_dir, image_name[:-4] + '_ind' + str(ind) + '.png'))
                plt.clf()

    number_values = len(accuracy)

    with open(os.path.join(save_dir, 'testset_metrics.txt'), 'w') as outFile:
        outFile.write(
            f'Accuracy {sum(accuracy) / number_values} Precision {sum(precision) / number_values} Recall {sum(recall) / number_values} F1Score{sum(f1score) / number_values} \n')
    outFile.close()


def watershed_cut(img_path, heatmap, save_dir, image_name):

    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (224, 224))
   

    sure_fg = np.where(heatmap > 0.9, 255, 0).astype(np.uint8)
    sure_bg = np.where(heatmap < 0.1, 0, 255).astype(np.uint8)
    unknown = np.where((heatmap>=0.1) & (heatmap<=0.9), 255, 0).astype(np.uint8)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg.squeeze())
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unk
    markers[unknown.squeeze() == 255] = 0
    markers = cv2.watershed(image, markers)
    markers = markers.astype(np.uint8)
   
 
    # Apply GrabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mask = np.zeros(image.shape[:2],np.uint8)
    mask[markers == 2] = 0
    mask[markers == 255] = 0
    mask[markers == 1] = 1
    iter_count = 5  # Number of iterations for GrabCut
    cv2.grabCut(image, mask, None, bgd_model, fgd_model, iterCount=iter_count, mode=cv2.GC_INIT_WITH_MASK)

    # Create a mask where the probable foreground and definite foreground are set to 1
    mask_2 = np.where((mask == 1), 0, 1).astype('uint8')

    # Apply the mask to the image
    result = image * mask_2[:, :, np.newaxis]
    plt.imshow(result)
    plt.axis('off')
    plt.savefig(os.path.join(os.path.join(save_dir, 'only_cuts'), image_name), bbox_inches='tight',
                pad_inches=0, dpi=300)

    return result, sure_fg, sure_bg


def main(model_path, save_dir):
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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,sampler=None)

    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 20)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    model.eval()
    model = model.cuda()
    if not os.path.exists(os.path.join(save_dir, 'only_cuts')):
        os.makedirs(os.path.join(save_dir, 'only_cuts'))
    GrabCamModel(model, save_dir, test_loader, batch_size, 20)


if __name__ == '__main__':

    # Multi Label
    model_path = r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50\models\model_f1_score_ep89.pth"
    save_dir=r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50\watershed_sb09_sf01"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    main(model_path, save_dir)

    
