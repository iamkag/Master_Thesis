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
from model_with_cbam.CBAM_Jognchan_pretrained_resnet50.model_resnet import ResNet, Bottleneck

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
                target_layers = [get_last_conv(model, 'CBAM')]
                cam = GradCAM(model=model, target_layers=target_layers)
                image_name = return_dict['image_name'][i]
                grayscale_cams = cam(input_tensor=input_var[i].unsqueeze(0), targets=targets)

                cut_img = GrabCut(img_path, grayscale_cams, save_dir,  image_name[:-4] + '_ind' + str(ind)+ '.png')
                cut_img = cv2.resize(cut_img, (224, 224))

                cut_img = cv2.cvtColor(cut_img, cv2.COLOR_BGR2RGB)
                img = Image.open(img_path).convert('RGB')
                img = np.array(img)
                img = cv2.resize(img, (224, 224))
                img = np.float32(img) / 255
                cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
                cam = np.uint8(255 * grayscale_cams[0, :])
                cam = cv2.merge([cam, cam, cam])
                images = np.hstack((np.uint8(255 * img), cam_image, cut_img))
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


def compute_rect(heatmap,threshold):
    heatmap_thresh = np.where(heatmap > threshold, 1, 0).astype('uint8')
    heatmap_thresh = heatmap_thresh.squeeze().astype('uint8')
    # Find contours
    contours, _ = cv2.findContours(heatmap_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    # Create bounding boxes and draw them on the image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rects.append((x, y, w, h))
        # cv2.rectangle(img_rgb, (x, y, w, h), (0, 255, 0), 2)

    if not rects:
        rect = [6, 6, 220, 220]
    else:
        # Find the top-left corner of the new rectangle
        x = min(rect[0] for rect in rects)
        y = min(rect[1] for rect in rects)

        # Find the bottom-right corner of the new rectangle
        w = max(rect[0] + rect[2] for rect in rects) - x
        h = max(rect[1] + rect[3] for rect in rects) - y

        x = x if x != 0 else 6
        y = y if y != 0 else 6
        w = w if w != 224 else 220
        h = h if h != 224 else 220
        # Define the new rectangle
        rect = (x, y, w, h)

    return rect

def GrabCut(img_path,heatmap,save_dir,image_name):

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_new = cv2.imread(img_path)
    img_new = cv2.resize(img_new, (224, 224))
    img_new = cv2.cvtColor(img_new, cv2.COLOR_RGB2BGR)

    
    img_rgb_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
    mask_new=np.zeros(img_rgb_new.shape[:2], np.uint8)

    ###############################################
    mask = np.zeros(img_rgb.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    ############################################
    # Threshold the heatmap
    threshold = 0.6
    rect=compute_rect(heatmap, threshold)

    cv2.grabCut(img_rgb, mask, rect, bgdModel, fgdModel, 20, cv2.GC_INIT_WITH_RECT)


    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    count_pixels = np.sum(mask2 == 1)


    if count_pixels<100:
        threshold = 0.8
        rect = compute_rect(heatmap, threshold)
        mask_black=np.zeros_like(img_rgb)
        cv2.rectangle(mask_black,(rect[0],rect[1]),(rect[2]+rect[0],rect[3]+rect[1]),(255,255,255),-1)
        img1 = cv2.bitwise_and(img_rgb,mask_black)

    else:
        img1 = img_rgb * mask2[:, :, np.newaxis]


    plt.imshow(img1)
    plt.axis('off')
    plt.savefig(os.path.join(os.path.join(save_dir, 'only_grab_cuts'), image_name), bbox_inches='tight', pad_inches=0, dpi=300)
    cv2.rectangle(img1,(rect[0],rect[1]),(rect[2]+rect[0],rect[3]+rect[1]),(0, 255, 0), 2)
    
    return img1


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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                             sampler=None)

    model = ResNet(Bottleneck, [3, 4, 6, 3], 'ImageNet', 20, 'CBAM')
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    model.eval()
    model = model.cuda()
    if not os.path.exists(os.path.join(save_dir, 'only_grab_cuts')):
        os.makedirs(os.path.join(save_dir, 'only_grab_cuts'))
    GrabCamModel(model, save_dir, test_loader, batch_size, 20)


if __name__ == '__main__':

    # Multi Label
    model_path = r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50_cbam_Jon\models\model_f1_score_ep64.pth"
    save_dir=r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50_cbam_Jon\grab_cut"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    main(model_path, save_dir)