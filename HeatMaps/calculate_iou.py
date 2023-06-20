import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from PIL import Image


def calculateIoU(gtMask, predMask):
    # Calculate the true positives,
    # false positives, and false negatives
    tp = 0
    fp = 0
    fn = 0

    for i in range(len(gtMask)):
        for j in range(len(gtMask[0])):
            if gtMask[i][j] == 1 and predMask[i][j] == 1:
                tp += 1
            elif gtMask[i][j] == 0 and predMask[i][j] == 1:
                fp += 1
            elif gtMask[i][j] == 1 and predMask[i][j] == 0:
                fn += 1

    # Calculate IoU
    iou = tp / (tp + fp + fn)

    return iou


def main(test_txt, gtmaskPath, predmaskPath):
    list_with_cutted_image = os.listdir(predmaskPath)

    with open(test_txt, 'r') as file:
        lines = file.readlines()
    file.close()

    for line in lines:
        file_name = line.split(' ')[0]
        seg_image = os.path.join(gtmaskPath, file_name)
        seg_image = Image.open(seg_image)
        seg_image = seg_image.resize((224, 224))
        seg_array = np.array(seg_image)
        tmp_labels = np.unique(seg_array)
        values_to_remove = [0, 255]
        labels = [x for x in tmp_labels if x not in values_to_remove]

        for label in labels:
            temp_true_mask = np.where((seg_array == label), 1, 0).astype('uint8')
            predicted_mask_name = file_name[:-4] + '_ind' + str(label-1) + '.png'

            found = any(predicted_mask_name in item for item in list_with_cutted_image)
            if found:
                list_with_cutted_image = [x for x in list_with_cutted_image if x != predicted_mask_name]
                predicted_image = Image.open(os.path.join(predmaskPath, predicted_mask_name))
                pred_mask = predicted_image.resize((224, 224))
                pred_mask = pred_mask.convert("L")
                pred_mask = np.array(pred_mask)
                # Convert the image array to a 2D array
                pred_mask[pred_mask != 0] = 1
                iou2 = np.round(calculateIoU(temp_true_mask, pred_mask), 3)
            else:
                iou2 = -1

            with open(os.path.join(predmaskPath, 'IoUScores.txt'), 'a') as out_file:
                out_file.write(f'{file_name[:-4]}_ind, {label} , {iou2}\n')
                print(file_name, iou2)
            out_file.close()


if __name__ == '__main__':
    test_txt = r"D:\MasterKagioglou\Data\Pascal_VOC_2012\test_seg.txt"
    gtmaskPath = r"D:\MasterKagioglou\Data\Pascal_VOC_2012\SegmentationClass"
    #predmaskPath = r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50_cbam_Jon\watershed_sp09_sn01\only_cuts"
    #predmaskPath = r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50_cbam_Jon\grab_cut\only_grab_cuts"
    #predmaskPath = r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50\grab_cut\only_grab_cuts"
    predmaskPath = r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50\watershed_sb09_sf01\only_cuts"


    main(test_txt, gtmaskPath, predmaskPath)
