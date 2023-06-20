import os
import matplotlib.pyplot as plt
import numpy as np

def plot_IOS(col,save_dir):

    count, bins, _ = plt.hist(col, bins=10, edgecolor='black')

    plt.title("IOU Scores")
    plt.xlabel("Data")
    plt.ylabel("Number of Values")

    # Display count values at the center of each bar
    for i in range(len(count)):
        if count[i] != 0:
            plt.text((bins[i] + bins[i + 1]) / 2, count[i], str(int(count[i])), ha='center', va='bottom')

    save_image= os.path.join(save_dir,"IOU Scores"+"png")
    plt.savefig(save_image)
    plt.close()

def hist_per_class(data,classes,classes_name,save_dir):

    unique_classes = np.unique(classes)

    for cls in unique_classes:
        cls_data = [data[i] for i in range(len(data)) if classes[i] == cls]

        count, bins, _ = plt.hist(cls_data, bins=10, edgecolor='black')

        plt.title(f"Histogram for Class {classes_name[cls-1]}")
        plt.xlabel("Data")
        plt.ylabel("Number of Values")

        for i in range(len(count)):
            if count[i] != 0:
                plt.text((bins[i] + bins[i + 1]) / 2, count[i], str(int(count[i])), ha='center', va='bottom')

        file_name = "histogram_class_{}.png".format(cls)
        save_image= os.path.join(save_dir,file_name)
        plt.savefig(save_image)
        plt.close()


def main(iou_txt,save_dir):
    
    with open(iou_txt, 'r') as file:
        lines = file.readlines()
    file.close()
    classes_labels=['Aeroplane', 'Bicycle','Bird', 'Boat', 'Bottle', 'Bus', 'Car' , 'Cat', 'Chair', 'Cow', 'Diningtable', 'Dog', 'Horse', 'Motorbike', 'Person', 'Potted plant', 'Sheep', 'Sofa', 'Train', 'TV/Monitor']
    col1 = [line.split(',')[0] for line in lines]
    col2 = [int(line.split(',')[1]) for line in lines]
    col3 = [float(line.split(',')[2]) for line in lines]
    
    plot_IOS(col3,save_dir)
    hist_per_class(col3,col2,classes_labels,save_dir)
   
if __name__ == '__main__':

    #CBAM
    #iou_txt=r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50_cbam_Jon\watershed_sp09_sn01\only_cuts\IoUScores.txt"
    #save_dir=r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50_cbam_Jon\watershed_sp09_sn01\hist_plots"

    #save_dir = r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50_cbam_Jon\grab_cut\hist_plots"
    #iou_txt = r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50_cbam_Jon\grab_cut\only_grab_cuts\IoUScores.txt"

    #RESNET
    #save_dir = r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50\grab_cut\hist_plots"
    #iou_txt =r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50\grab_cut\only_grab_cuts\IoUScores.txt"

    save_dir = r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50\watershed_sb09_sf01\hist_plots"
    iou_txt = r"D:\MasterKagioglou\Pascal_VOC\RESULTS_resnet50\watershed_sb09_sf01\only_cuts\IoUScores.txt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    main(iou_txt,save_dir)   