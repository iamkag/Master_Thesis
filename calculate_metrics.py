import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score

class Metrics_model(object):
    def __init__(self,num_of_classes):

        self.num_of_classes=num_of_classes
        
        # Initialize metrics
        self.accuracy = Accuracy(task="multilabel",num_labels=self.num_of_classes,average='macro').cuda()
        self.precision = Precision(task="multilabel",num_labels=self.num_of_classes,average='macro').cuda()
        self.recall = Recall(task="multilabel",num_labels=self.num_of_classes,average='macro').cuda()
        self.f1 = F1Score(task="multilabel",num_labels=self.num_of_classes,average='macro').cuda()

        self.training_loss_list = []
        self.training_acc_list = []
        self.training_prec_list = []
        self.training_recall_list = []
        self.training_f1_list = []

        self.val_loss_list = []
        self.val_acc_list = []
        self.val_prec_list = []
        self.val_recall_list = []
        self.val_f1_list = []

    def updateTrainList(self, curr_loss, curr_accuracy, curr_prec, curr_recall, curr_f1score):
        self.training_loss_list.append(curr_loss)
        self.training_acc_list.append(curr_accuracy)
        self.training_prec_list.append(curr_prec)
        self.training_recall_list.append(curr_recall)
        self.training_f1_list.append(curr_f1score)

    def getTrainList(self):
        return self.training_loss_list, self.training_acc_list, self.training_prec_list, self.training_recall_list, self.training_f1_list

    def updateValidationList(self, curr_loss, curr_accuracy, curr_prec, curr_recall, curr_f1score):
        self.val_loss_list.append(curr_loss)
        self.val_acc_list.append(curr_accuracy)
        self.val_prec_list.append(curr_prec)
        self.val_recall_list.append(curr_recall)
        self.val_f1_list.append(curr_f1score)

    def getValidationList(self):
        return self.val_loss_list, self.val_acc_list, self.val_prec_list, self.val_recall_list, self.val_f1_list

    def calculateMetrics(self, preds, true_labels):
        # Convert probabilities to binary predictions
        threshold = 0.5
        predicted_labels = (preds >= threshold).float()

        batch_accuracy = self.accuracy(predicted_labels, true_labels)
        batch_precision = self.precision(predicted_labels, true_labels)
        batch_recall = self.recall(predicted_labels, true_labels)
        batch_f1_score = self.f1(predicted_labels, true_labels)

        
        return batch_accuracy, batch_precision, batch_recall, batch_f1_score
