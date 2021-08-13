import os
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from models import get_model
from dataset import datasetLoader
from utils import compute_class_weights


class ClassifierAgent:
    def __init__(self, train_anno, test_anno, exp_name, backbone, load_model=None, lr=1e-4, batch_size=16, dropout=0.3,
                 shuffle=True, train=False, device='cpu'):
        self.exp_name = exp_name
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

        # Create loaders for training and testing datasets
        self.train_loader, dataset = datasetLoader(train_anno, batch_size=self.batch_size, shuffle=shuffle, device=self.device)
        self.test_loader, _ = datasetLoader(test_anno, batch_size=self.batch_size, shuffle=shuffle, device=self.device)

        # Compute the class weights in case of unbalanced data
        weights_dict = compute_class_weights(dataset.dict_numbers)

        # Create tensor containing class weights
        self.class_weights = torch.tensor(list(weights_dict.values())).to(self.device)
        print("Class weights: ", self.class_weights)

        # Initialize neural model and training optimizer
        self.classifier = get_model(backbone, dropout_rate=dropout, train=train).to(self.device)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr, weight_decay=1e-5)

        # self.loss_fn = nn.BCELoss(weight=self.class_weights)

        # Load model if path is given as argument
        if load_model:
            self.classifier.load_state_dict(torch.load(load_model, map_location=torch.device(self.device)))

        # Set experiment's directory path
        self.writer_dir = os.path.join(r"./experiments", exp_name)

        # Check if the path already exists
        if os.path.exists(self.writer_dir) is False:
            # Create directory at the given path
            os.mkdir(self.writer_dir)

            # Create Tensorboard Summary for the current experiment
            self.writer = SummaryWriter(self.writer_dir, flush_secs=60)

    def accuracy(self, y_pred, y_test, threshold=0.5):
        '''
        Compute model's accuracy for the actual batch
        :param y_pred: batch of the model's prediction. Tensor of size [batch, 1]
        :param y_test: batch of the correct labels. Tensor of size [batch, 1]
        :param threshold: threshold value for valid predictions (Float).
        :return: The model's accuracy for the current batch (Float).
        '''
        preds = y_pred

        # Change predictions to 1 and 0 according to the threshold value
        preds[torch.where(preds >= threshold)] = 1
        preds[torch.where(preds < threshold)] = 0

        # Count the correct predictions
        correct_sum = (preds == y_test).sum().float()

        # Divide the number of correct predictions by the total number of predictions
        acc = correct_sum / preds.size(0)
        acc = torch.round(acc * 100)

        return acc

    def perf_metrics(self, y_pred, y_test, threshold=0.5):
        '''
        Compute model's performance for the current batch
        :param y_pred: batch of the model's prediction. Tensor of size [batch, 1]
        :param y_test: batch of the correct labels. Tensor of size [batch, 1]
        :param threshold: threshold value for valid predictions (Float).
        :return: Precision, Recall and F1 score for the current batch
        '''
        preds = y_pred

        # Change predictions to 1 and 0 according to the threshold value
        preds[torch.where(preds >= threshold)] = 1
        preds[torch.where(preds < threshold)] = 0

        # Count the number of True Positives
        tp = (preds * y_test).sum().float()

        # Compute precision by dividing TP to the total number of model's predictions that are 1
        prec = tp / (preds.sum() + 1e-8)

        # Compute recall by dividing TP to the total number of actual case that are 1
        recall = tp / (y_test.sum() + 1e-8)

        # Compute F1 score with the help of precision and recall
        f1 = 2 * prec * recall / (prec + recall + 1e-8)

        return torch.round(prec * 100), torch.round(recall * 100), torch.round(f1 * 100)

    def train(self, num_epochs):
        print("Training Started...")
        train_length = len(self.train_loader)

        for epoch in range(num_epochs + 1):

            # Set neural model to train mode
            self.classifier.train()

            train_loss = 0
            train_acc = 0
            train_prec = 0
            train_recall = 0
            train_f1 = 0

            for imgs, labels in self.train_loader:

                # Set weight for every case in the batch according to its label
                weight = torch.ones_like(labels).to(self.device)
                for idx in range(labels.size(0)):
                    weight[idx, 0] = self.class_weights[0] if labels[idx, 0] == 0 else self.class_weights[1]

                # Make predictions using neural model
                preds = self.classifier(imgs)

                self.optimizer.zero_grad()
                # loss = self.loss_fn(preds, labels.float())

                # Compute loss wrt to the class weights
                loss = torch.nn.functional.binary_cross_entropy(preds, labels.float(), weight=weight)

                # Send loss back into the neural architecture
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                # Compute accuracy for the current batch
                acc = self.accuracy(preds, labels)
                train_acc += acc.item()

                # Compute perfomance metrics for the current batch
                prec, recall, f1 = self.perf_metrics(preds, labels)
                train_prec += prec.item()
                train_recall += recall.item()
                train_f1 += f1.item()

            train_loss /= train_length
            train_acc /= train_length
            train_prec /= train_length
            train_recall /= train_length
            train_f1 /= train_length

            # Evaluate model at the end of every epoch
            val_loss, val_acc, val_prec, val_recall, val_f1 = self.evaluate()

            print("Epoch: {}/{} | Train Loss: {:.4f} | Train Acc: {:.4f} | Train Prec : {:.4f} | Train Recall: {:.4f} "
                  "| Train F1: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.4f} | Val Prec : {:.4f} | Val Recall: {:.4f} | "
                  "Val F1: {:.4f}".format(epoch, num_epochs, train_loss, train_acc, train_prec, train_recall, train_f1,
                                          val_loss, val_acc, val_prec, val_recall, val_f1))

            # Write values into the Tensorboard Summary
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/acc", train_acc, epoch)
            self.writer.add_scalar("train/prec", train_prec, epoch)
            self.writer.add_scalar("train/recall", train_recall, epoch)
            self.writer.add_scalar("train/f1", train_f1, epoch)

            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.writer.add_scalar("val/acc", val_acc, epoch)
            self.writer.add_scalar("val/prec", val_prec, epoch)
            self.writer.add_scalar("val/recall", val_recall, epoch)
            self.writer.add_scalar("val/f1", val_f1, epoch)

            # Save model after every epoch
            torch.save(self.classifier.state_dict(), r"./{}/model_state_dict.pth".format(self.writer_dir))

            # Early stopping according to the F1 score
            if val_f1 > 95.:
                break

    def evaluate(self):

        # Set neural model to evaluation mode to disable batch normalization and dropout
        self.classifier.eval()
        val_length = len(self.test_loader)

        val_loss, val_acc, val_prec, val_recall, val_f1 = 0, 0, 0, 0, 0

        for imgs, labels in self.test_loader:

            # Don't compute grads for evaluation
            with torch.no_grad():

                # Make predicitons using the neural model
                preds = self.classifier(imgs)

                val_loss += torch.nn.functional.binary_cross_entropy(preds, labels.float()).item()
                val_acc += self.accuracy(preds, labels).item()
                prec, recall, f1 = self.perf_metrics(preds, labels)
                val_prec += prec.item()
                val_recall += recall.item()
                val_f1 += f1.item()

        return val_loss / val_length, val_acc / val_length, val_prec / val_length, val_recall / val_length, val_f1 / val_length



