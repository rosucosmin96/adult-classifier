import sys
import torch

from agent import ClassifierAgent

if __name__ == "__main__":

    # Set path of annotation files
    train_anno = r"./data/UTKFace/train_annotation.csv"
    test_anno = r"./data/UTKFace/val_annotation.csv"

    # Set device according to the hardware
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training device: ", device)

    # Set neural network architecture
    backbone = 'classifier'

    # Name of the experiment's directory
    exp_name = 'exp3'

    # Create agent to train
    agent = ClassifierAgent(train_anno, test_anno, exp_name, backbone, device=device)

    # Train neural model
    agent.train(30)


