import torch
import torchvision

from PIL import Image

from models import get_model
from utils import processImg
from agent import ClassifierAgent


class Agent:
    def __init__(self, backbone, model_path):
        torch.backends.cudnn.benchmark = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.classifier = get_model(backbone).to(self.device)

        if model_path:
            self.classifier.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))

    def classify_img(self, img_path):
        self.classifier.eval()

        raw_img = Image.open(img_path)
        img = processImg(raw_img, resize=True)

        img = torchvision.transforms.ToTensor()(img)

        if len(img.shape) < 3:
            img = img.unsqueeze(0)

        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.classifier(img)

        if preds >= .7:
            return 'Adult'
        else:
            return 'Not Adult'


if __name__ == '__main__':
    train_anno = r"./data/UTKFace/val_annotation.csv"
    test_anno = r"./data/UTKFace/test_annotation.csv"

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    backbone = 'classifier'
    exp_name = 'exp2'
    model_path = r'./experiments/exp3/model_state_dict.pth'

    agent = ClassifierAgent(train_anno, test_anno, exp_name, backbone, load_model=model_path, device=device)
    _, acc, prec, recall, f1 = agent.evaluate()
    print('Accuracy: ', acc)
    print('Precision: ', prec)
    print('Recall: ', recall)
    print('F1 score: ', f1)

    model_agent = Agent(backbone=backbone, model_path=model_path)

    img_path = r"C:\Users\Cosmin\Desktop\franci_2012.jpg"
    prediction = model_agent.classify_img(img_path)
    print(prediction)
