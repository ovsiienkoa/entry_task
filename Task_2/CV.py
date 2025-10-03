import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from torchvision.io import decode_image
from torchvision import models

from torch.utils import data
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# default SimCLR augmentation (code from BYOL, because I like it)
import random
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class CVDataset(torch.utils.data.Dataset):

    label_to_index = { #in comments imagenet ids
            'butterfly':0, #322
             'cat':1, #too many different cars
             'chicken':2, #7 cock
             'cow':3,
             'dog':4,
             'elephant':5, # 385 indian elephant
             'horse':6,
             'sheep':7,
             'spider':8,
             'squirrel':9
        }

    DEFAULT_AUG = torch.nn.Sequential(
        RandomApply(
            T.ColorJitter(0.8, 0.8, 0.8, 0.2),
            p=0.3
        ),
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        RandomApply(
            T.GaussianBlur((3, 3), (1.0, 2.0)),
            p=0.2
        )
    )

    def __init__(self, folder_path:str, eval_size:float, test_size:float, connector:T):
        super().__init__()
        self.data = []
        self.targets = []
        self.train_index = None
        self.test_index = None
        self.eval_index = None


        for folder in os.listdir(folder_path):

            target = torch.zeros(10)
            target[CVDataset.label_to_index[folder]] = 1
            expected_number_of_files = len(os.listdir(os.path.join(folder_path, folder))[:600]) * 2#*2 because we create to copies with different augs #[:600] only because I have 16 gb of ram
            errors_counter = 0

            for file in os.listdir(f"{folder_path}/{folder}")[:600]: #[:600] - 16 gb of ram

                image_tensor = decode_image(f"{folder_path}/{folder}"+"/"+file)
                try:
                    self.data.append(connector(CVDataset.DEFAULT_AUG(image_tensor[:3, :, :]))) #[:3], because png pics have 4 channels and the last one is redundant
                    self.data.append(connector(CVDataset.DEFAULT_AUG(image_tensor[:3, :, :])))
                except RuntimeError:
                    errors_counter +=1
                    print(f"file, folder: {file}, {folder}")

            target = target.expand(expected_number_of_files - 2*errors_counter, 10)
            self.targets.extend(target)

        self.data = torch.stack(self.data, dim = 0)
        self.targets = torch.stack(self.targets, dim = 0)

        self.train_index, self.eval_index = train_test_split(np.arange(0, len(self.targets)), test_size=(test_size+eval_size), random_state=42)
        self.eval_index, self.test_index = train_test_split(self.eval_index, test_size= test_size / (test_size+eval_size), random_state=42)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

class EfficientNet:
    def __init__(self, new_head:bool = False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.base_conf = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights = self.base_conf)
        self.connector = self.base_conf.transforms()
         #I do understand, that ImageNet has these classes that I am willing to train on, but considering the quality of ImageNet (todo paste promo link) it won't bother to finetune tiny model
        if new_head:
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=1280, out_features=10, bias=True) #todo weight init
            )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        self.loss = nn.CrossEntropyLoss()

    def train(self, train_dataloader:data.DataLoader, eval_dataloader:data.DataLoader, train_steps:int):

        #freezing backbone
        for param in self.model.features.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        self.model.to(self.device)

        writer = SummaryWriter()
        for batch in range(train_steps):
            cum_loss = 0  # cumulative loss for plot

            self.optimizer.zero_grad()

            features, targets = next(iter(train_dataloader))
            features = features.to(self.device)

            prediction = self.model(features)
            output = self.loss(prediction, targets.to(self.device))
            cum_loss += output.item()
            output.backward()
            self.optimizer.step()

            if batch % 20 == 0:
                avg_val_loss = self.eval(eval_dataloader)
                writer.add_scalars(f'loss', {"train": cum_loss/ 20, "eval": avg_val_loss}, batch)

        writer.close()

    def eval(self, dataloader:data.DataLoader) -> float:
        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            for batch in dataloader:
                features, targets = batch
                features = features.to(self.device)
                prediction = self.model(features)
                output = self.loss(prediction, targets.to(self.device))
                val_loss += output.item()

        return val_loss / len(dataloader)

    def predict(self, inputs) -> torch.Tensor:
        self.model.to(self.device)
        with torch.no_grad():
            self.model.eval()
            prediction = self.model(inputs.to(self.device))
            prediction = torch.argmax(prediction, dim = 1)
        return prediction

    def eval_statistics(self, dataloader:data.DataLoader):
        predicted_labels = []
        target_labels = []
        for batch in dataloader:
            features, targets = batch
            predicted_label = self.predict(features)
            target_label = torch.argmax(targets, dim=1)

            predicted_labels.extend(predicted_label.cpu().detach().numpy())
            target_labels.extend(target_label.cpu().detach().numpy())

        conf_matrix = confusion_matrix(target_labels, predicted_labels)
        conf_report = classification_report(target_labels, predicted_labels, target_names=CVDataset.label_to_index.keys())
        return conf_matrix, conf_report


    def save(self, path:str):
        torch.save(self.model.state_dict(), path)
        torch.save(self.optimizer.state_dict(), f"{path[:-3]}_optim.pt")

    def load(self, path:str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.optimizer.load_state_dict(torch.load(f"{path[:-3]}_optim.pt", weights_only=True))

if __name__ == "__main__":
    en_b0 = EfficientNet()
    animals_10_dataset = CVDataset("data/raw-img", 0.2, 0.1, en_b0.connector)
    batch_size = 64

    train_dataloader = data.DataLoader(
        data.Subset(animals_10_dataset, animals_10_dataset.train_index),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    eval_dataloader = data.DataLoader(
        data.Subset(animals_10_dataset, animals_10_dataset.eval_index),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    new_en_b0 = EfficientNet(new_head=True)
    new_en_b0.train(train_dataloader, eval_dataloader, train_steps=400)
    new_en_b0.save("models/new_head_from_cvpy.pt")

    mat, rep = new_en_b0.eval_statistics(eval_dataloader)
    print(rep)