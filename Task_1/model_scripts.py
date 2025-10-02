import abc

from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
#import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

import torchvision
from torchvision.datasets import MNIST
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class MnistClassifierInterface(abc.ABC):
    def __init__(self):
        self.batch_size = 128
        self.model_py_type = None
        self.device = {"torch": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")} #select suitable device

    @abc.abstractmethod
    def train(self, inputs, targets) -> float:
        pass

    @abc.abstractmethod
    def predict(self, inputs) -> np.array:
        pass

    @abc.abstractmethod
    def evaluate(self, inputs, targets) -> float:
        pass

    # @abc.abstractmethod
    # def save(self, path:str):
    #     pass
    #
    # @abc.abstractmethod
    # def load(self, path:str):
    #     pass

class RFModel(MnistClassifierInterface):
    def __init__(self):
        super().__init__()
        self.model_py_type = "sklearn"
        self.model = RandomForestClassifier(n_estimators=500, n_jobs = -1, random_state=42)

    def train(self, inputs:np.array, targets:np.array) -> float:
        inputs = inputs.reshape((-1,28*28))
        self.model.fit(inputs, targets)
        predicted = np.array(self.model.predict_proba(inputs))

        ce_targets = np.zeros((len(targets), 10))
        ce_targets[np.arange(len(targets)), targets] = 1

        return log_loss(ce_targets, predicted)

    def evaluate(self, inputs:np.array, targets:np.array) -> float:
        inputs = inputs.reshape((-1,28*28))
        predicted = self.model.predict_proba(inputs)
        return log_loss(targets, predicted)

    def predict(self, inputs: np.array(Image)) -> np.array:
        inputs = np.array(inputs).reshape((-1, 28 * 28))
        return self.model.predict_proba(inputs)

    # def save(self, path:str):
    #     joblib.dump(self.model, path)


class TorchBasedModel(MnistClassifierInterface):
    def __init__(self):
        super().__init__()
        self.model_py_type = "torch"
        self.device = self.device["torch"]
        self.model = None
        self.optimizer = None
        self.loss = None
        self.model_connector = None

    def train(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        self.model.to(self.device)
        cum_loss = 0  # cumulative loss for plot
        for image, target in zip(torch.split(inputs, self.batch_size), torch.split(targets, self.batch_size)):  # shard dataset in batches using torch.split()
            self.optimizer.zero_grad()
            image = self.model_connector(image).to(self.device)

            prediction = self.model(image)
            output = self.loss(prediction, target.to(self.device))
            cum_loss += output.item()
            output.backward()
            self.optimizer.step()

        return cum_loss


    def evaluate(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        val_loss = 0  # cumulative loss for plot
        for image, target in zip(torch.split(inputs, self.batch_size), torch.split(targets, self.batch_size)):  # shard dataset in batches using torch.split()

            image = self.model_connector(image).to(self.device)
            prediction = self.model(image)
            output = self.loss(prediction, target.to(self.device))
            val_loss += output.item()

        return val_loss


    def predict(self, inputs: np.array(Image)) -> np.array:
        inputs = self.model_connector(torchvision.transforms.ToTensor()(inputs)).to(
            self.device)  # np.array(list(PIL.Image))) -> todo
        output = self.model(inputs)
        output = F.softmax(output, dim = 1).cpu().detach().numpy()  # softmax, because we want to return probs
        return output

    # def save(self, path:str):
    #     torch.save(self.model.state_dict(), path)

class FFModel(TorchBasedModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10), #without softmax, because it will cause the same operation in CELoss
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        self.loss = nn.CrossEntropyLoss()
        self.model_connector = nn.Flatten(start_dim = 1)

class CNNModel(TorchBasedModel):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 5), # 28x28 -> 24x24
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #24x24 -> 12x12
            nn.Conv2d(8, 16, 3), #12x12 -> 10x10
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #10x10 -> 5x5
            nn.Conv2d(16, 32, 3), #5x5 -> 3x3
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),  # 3x3 -> 1x1
            nn.Flatten(), #[batch_size, channels, 1, 1] -> [batch_size, channels]
            nn.Linear(64, 10),
            nn.ReLU() #without softmax, because it will cause the same operation in CELoss
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        self.loss = nn.CrossEntropyLoss()

        def unsqueeze_in_function(x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(1)

        self.model_connector = unsqueeze_in_function

class MnistClassifier:
    def __init__(self, algorithm:str):

        self.algorithm = algorithm.lower()
        self.train_index, self.test_index = None, None

        if self.algorithm == 'nn':
            self.model = FFModel()
        elif self.algorithm == 'cnn':
            self.model = CNNModel()
        elif self.algorithm == 'rf':
            self.model = RFModel()
        else:
            raise AttributeError(f"\"{algorithm}\" as algorithm type is not supported")

    def train(self, dataset: torchvision.datasets.MNIST):

        # tiny dataset preprocessing
        features, targets = dataset.data, dataset.targets
        features = features / 255  # normalize tensors from [0,255] -> [0,1]
        #tiny dataset post-preprocessing
        if self.model.model_py_type == "torch":
            # next 3 lines will refactor targets from labels into one-hot vectors
            new_targets = torch.zeros(len(targets), 10)
            new_targets[np.arange(len(targets)), targets] = 1
            targets = new_targets

        elif self.model.model_py_type == "sklearn":
            features = features.numpy()
            targets = targets.numpy()

        else:
            raise KeyError(f"tiny dataset post-preprocessing isn't available for \"{self.model.model_py_type}\" of \"{self.algorithm}\" algorithm")

        #dataset split into train/eval folders
        if self.train_index is None:
            self.train_index, self.test_index = train_test_split(np.arange(0, len(targets)), test_size=0.4, random_state=42)

        train_size = len(self.train_index)
        eval_size = len(self.test_index)

        #model training and monitoring
        num_epochs = 10 if self.model.model_py_type == "torch" else 1

        writer = SummaryWriter()
        for epoch in range(num_epochs):
            train_loss = self.model.train(features[self.train_index], targets[self.train_index])
            val_loss = self.model.evaluate(features[self.test_index], targets[self.test_index])
            writer.add_scalars(f'loss/{self.algorithm}', {"train": train_loss/ train_size, "eval": val_loss/ eval_size}, epoch)

        writer.close()

    def predict(self, inputs: np.array(Image)) -> np.array(int):
        outputs = self.model.predict(inputs)
        labels = np.argmax(outputs, axis=1) #argmax because we want to return only label
        return labels

    # def save(self, path:str):
    #     self.model.save(path)

if __name__ == '__main__':
    dataset = MNIST(root="./data", download=True)

    options = ["rf", "nn", "cnn"]
    for option in options:
        classifier = MnistClassifier(algorithm=option)
        classifier.train(dataset)

        funny_ind = classifier.test_index[0]
        print(dataset[funny_ind][1])
        print(classifier.predict(dataset[funny_ind][0]))
