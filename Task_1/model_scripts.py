import abc

from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torchvision
from torchvision.datasets import MNIST
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class MnistClassifierInterface(abc.ABC):
    def __init__(self):
        self.device = {}
        self.batch_size = 128
        self.device = {"torch": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")} #select suitable device

    @abc.abstractmethod
    def train(self, inputs, targets):
        pass

    @abc.abstractmethod
    def predict(self, inputs):
        pass

# class RFModel(MnistClassifierInterface):
#     def __init__(self):
#
#     def train(self):
#
#     def predict(self):


#=====================test
class TorchBasedModel(MnistClassifierInterface):
    def __init__(self):
        super().__init__()
        self.device = self.device["torch"]
        self.model = None
        self.optimizer = None
        self.loss = None
        self.model_connector = None

    def train(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        self.model.to(self.device)
        cum_loss = 0  # cumulative loss for plot
        for image, target in zip(torch.split(inputs, self.batch_size),
                                 torch.split(targets, self.batch_size)):  # shard dataset in batches using torch.split()
            self.optimizer.zero_grad()
            image = self.model_connector(image).to(
                self.device)  # torch.Tensor(shape = (batch_size, h ,w)) -> torch.Tensor(shape = (batch_size, h * w))

            prediction = self.model(image)
            output = self.loss(prediction, target.to(self.device))
            cum_loss += output.item()
            output.backward()
            self.optimizer.step()

        return cum_loss


    def evaluate(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        val_loss = 0  # cumulative loss for plot
        for image, target in zip(torch.split(inputs, self.batch_size),
                                 torch.split(targets, self.batch_size)):  # shard dataset in batches using torch.split()

            image = self.model_connector(image).to(
                self.device)  # torch.Tensor(shape = (batch_size, h ,w)) -> torch.Tensor(shape = (batch_size,channels,h, w))
            prediction = self.model(image)
            output = self.loss(prediction, target.to(self.device))
            val_loss += output.item()

        return val_loss


    def predict(self, inputs: np.array(Image)) -> torch.Tensor:
        inputs = self.model_connector(torchvision.transforms.ToTensor()(inputs)).to(
            self.device)  # np.array(list(PIL.Image))) -> todo
        print(inputs.shape)
        output = self.model(inputs)
        output = F.softmax(output)  # softmax, because we want to return probs
        return output
#==============test
class FFModel(TorchBasedModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
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
            nn.ReLU() #without softmax, because it will cause the same operation in CELoss todo
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        self.loss = nn.CrossEntropyLoss()

        def unsqueeze_in_function(x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(1)

        self.model_connector = unsqueeze_in_function
#===========test
#==========works perfectly fine!
# class FFModel(MnistClassifierInterface):
#     def __init__(self):
#         super().__init__()
#         self.device = self.device["torch"]
#         self.model = nn.Sequential(
#             nn.Linear(28 * 28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#         )
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)
#         self.loss = nn.CrossEntropyLoss()
#
#     def train(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
#         self.model.to(self.device)
#         cum_loss = 0 # cumulative loss for plot
#         for image, target in zip(torch.split(inputs, self.batch_size), torch.split(targets, self.batch_size)): #shard dataset in batches using torch.split()
#             self.optimizer.zero_grad()
#             image = image.flatten(start_dim=1).to(self.device)  # torch.Tensor(shape = (batch_size, h ,w)) -> torch.Tensor(shape = (batch_size, h * w))
#
#             prediction = self.model(image)
#             output = self.loss(prediction, target.to(self.device))
#             cum_loss += output.item()
#             output.backward()
#             self.optimizer.step()
#
#         return cum_loss
#
#     def evaluate(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
#
#         val_loss = 0  # cumulative loss for plot
#         for image, target in zip(torch.split(inputs, self.batch_size),torch.split(targets, self.batch_size)):  # shard dataset in batches using torch.split()
#
#             image = image.flatten(start_dim=1).to(self.device)  # torch.Tensor(shape = (batch_size, h ,w)) -> torch.Tensor(shape = (batch_size,channels,h, w))
#             prediction = self.model(image)
#             output = self.loss(prediction, target.to(self.device))
#             val_loss += output.item()
#
#         return val_loss
#
#     def predict(self, inputs: np.array(Image)) -> torch.Tensor:
#         inputs = torchvision.transforms.ToTensor()(inputs).unsqueeze(1).to(self.device) #np.array(list(PIL.Image))) -> todo
#         print(inputs.shape)
#         output = self.model(inputs)
#         output = F.softmax(output) #softmax, because we want to return probs
#         return output
#==========works perfectly fine!
#==========works perfectly fine!
# class CNNModel(MnistClassifierInterface):
#     def __init__(self):
#         super().__init__()
#         self.device = self.device["torch"]
#         self.model = nn.Sequential(
#             nn.Conv2d(1, 8, 5), # 28x28 -> 24x24
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), #24x24 -> 12x12
#             nn.Conv2d(8, 16, 3), #12x12 -> 10x10
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), #10x10 -> 5x5
#             nn.Conv2d(16, 32, 3), #5x5 -> 3x3
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3),  # 3x3 -> 1x1
#             nn.Flatten(), #[batch_size, channels, 1, 1] -> [batch_size, channels]
#             nn.Linear(64, 10),
#             nn.ReLU() #without softmax, because it will cause the same operation in CELoss todo
#         ).to(self.device)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)
#         self.loss = nn.CrossEntropyLoss()
#
#     def train(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
#
#         cum_loss = 0 # cumulative loss for plot
#         for image, target in zip(torch.split(inputs, self.batch_size), torch.split(targets, self.batch_size)): #shard dataset in batches using torch.split()
#             self.optimizer.zero_grad()
#             image = image.unsqueeze(1).to(self.device) # torch.Tensor(shape = (batch_size, h ,w)) -> torch.Tensor(shape = (batch_size,channels,h, w))
#             prediction = self.model(image)
#             output = self.loss(prediction, target.to(self.device))
#             cum_loss += output.item()
#             output.backward()
#             self.optimizer.step()
#
#         return cum_loss
#
#
#     def evaluate(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
#
#         val_loss = 0  # cumulative loss for plot
#         for image, target in zip(torch.split(inputs, self.batch_size),torch.split(targets, self.batch_size)):  # shard dataset in batches using torch.split()
#
#             image = image.unsqueeze(1).to(self.device) # torch.Tensor(shape = (batch_size, h ,w)) -> torch.Tensor(shape = (batch_size,channels,h, w))
#             prediction = self.model(image)
#             output = self.loss(prediction, target.to(self.device))
#             val_loss += output.item()
#
#         return val_loss
#
#     def predict(self, inputs: np.array(Image)) -> torch.Tensor:
#         inputs = torchvision.transforms.ToTensor()(inputs).unsqueeze(1).to(self.device) #np.array(list(PIL.Image))) -> todo
#         print(inputs.shape)
#         output = self.model(inputs)
#         output = F.softmax(output) #softmax, because we want to return probs
#         return output
#==========works perfectly fine!

class MnistClassifier:
    def __init__(self, algorithm:str):
        self.algorithm = algorithm.lower()
        self.train_index, self.test_index = None, None
        if self.algorithm == 'nn':
            self.model = FFModel()
        elif self.algorithm == 'cnn':
            self.model = CNNModel()
        elif self.algorithm == 'rf':
            print('skip')
            #self.model = RFModel()
        else:
            raise AttributeError(f"\"{algorithm}\" as algorithm type is not supported")

    def train(self, dataset: torchvision.datasets.MNIST):
        #if self.alg == torch -> torch_preprocess | preprocess (torch
        #if self.alg != torch -> np_preprocess | preprocess (np)
        features, targets = None, None
        if self.algorithm == 'rf': #todo maybe not simple if else but elifs / structures
            pass
        else:
            features, targets = dataset.data, dataset.targets
            #next 2 lines will refactor targets from labels into one-hot vectors
            new_targets = torch.zeros(len(targets), 10)
            new_targets[np.arange(len(targets)), targets] = 1
            targets = new_targets

            features = features/255 # normalize tensors from [0,255] -> [0,1]

        if self.train_index is None:
            self.train_index, self.test_index = train_test_split(np.arange(0, len(targets)), test_size=0.2, random_state=42)
        train_size = len(self.train_index)
        eval_size = len(self.test_index)
        writer = SummaryWriter()
        for epoch in range(10):
            train_loss = self.model.train(features[self.train_index], targets[self.train_index])
            val_loss = self.model.evaluate(features[self.test_index], targets[self.test_index])
            writer.add_scalars(f'loss/{self.algorithm}', {"train": train_loss/ train_size, "eval": val_loss/ eval_size}, epoch)

        writer.close()

    def predict(self, inputs: np.array(Image)) -> np.array(int):
        outputs = self.model.predict(inputs)
        labels = torch.argmax(outputs, dim=1).numpy() #argmax because we want to return only label
        return labels


if __name__ == '__main__':
    classifier = MnistClassifier(algorithm='cnn')
    dataset = MNIST(root = "./data", download = True)

    classifier.train(dataset)
