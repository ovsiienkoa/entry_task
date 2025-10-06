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
    """
    Abstract Base Class (Interface) for all MNIST classification models.

    Defines the standard methods and attributes required for any classifier
    to be used within the MnistClassifier framework.
    """
    def __init__(self):
        """
        Initializes common parameters for all classifiers.
        """
        self.batch_size = 128
        self.model_py_type = None # String identifier for the underlying framework (e.g., "sklearn", "torch")
        self.device = {"torch": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")} #select suitable device

    @abc.abstractmethod
    def train(self, inputs, targets) -> float:
        """
        Trains the classifier model on the provided data.

        Args:
            inputs: Training features (e.g., image data). Type depends on the specific model
                    implementation (e.g., np.array for RF, torch.Tensor for PyTorch models).
            targets: Training labels. Type depends on the specific model.

        Returns:
            float: The training loss or an equivalent measure of training performance.
        """
        pass

    @abc.abstractmethod
    def predict(self, inputs) -> np.array:
        """
        Generates probability predictions for the input data.

        Args:
            inputs: Data to predict on, typically an image or a batch of images.
                    Type can be flexible (e.g., PIL.Image or np.array(Image)).

        Returns:
            np.array: Array of probability distributions over the 10 classes.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, inputs, targets) -> float:
        """
        Evaluates the classifier model on the provided data (e.g., validation set). (Basically, self.train without weights updates, the same metric is being returned)

        Args:
            inputs: Evaluation features. Type depends on the specific model.
            targets: True evaluation labels. Type depends on the specific model.

        Returns:
            float: The evaluation loss or an equivalent measure of validation performance.
        """
        pass

class RFModel(MnistClassifierInterface):
    """
    Random Forest Classifier implementation for MNIST, based on scikit-learn.
    """
    def __init__(self):
        """
        Initializes the RFModel, setting up the scikit-learn Random Forest classifier.
        """
        super().__init__()
        self.model_py_type = "sklearn"
        self.model = RandomForestClassifier(n_estimators=500, n_jobs = -1, random_state=42)

    def train(self, inputs:np.array, targets:np.array) -> float:
        """
        Trains the Random Forest model.

        Args:
            inputs (np.array): Training features (images as numpy arrays, expected shape (N, 28, 28)).
            targets (np.array): Training labels (integer class indices, expected shape (N,)).

        Returns:
            float: avg cross-entropy loss per batch on the training set after training.
        """
        # Reshape images from (N, 28, 28) to (N, 784) for the tree
        inputs = inputs.reshape((-1,28*28))
        self.model.fit(inputs, targets)
        predicted = np.array(self.model.predict_proba(inputs))

        # Convert integer targets to one-hot encoding for CE calculation
        ce_targets = np.zeros((len(targets), 10))
        ce_targets[np.arange(len(targets)), targets] = 1

        return log_loss(ce_targets, predicted)

    def evaluate(self, inputs:np.array, targets:np.array) -> float:
        """
        Evaluates the Random Forest model (train method without .fit part, same metric)

        Args:
            inputs (np.array): Evaluation features (images as numpy arrays, expected shape (N, 28, 28)).
            targets (np.array): True evaluation labels (one-hot encoded, expected shape (N, 10)).

        Returns:
            float: The cross-entropy loss on the evaluation set.
        """
        # Reshape images from (N, 28, 28) to (N, 784) for the tree
        inputs = inputs.reshape((-1,28*28))
        predicted = self.model.predict_proba(inputs)
        return log_loss(targets, predicted)

    def predict(self, inputs: np.array(Image)) -> np.array:
        """
        Generates probability predictions.

        Args:
            inputs (np.array): Input image data (e.g., a single PIL Image converted to np.array, or a batch).
                               Expected shape (N, 28, 28) or (28, 28) for a single image.

        Returns:
            np.array: Probability distributions over the 10 classes, shape (N, 10).
        """
        # Reshape images from (N, 28, 28) to (N, 784) for the tree
        inputs = np.array(inputs).reshape((-1, 28 * 28))
        return self.model.predict_proba(inputs)



class TorchBasedModel(MnistClassifierInterface):
    """
    Abstract Base Class for models implemented using PyTorch (Feed-Forward and CNN).
    Handles common PyTorch initialization, training loops, evaluation, and prediction logic.
    """
    def __init__(self):
        """
        Initializes common PyTorch elements: device, model, optimizer, loss function,
        and a 'connector' function for pre-processing input tensors before feeding to the model.
        """
        super().__init__()
        self.model_py_type = "torch"
        self.device = self.device["torch"]
        self.model = None # Specific model (NN/CNN) defined in subclasses
        self.optimizer = None #dummy variable
        self.loss = None #dummy variable
        self.model_connector = None # A function (e.g., Flatten or Unsqueeze) to prepare the input tensor for the specific model architecture

    def train(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Performs one epoch of training for the PyTorch model.

        Args:
            inputs (torch.Tensor): Training features (images, normalized tensor).
            targets (torch.Tensor): Training labels (one-hot encoded tensor).

        Returns:
            float: The cumulative training loss over the epoch.
        """
        self.model.to(self.device)
        self.model.train()

        # inputs: (N, 28*28) for FF, (N, 1, 28, 28) for CNN
        # targets: (N, 10)
        cum_loss = 0  # cumulative loss for plot
        for image, target in zip(torch.split(inputs, self.batch_size), torch.split(targets, self.batch_size)):  # shard dataset in batches using torch.split()
            self.optimizer.zero_grad()
            # Apply model_connector (e.g., Flatten for FF, Unsqueeze for CNN if needed) and move to device
            image = self.model_connector(image).to(self.device)

            prediction = self.model(image)
            output = self.loss(prediction, target.to(self.device))
            cum_loss += output.item()
            output.backward()
            self.optimizer.step()

        return cum_loss


    def evaluate(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Evaluates the PyTorch model on a given dataset partition.

        Args:
            inputs (torch.Tensor): Evaluation features.
            targets (torch.Tensor): True evaluation labels (one-hot encoded tensor).

        Returns:
            float: The cumulative evaluation loss.
        """

        val_loss = 0  # cumulative loss for plot
        self.model.eval()
        with torch.no_grad():
            for image, target in zip(torch.split(inputs, self.batch_size), torch.split(targets, self.batch_size)):  # shard dataset in batches using torch.split()

                image = self.model_connector(image).to(self.device)
                prediction = self.model(image)
                output = self.loss(prediction, target.to(self.device))
                val_loss += output.item()

        return val_loss


    def predict(self, inputs: np.array(Image)) -> np.array:
        """
        Generates probability predictions using the PyTorch model.

        Args:
            inputs (np.array): Input image data (expected to be a PIL Image or list of PIL Images,
                               converted to np.array by the caller's context).

        Returns:
            np.array: Probability distributions over the 10 classes, shape (N, 10).
        """
        inputs = self.model_connector(torchvision.transforms.ToTensor()(inputs)).to(
            self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(inputs)
            output = F.softmax(output, dim = 1).cpu().detach().numpy()  # softmax, because we want to return probs
        return output

class FFModel(TorchBasedModel):
    """
    Feed-Forward Neural Network (Fully Connected) Classifier for MNIST.
    """
    def __init__(self, loss= nn.CrossEntropyLoss(), optimizer: torch.optim = optim.Adam, learning_rate:float = 0.003):
        """
        Initializes the FFModel, defining the network architecture, optimizer, and loss function.
        """
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10), #without softmax, because it will cause the same operation in CELoss
        )
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.loss = loss
        #(B, 28, 28) -> (B, 784).
        self.model_connector = nn.Flatten(start_dim = 1) #beacuse nn.Linear accepts only (a,b) inputs

class CNNModel(TorchBasedModel):
    """
   Convolutional Neural Network Classifier for MNIST.
   """
    def __init__(self, loss= nn.CrossEntropyLoss(), optimizer: torch.optim = optim.Adam, learning_rate:float = 0.003):
        """
        Initializes the CNNModel, defining the network architecture, optimizer, and loss function.
        """
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
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.loss = loss

        def unsqueeze_in_function(x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(1) # Add channel dimension at index 1 because CNN accepts only (N, C, H, W)

        self.model_connector = unsqueeze_in_function

class MnistClassifier:
    """
        A unified wrapper class for MNIST classification, implementing the Strategy pattern.

        It selects the appropriate underlying classifier (RF, NN, or CNN) based on the
        initialization parameter and provides a consistent interface for training and prediction.
    """
    def __init__(self, algorithm:str):
        """
        Initializes the MnistClassifier, instantiating the selected model.

        Args:
            algorithm (str): The classification algorithm to use.
                             Possible values: 'rf' (Random Forest), 'nn' (Feed-Forward NN), 'cnn' (CNN).

        Raises:
            AttributeError: If an unsupported algorithm name is provided.
        """
        self.algorithm = algorithm.lower()
        self.train_index, self.test_index = None, None #indicies for train and test sets

        if self.algorithm == 'nn':
            self.model = FFModel()
        elif self.algorithm == 'cnn':
            self.model = CNNModel()
        elif self.algorithm == 'rf':
            self.model = RFModel()
        else:
            raise AttributeError(f"\"{algorithm}\" as algorithm type is not supported")

    def train(self, dataset: torchvision.datasets.MNIST):
        """
        Loads, pre-processes, splits the MNIST dataset, and trains the selected model.
        Training progress is logged to TensorBoard in 'runs' directory.

        Args:
            dataset (torchvision.datasets.MNIST): The full MNIST dataset object.
        """
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

        # Model training loop: 10 epochs for torch models, 1 for scikit-learn (RF)
        num_epochs = 10 if self.model.model_py_type == "torch" else 1

        writer = SummaryWriter()
        for epoch in range(num_epochs):
            train_loss = self.model.train(features[self.train_index], targets[self.train_index])
            val_loss = self.model.evaluate(features[self.test_index], targets[self.test_index])
            writer.add_scalars(f'loss/{self.algorithm}', {"train": train_loss/ train_size, "eval": val_loss/ eval_size}, epoch)

        writer.close()

    def predict(self, inputs: np.array(Image)) -> np.array(int):
        """
        Generates final class predictions (labels) for the input data.

        This method first calls the model's `predict` method to get probabilities
        and then converts them to the final class label.

        Args:
            inputs (np.array): Input image data (e.g., a single PIL Image converted to np.array).

        Returns:
            np.array: Predicted class labels (integers from 0 to 9), shape (N,).
        """
        outputs = self.model.predict(inputs)
        labels = np.argmax(outputs, axis=1) #argmax because we want to return only label
        return labels

if __name__ == '__main__':
    # Download and load the MNIST dataset
    dataset = MNIST(root="./data", download=True)

    options = ["rf", "nn", "cnn"]
    for option in options:
        classifier = MnistClassifier(algorithm=option)
        classifier.train(dataset)

        funny_ind = classifier.test_index[0]
        print(dataset[funny_ind][1])
        print(classifier.predict(dataset[funny_ind][0]))
