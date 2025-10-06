import os
import argparse
import json

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

import random
class RandomApply(nn.Module):
    """
    Applies a given transformation function 'fn' with a probability 'p'. (*for random image augmentations in the dataset)

    Attributes:
        fn (callable): The transformation function to apply.
        p (float): The probability of applying the transformation.
    """
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class CVDataset(torch.utils.data.Dataset):
    """
        Custom PyTorch Dataset for loading and augmenting an animal image classification dataset.
        It supports multi-class classification and performs train/eval/test split. But in this case test and eval are being use as the same split
        Images are loaded, augmented twice (SimCLR style), and stored in memory.

        Attributes:
            data (torch.Tensor): Tensor to hold processed image tensors.
            targets (torch.Tensor): Tensor to hold one-hot encoded target tensors.
            train_index (np.ndarray): Indices for the training set.
            test_index (np.ndarray): Indices for the test set.
            eval_index (np.ndarray): Indices for the evaluation (validation) set.
            label_to_index (dict): Maps class names (folder names) to integer indices.
            class_num (int): Total number of unique classes.
            DEFAULT_AUG (torch.nn.Sequential): Default SimCLR-like augmentation pipeline.
        """
    # default*(no normalization because it's already in EfficientNet and no random crops) SimCLR augmentation
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
        """
        Initializes the CVDataset. Loads images, performs augmentation, and splits indices.

        Args:
            folder_path (str): Path to the root directory where each sub-folder is a class.
            eval_size (float): Portion of the data to be used for the evaluation set.
            test_size (float): Portion of the data to be used for the test set.
            connector (torchvision.transforms): A final transformation (e.g., normalization) to be applied
                           after the default augmentations.
        """
        super().__init__()
        self.data = []
        self.targets = []
        self.train_index = None
        self.test_index = None
        self.eval_index = None

        self.label_to_index = {}
        labels = os.listdir(folder_path)
        # Determine classes from sub-folder names
        for i, label in enumerate(labels):
            self.label_to_index.update({label: i})

        self.class_num = len(labels)

        for folder in os.listdir(folder_path):

            # Create one-hot encoded target tensor for the current class
            target = torch.zeros(self.class_num)
            target[self.label_to_index[folder]] = 1

            # Estimate expected number of files for target tensor expansion
            # Limiting to 400 files per class for memory reasons, and *2 for double augmentation
            expected_number_of_files = len(os.listdir(os.path.join(folder_path, folder))[:400]) * 2
            errors_counter = 0 #variable for calculating exact number of files

            for file in os.listdir(f"{folder_path}/{folder}")[:400]: #[:400] - 16 gb of ram

                image_tensor = decode_image(f"{folder_path}/{folder}"+"/"+file) # Read image as tensor
                try:
                    # Apply augmentation twice (SimCLR style) and the final connector transform
                    # [:3] handles images with 4 channels (like PNG) by taking only the first 3 (RGB)
                    self.data.append(connector(CVDataset.DEFAULT_AUG(image_tensor[:3, :, :])))
                    self.data.append(connector(CVDataset.DEFAULT_AUG(image_tensor[:3, :, :])))
                except RuntimeError:
                    errors_counter +=1
                    print(f"file, folder: {file}, {folder}")

            # Expand the one-hot target to match the number of samples (2x images - 2x errors)
            target = target.expand(expected_number_of_files - 2*errors_counter, self.class_num)
            self.targets.extend(target)

        self.data = torch.stack(self.data, dim = 0)
        self.targets = torch.stack(self.targets, dim = 0)

        # Split indices into train, eval, and test sets
        # First split into train and (eval+test)
        self.train_index, self.eval_index = train_test_split(np.arange(0, len(self.targets)), test_size=(test_size+eval_size), random_state=42)
        self.eval_index, self.test_index = train_test_split(self.eval_index, test_size= test_size / (test_size+eval_size), random_state=42)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

class EfficientNet:
    """
    A wrapper class for linear-probing the EfficientNet-B0 model for image classification.

    The model is initialized with ImageNet pre-trained weights, and the classifier head
    is replaced for the specific number of animal classes.

    Attributes:
        device (torch.device): Device (CPU or CUDA) to run the model on.
        base_conf (EfficientNet_B0_Weights): Pre-trained weights configuration.
        model (EfficientNet): The loaded and modified EfficientNet-B0 model.
        connector (callable): Image transformation pipeline from the pre-trained weights.
        optimizer (optim.Optimizer): Adam optimizer.
        loss (nn.Module): Cross-Entropy Loss function.
        label_to_index (dict): Mapping from class labels to indices.
    """
    def __init__(self, label_to_index:dict = {"removed_class":0}):
        """
        Initializes the EfficientNet model.

        Args:
            label_to_index (dict): Dictionary mapping class names to indices.
                                   Defaults to a placeholder for initial load.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.base_conf = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights = self.base_conf) # creating a pretrained EfficientNet_B0
        self.connector = self.base_conf.transforms() # Get the preprocessor for EfficientNet_B0
         #I do understand, that ImageNet has these classes that I am willing to train on, but considering the quality of ImageNet (todo paste promo link) it won't bother to finetune tiny model
        self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=1280, out_features=len(label_to_index), bias=True) #todo weight init
            )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        self.loss = nn.CrossEntropyLoss()
        self.label_to_index = label_to_index

    def train(self, train_dataloader:data.DataLoader, eval_dataloader:data.DataLoader, train_steps:int):
        """
       Runs the training loop for the model, logging loss to TensorBoard(dir = runs).

       The model's backbone is frozen, and only the new classifier head is trained.

       Args:
           train_dataloader (data.DataLoader): DataLoader for the training data.
           eval_dataloader (data.DataLoader): DataLoader for the evaluation data.
           train_steps (int): The number of batches (steps) to process during training.

       Raises:
           ValueError: If the `label_to_index` is not properly initialized.
       """

        if len(self.label_to_index) == 1:
            raise ValueError("label_to_index cannot be of length 1, you should reinitialize your model with label_to_index attribute")

        #freezing backbone
        for param in self.model.features.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        self.model.to(self.device)
        self.model.train()

        writer = SummaryWriter()
        iterator = iter(train_dataloader)
        for batch in range(train_steps):
            cum_loss = 0  # cumulative loss for plot

            self.optimizer.zero_grad()

            try:
                features, targets = next(iterator)
            except StopIteration:
                # Re-initialize iterator if all data has been processed (useful for fixed steps)
                iterator = iter(train_dataloader)
                features, targets = next(iterator)

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
        """
        Evaluates the model on a given dataloader.

        Args:
            dataloader (data.DataLoader): DataLoader for the evaluation data.

        Returns:
            float: The average loss over the evaluation dataset.
        """
        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            for batch in dataloader:
                features, targets = batch
                features = features.to(self.device)
                prediction = self.model(features)
                output = self.loss(prediction, targets.to(self.device))
                val_loss += output.item()

        self.model.train()
        return val_loss / len(dataloader)

    def predict(self, inputs:torch.Tensor) -> torch.Tensor:
        """
        Performs inference on a batch of input tensors.

        Args:
            inputs (torch.Tensor): A batch of input image tensors.

        Returns:
            torch.Tensor: Only the predicted class indices for the batch.
        """
        self.model.to(self.device)
        with torch.no_grad():
            self.model.eval()
            prediction = self.model(inputs.to(self.device))
            prediction = torch.argmax(prediction, dim = 1)  # Get the index of the max log-probability
        return prediction

    def eval_statistics(self, dataloader:data.DataLoader):
        """
        Computes the confusion matrix and classification report for the dataset.

        Args:
            dataloader (data.DataLoader): DataLoader for the data to be evaluated.

        Returns:
            tuple: (confusion_matrix, classification_report)
        """
        predicted_labels = []
        target_labels = []
        for batch in dataloader:
            features, targets = batch
            predicted_label = self.predict(features)

            target_label = torch.argmax(targets, dim=1) # Convert one-hot target to class index

            # Collect results for sklearn
            predicted_labels.extend(predicted_label.cpu().detach().numpy())
            target_labels.extend(target_label.cpu().detach().numpy())

        conf_matrix = confusion_matrix(target_labels, predicted_labels)
        conf_report = classification_report(target_labels, predicted_labels, target_names=self.label_to_index.keys()) # Use class labels for better report readability
        return conf_matrix, conf_report


    def save(self, path:str):
        """
        Saves the model state, optimizer state, and class index mapping.

        Args:
            path (str): Directory path to save the checkpoints.
        """
        torch.save(self.model.state_dict(), f"{path}/en_b0.pt")
        torch.save(self.optimizer.state_dict(), f"{path}/en_b0_optim.pt")
        with open(f'{path}/en_b0_label_to_index.txt', 'w+') as f:
            json.dump(self.label_to_index, f)

    def load(self, path:str):
        """
       Loads the model state, optimizer state, and class index mapping from a path.

       Args:
           path (str): Directory path from which to load the checkpoints.
       """
        with open(f'{path}/en_b0_label_to_index.txt', 'r+') as f:
            self.label_to_index = json.load(f)

        self.model.classifier = nn.Sequential( #because we can't just reinitialize the model object inside this very object
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=len(self.label_to_index), bias=True)
        )

        self.model.load_state_dict(torch.load(f"{path}/en_b0.pt", weights_only=True))
        self.optimizer.load_state_dict(torch.load(f"{path}/en_b0_optim.pt", weights_only=True))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Define the parameters/arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="en-b0",
        help="en-bo: EfficientNet-B0; ",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="data/raw-img",
        help="Path to the directory containing the NER training data."
    )
    parser.add_argument(
        "--eval_size",
        type=float,
        default=0.2,
        help="portion for eval"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="portion for test"
    )
    parser.add_argument(
        "--model_checkpoint_dir",
        type=str,
        default="./models",
        help="Directory to save the trained model checkpoints."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size."
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=400,
        help="Number of processed batches during training."
    )

    args = parser.parse_args()

    if args.model_name == "en-b0": #we create the model object twice, because dataset is need to how to prepare data for exact data, meanwhile the model needs labels from dataset
        model = EfficientNet()
    else:
        raise Exception("Invalid model_name")

    animals_10_dataset = CVDataset(args.train_data_path, args.eval_size, args.test_size, model.connector)

    if args.model_name == "en-b0": #we create the model object twice, because dataset is need to how to prepare data for exact data, meanwhile the model needs labels from dataset
        model = EfficientNet(animals_10_dataset.label_to_index)
    else:
        raise Exception("Invalid model_name")


    batch_size = args.batch_size

    train_dataloader = data.DataLoader(
        data.Subset(animals_10_dataset, animals_10_dataset.train_index),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    eval_dataloader = data.DataLoader(
        data.Subset(animals_10_dataset, np.concatenate([animals_10_dataset.eval_index, animals_10_dataset.test_index], axis = 0)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    model.train(train_dataloader, eval_dataloader, train_steps=args.train_steps)
    model.save(f"{args.model_checkpoint_dir}")

    mat, rep = model.eval_statistics(eval_dataloader)
    print(rep)