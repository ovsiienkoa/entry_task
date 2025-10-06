I don't have any thoughts about the first task, so I wrote everywhere pluses except for the few points, where my solution a bit differs from instructions 
#
Task 1: MNIST Image Classification (OOP/Strategy Pattern)
General requirements for the test:

● The source code should be written in Python 3. (+)

● The code should be clear for understanding and well-commented. (+)

● All solutions should be put into the GitHub repository. Each task should:(+)

○ be in a separate folder.(+)

○ contain its own readme file with a solution explanation and details on how to set up the project.(+)

○ requirements.txt with all libraries used in the solution.(+)

● All the documentation, comments, and other text information around the project should be written in English.(+)

● Demo that should be represented like a Jupyter Notebook and contain examples of how your solution is working including a description of the edge cases.(+)


Task 1. Image classification + OOP

In this task, you need to use a publicly available simple MNIST dataset and build 3 classification

models around it. It should be the following models:

1) Random Forest;(+)

2) Feed-Forward Neural Network;(+)

3) Convolutional Neural Network;(+)

Each model should be a separate class that implements MnistClassifierInterface with 2 abstract methods - train and predict.(+)
Finally, each of your three models should be hidden under another MnistClassifier class.(+)
MnistClassifer takes an algorithm as an input parameter.(+)

Possible values for the algorithm are: cnn, rf, and nn for the three models described above.

The solution should contain:

● Interface for models called MnistClassifierInterface.(+)

● 3 classes (1 for each model) that implement MnistClassifierInterface.(~)

● MnistClassifier, which takes as an input parameter the name of the algorithm and provides predictions with exactly the same structure (inputs and outputs) not depending on the selected algorithm.(+)

#
3. Project Setup

3.1. Dependencies
The required libraries can be installed using the requirements.txt file (not included here, but must be created):

pip install -r requirements.txt

conda install --yes -c pytorch -c nvidia -c conda-forge --file requirements.txt

The necessary dependencies are:

notebook
numpy
scikit-learn
tensorboard
pytorch
torchvision
pytorch-cuda=12.1

3.2. Running the Code
Place the provided Python script (mnist_classifier.py or similar) in the task folder.

The script's main execution block (if __name__ == '__main__':) will automatically download the MNIST dataset to a local ./data directory if not already present.

Execute the script:

python mnist_classifier.py

The script will sequentially initialize, train, and test a single sample prediction for all three algorithms (rf, nn, cnn).
#
4. Demo and Monitoring
The training process for the PyTorch models (nn and cnn) runs for 10 epochs, while the Random Forest does not. Loss metrics (Train and Evaluation) for all models are automatically logged using torch.utils.tensorboard.SummaryWriter.

To view the training curves, navigate to the project directory and start the TensorBoard server:

tensorboard --logdir=runs

The Demo (Jupyter Notebook) contain examples of how to initialize and use the MnistClassifier wrapper class.
#
Solution:
Megabasic implementation of required models. 
The only difference is that I've made classes CNNMOdel and FFModel inherit from TorchBasedModel that inherits from MnistClassifierInterface. 
I decide to do so, because the only thing that differ them is model structure in nn.Layers

What could be improved:
* save/load methods;
* evaluation? I just don't really know if it makes sense, especially after seeing graphs in tensorboard where lines just laid down after 3rd epoch