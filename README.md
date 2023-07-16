# MNIST-classsifier

This repository contains an implementation of a neural network model for classifying the MNIST dataset. The goal is to accurately classify handwritten digits (0-9) using an artificial intelligence approach.

## Key Features
- AI-based Approach: The classification model utilizes artificial neural networks (ANN) to learn and classify the MNIST dataset.
- Training and Testing: The model is trained on a portion of the dataset using a specified number of epochs. It undergoes backpropagation to adjust weights and biases, optimizing classification accuracy.
- Floating Point Operations (FLOPs): The repository provides insights into the FLOPs required during the training and inference process, helping to assess the computational complexity of the model.
- Epochs and Convergence: The model's performance can be evaluated by analyzing its convergence behavior over the specified number of epochs.
- Triangular Learning Rates: The repository explores the effectiveness of using triangular learning rate schedules, which dynamically adjust the learning rate during training, aiding in faster convergence and avoiding local minima.
- Normalization Techniques: The dataset is preprocessed using normalization techniques to ensure optimal input scaling, improving the model's performance.

## Usage
1. Clone the repository to your local machine.
2. Install the required dependencies mentioned in the `requirements.txt` file.
3. Run the training script to train the model on the MNIST dataset.
4. Evaluate the model's performance using the provided evaluation metrics.
5. Test the model by inputting custom images or the MNIST test dataset.

By utilizing artificial neural networks and incorporating normalization techniques, this repository aims to provide an efficient and accurate solution for MNIST digit classification tasks.
