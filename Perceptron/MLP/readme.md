# Multilayer Perceptron (MLP)

## Overview
A Multilayer Perceptron (MLP) is a class of artificial neural networks (ANNs) designed to model complex patterns in data. It is widely used for classification and regression tasks.

## Architecture
MLP consists of three types of layers:
1. **Input Layer** - Receives raw data inputs.
2. **Hidden Layers** - Consists of multiple fully connected layers that apply transformations.
3. **Output Layer** - Produces the final prediction.

Each neuron in a layer is connected to every neuron in the next layer, and each connection has an associated weight.

## Learning Process
The learning process in an MLP consists of the following steps:

### 1. Forward Propagation
   - Input data is passed through the network.
   - Each neuron computes a **weighted sum** of its inputs.
   - The result is passed through an **activation function** to introduce non-linearity (e.g., ReLU, Sigmoid, Tanh).

### 2. Loss Computation
   - The model output is compared to the actual target values using a **loss function**.
   - Common loss functions include:
     - Binary Cross Entropy (for binary classification)
     - Mean Squared Error (MSE) (for regression)
     - Categorical Cross Entropy (for multi-class classification)

### 3. Backpropagation
   - The loss is propagated backward through the network to adjust weights.
   - This involves:
     - **Gradient Calculation**: Partial derivatives of the loss with respect to each weight are computed.
     - **Error Propagation**: Gradients are passed backward through layers.
     - **Weight Update**: Weights are updated using gradient descent.

### 4. Optimization
   - Optimization algorithms adjust weights to minimize loss.
   - Common optimizers include:
     - **Stochastic Gradient Descent (SGD)**
     - **Adam Optimizer**

## Comparison: Perceptron vs. MLP
| Feature        | Perceptron  | MLP  |
|---------------|------------|------|
| Number of Layers | Single (No hidden layers) | Multiple (With hidden layers) |
| Activation Function | Step Function | ReLU, Sigmoid, Tanh, Softmax |
| Linearity | Only linearly separable problems | Can handle non-linear problems |
| Learning Algorithm | Perceptron Learning Rule | Backpropagation |
| Usage | Binary classification | Multi-class classification, Regression |

## Regularization in MLP
- **Dropout**: Randomly drops units during training to prevent overfitting.
- **L2 Regularization**: Adds a penalty term to the loss function to prevent large weight values.

## Model Evaluation
- The trained model is evaluated on test data using metrics such as accuracy, precision, recall, and F1-score.

## References
- [Analytics Vidhya](https://www.analyticsvidhya.com)
- [Towards Data Science](https://towardsdatascience.com)
- [Scikit-Learn Documentation](https://scikit-learn.org)
