# Multi-Layer Perceptrons
*(and a general introduction to Deep Learning, Gradient Descent, and Neural Network Optimization)*

# Table of Contents
- [Conceptual Explanation](#conceptual-explanation)
- [Forward Propagation](#forward-propagation)
- [Back-Propagation](#back-propagation)
- [Stochastic Gradient Descent](#stochastic-gradient-descent)
- [Example: Forward and Backward Pass](#example)
- [Preface to Code](#preface-to-code)
- [Sources](#sources)

## Conceptual Explanation

The following assumes you have some knowledge of:
- Calculus
- Derivatives in context
- Matrix–vector products
- A basic idea of Machine Learning


![Single-Layer Neural Network (blank)](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Single-Layer_Neural_Network-Vector-Blank.svg/1134px-Single-Layer_Neural_Network-Vector-Blank.svg.png)
*Image by [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Single-Layer_Neural_Network-Vector-Blank.svg), used under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).*

Multi-Layer Perceptrons form the backbone of deep learning. They were invented in 1958, and are still ubiquitously used in some form or another in modern Deep Learning models. In the future, they might become less prominent in favor of other more modern versions, but as of right now they are the first things that come to mind when thinking about artificial neural networks.

In biological neural networks, neurons receive inputs from other neurons by “dendrites” or branch like structures that branch out to other neurons. They then send outputs to other neurons through “axons.” These outputs come in the form of electrical signals, triggered by something called an action potential. When an action potential is “fired,” you can consider the neuron as “activated,” where it has outputs that are being passed to the neurons that it is connected to.

Multi-Layer Perceptrons are the most comparable neural networks to biological ones, where signals pass forward from layer to layer. In contrast to biological neural networks, the neurons “learn” how much they should activate (compared to activating being an all-or-nothing thing in real neurons). Eventually, the neural network learns over time the degree to which each “neuron” should activate, and can eventually use this to tailor activation based on the problem to be solved. Real biological neurons don’t work like this, but with parameters called “weights” and “biases,” each neuron can change its magnitude of activation for inputs over time, creating a flexible system for learning compared to a hard and set algorithm.

Multi-Layer Perceptrons can be used to solve problems from digit classification based on images, predicting housing prices, and classifying email as spam. They have a wide variety of applications, and are also used as components in other bigger AI models (such as ChatGPT).

A multi-layer perceptron consists of several “layers” which consist of neurons. These neurons each perform a computation, and then pass their outputs through an “activation function,” which simulates a neuron’s activation based on the previous inputs it receives (Note: this is not the same as an actual biological neuron’s activation, but is meant to mimic it, where a neuron “activates” to a certain level based on its inputs). After this, the output of said layer is passed forward to another, and so on until the final layer’s outputs are computed. You can compare this to a biological neural network, where each neuron is “fired” based on the inputs of all the neurons that connect to it. In a multi-layer perceptron, all the neurons are connected to all the neurons of the previous layer, and are “fully connected.” After this, we calculate the “Loss,” a way to measure how “incorrect” a neural network’s predictions are, using a function. The neural network then calculates the derivative of the loss with respect to its most recent layer’s output, updating the parameters of the most recent layer (weights, biases) based on the derivative, and passing the derivative of the loss with respect to the previous layer’s output (or the current layer’s input) backwards, so that each layer can update its parameters based on the derivative (how this is done will be explained later). This is done until the first layer has updated its parameters, in which case the network is ready for its next input. The goal of this process is to find some minimum of the loss with respect to the network’s parameters of each layer, which means that error is minimized and the model is making accurate predictions.

The first thing I just described was “Forward-Propagation,” the process of passing an input through a neural network/multi-layer perceptron with several layers; “Back-Propagation,” the process of passing the derivative (can also be called the “gradient” to be more accurate to multi-variable calculus) backwards; and “Gradient Descent,” the process of updating parameters of the neural network based on the derivative/gradient. Each will be explained in depth in this document.

### Forward-Propagation

As described earlier, forward propagation is the process of passing inputs through a neural network.

Each neuron performs a linear function, where it calculates its own “raw” activation. This process is done with a matrix–vector product between a list of all the weights of a layer and the layer’s inputs, and vector addition between the product of weights and inputs and the biases. (Keep note of the following equations; they will become very important.)

**Correct dimensional form (treat inputs as a row vector):**

- Let $I \in \mathbb{R}^{1 \times n}$ be the inputs,(1 x n vector of inputs)

- $W \in \mathbb{R}^{n \times m}$ the weights, (n x m vector of weights)

- $b \in \mathbb{R}^{1 \times m}$ the biases. (1 x m vector of biases)

Then the raw activation is:

$$
Z = I \cdot W + b
$$

This is simply a linear equation, where the inputs are multiplied by the weights, and added to the biases.

Then, its raw activation (what you get from the weights * inputs + biases equation) is passed through an activation function. The reason why these are used is because if there were only linear functions involved, neural networks would be unable to model nonlinear relationships (which are the most important ones). The simple act of passing a raw activation through one makes all the difference here, and turns a linear process into a nonlinear one.

$$
A(Z) = f(Z)
$$

One example of how simplistic this can be is how one of the most preferred activation functions for use between hidden layers is called Rectified Linear Unit or “ReLU.” ReLU is a function which performs something simple: if an input is below 0, it becomes zero. If it is above zero, it stays the same.

$$
\mathrm{ReLU}(Z) = \max(0, Z)
$$


The fact that this is a nonlinear relationship suffices for the task. Other functions such as polynomial, sigmoid, or hyperbolic tangent functions can be used, but ReLU has managed to be one of the most prominent while being the most simple.

When reaching the final layer of a neural network, activation functions have to be more selective. One example of this is a classification problem. Say you want to train a multi-layer perceptron to be able to predict one of 10 numbers. The numbers can be treated as distinct separate categories, which would make using a standard activation function inconvenient since it produces number values. One activation function used specifically for prediction across categories is called softmax.

$$
\mathrm{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$


Softmax is a function that turns any vector (which is what your raw activations will look like) into a probability distribution. This is used specifically in the final layer (when predictions are necessary), and represents the probabilities of each index being the correct prediction. The model learns over time which index corresponds to which category (with gradient descent) and, if trained right, will eventually be able to “guess” the right category based on inputs. The details of the formula matter less than knowing what it does for classification contexts.

In short, forward propagation is the process of passing inputs between hidden layers to get an output. Each hidden layer computes its activations with “neurons,” which have weights and biases, and a non-linear activation function. This is done in a vectorized process where a matrix that represents weights for all neurons in the layer is multiplied with a vector that represents inputs and added to a bias vector, which is then all passed through an activation function. Eventually, this reaches the final layer, where specific activation functions can be used depending on the problem to be solved.

Now, we will move onto Back-Propagation.

---

### Back-Propagation

![Backpropagation in Neural Network](https://media.geeksforgeeks.org/wp-content/uploads/20250701163824448467/Backpropagation-in-Neural-Network-1.webp)
*Image by [GeeksforGeeks](https://www.geeksforgeeks.org/), used with permission.*


The first thing that has to be done for back-propagation is for loss to be computed. This is done with a specifically tailored function, similar to the final layer. You don’t need to memorize these formulas, but the idea of different loss functions being used for different problems is worth keeping in mind. Here are some examples:

- **Mean-Squared Error:** Used for regression problems (where a specific value has to be predicted).
- **Cross-Entropy Loss:** Used for classification problems (where a prediction has to be done over categories).
- **Mean-Absolute Error:** Used for regression problems.

**Mean Absolute Error (MAE):**

$$
MAE = \frac{1}{N} \sum_{i=1}^{N} \left| y_i - \hat{y}_i \right|
$$

**Mean Squared Error (MSE):**

$$
MSE = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \hat{y}_i \right)^2
$$

**Categorical Cross-Entropy (CCE):**

$$
CCE = - \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \,\log\\big(\hat{y}_{i,c}\big)
$$


The specific formulas are of little importance for the purposes of explanation. Regardless of which loss function you choose for your problem, loss can be represented like this:

$$
L(\text{Correct}, A) = f(\text{Correct}, A)
$$

Now, we calculate the derivative of the loss with respect to the most recent activation. This can be represented as $( \frac{dL}{dA} $).

The main problem here is that we need the derivative of the loss with respect to each layer’s weights and biases in order to do any useful updates. This means that we are trying to find $( \frac{dL}{dW} $) and $( \frac{dL}{db} $) (the derivative of the loss with respect to the weights and biases respectively). As you recall, the final layer’s activation is calculated like this:

$$
A(Z) = f(Z)
$$

And \( Z \) (the raw activation) is calculated like this (matching the dimensions from Forward-Propagation: inputs as a \(1 \times n\) row vector):

$$
Z = I \cdot W + b
$$
- $I \in \mathbb{R}^{1 \times n}$ (inputs)

- $W \in \mathbb{R}^{n \times m}$ (weights)

- $b \in \mathbb{R}^{1 \times m}$ (biases)

- $Z, A \in \mathbb{R}^{1 \times m}$

With the inputs being the activation vector of the previous layer and so on and so forth.

This gives us handy equations for finding the derivative of the loss with respect to the weights and biases, as \( A \) is a function of \( Z \), and \( Z \) is a function of \( W \) and \( b \):

$$
\frac{dL}{dW} = \frac{dL}{dA} \cdot \frac{dA}{dZ} \cdot \frac{dZ}{dW},
\qquad
\frac{dL}{db} = \frac{dL}{dA} \cdot \frac{dA}{dZ} \cdot \frac{dZ}{db}
$$


The specifics for how this is computed will differ based on the activation functions you choose, but this is the general way this is computed.

Now, this doesn’t address the problem of passing the gradients back, but it has a simple answer: we simply calculate the derivative of the loss with respect to the previous layer’s activation (which is identical to this layer’s input):

$\frac{dL}{dI} = \frac{dL}{dA} \cdot \frac{dA}{dZ} \cdot \frac{dZ}{dI}$

And we repeat the process.

Multi-Layer perceptrons are the best choice for demonstrating back-propagation, as more complex neural networks can get much messier in terms of how this is done, to the point where manually calculating gradients can be extremely cumbersome in comparison to using automatic differentiation, a process that tracks the gradients of each parameter with regards to another that libraries such as PyTorch or TensorFlow have.

---

### Stochastic Gradient Descent

Once you have these gradients, you can perform Stochastic Gradient Descent (abbreviated as SGD). A gradient (or derivative for that matter) represents the rate of change of a certain value with respect to another. One other thing they tell you is the direction of steepest ascent of a curve (as those familiar with multi-variable calculus would know).

The main goal of SGD is to find weights which generally minimize loss across many different examples. This is why we don’t simply calculate the absolute minimum of the loss with respect to the weights and biases and set them to that, because this would not generalize over many examples. This is also why there’s a “Stochastic” in SGD (stochastic is another word for random). The best way to do this is to define a learning rate (which is how much we want the model to “learn” from the gradient), and multiply it by the gradient, then subtract this product from each parameter:

$$
\text{Parameter}\ \mathrel{-=}\ \text{learning rate} \times \text{gradient}
$$

For the same purpose of generalization we use a learning rate factor with the gradient. If we didn’t do this, then the weights would bounce around a good settling point for the dataset it’s being trained on because of differences in specific examples.

Now, gradient descent can get much more complicated than this when serious optimization algorithms are used. These implement many things from warmups (lowering the learning rate until a certain number of training examples is hit), learning rate decay (lowering the learning rate over time to reduce overfitting, where the model learns too much about the training dataset and fails to generalize outside of it), and other protocols. 

---

### Example:

Let’s say we have a two-layer neural network, where we’re trying to calculate the price of a candy bar based on the amount of quarters and nickels that you paid for it with. The quarters and nickels can be represented as a vector, where \[quarters, nickels\] are two distinct features. For example, we can have the vector be \([5, 6]\) or 5 quarters and 6 nickels. We can set the first layer to have two neurons, and the final one to have one (so that we can get a singular output). The weights of the first layer would be represented as a $(2 \times 2$) matrix (so that we can get a 2D vector output to represent the activations of two “neurons”), and the biases to be a 1D vector.

Here’s an example of the neural network passing inputs through its first layer. The activation function can be **ReLU**. The numbers I will be using are far larger than they would be in a real neural network, so keep that in mind.

## Layer 1 (two neurons)

$$
W^{[1]} =
\begin{bmatrix}
2 & 2 \\
1 & 3
\end{bmatrix},
\quad
I =
\begin{bmatrix}
5 & 6
\end{bmatrix},
\quad
b^{[1]} =
\begin{bmatrix}
1 & 1
\end{bmatrix}
$$

Raw activation (inputs as a $1 \times 2$ row vector, weights $2 \times 2$, biases $1 \times 2$):

$$
\begin{aligned}
Z^{[1]} &= I \cdot W^{[1]} + b^{[1]} \\
&= \begin{bmatrix} 5 & 6 \end{bmatrix}
   \begin{bmatrix}
   2 & 2 \\
   1 & 3
   \end{bmatrix}
   {+} \begin{bmatrix} 1 & 1 \end{bmatrix} \\
&= \begin{bmatrix}
      5 \cdot 2 + 6 \cdot 1 & 5 \cdot 2 + 6 \cdot 3
   \end{bmatrix}
   {+} \begin{bmatrix} 1 & 1 \end{bmatrix} \\
&= \begin{bmatrix} 16 & 28 \end{bmatrix}
   {+} \begin{bmatrix} 1 & 1 \end{bmatrix} \\
&= \begin{bmatrix} 17 & 29 \end{bmatrix}
\end{aligned}
$$



$$
A^{[1]} = \text{ReLU}(Z^{[1]}) = \begin{bmatrix} 17 & 29 \end{bmatrix}
$$

---

## Layer 2 (final single neuron)

$$
W^{[2]} =
\begin{bmatrix}
0.5 \\
0
\end{bmatrix},
\quad
b^{[2]} = 3.5,
\quad
I^{[2]} \equiv A^{[1]} = \begin{bmatrix} 17 & 29 \end{bmatrix}
$$

$$
\begin{aligned}
Z^{[2]} &= A^{[1]} \cdot W^{[2]} + b^{[2]} \\
        &= \begin{bmatrix} 17 & 29 \end{bmatrix} 
           \begin{bmatrix} 0.5 \\ 0 \end{bmatrix} + 3.5 \\
        &= 8.5 + 3.5 \\
        &= 12
\end{aligned}
$$



$$
A^{[2]} = \text{ReLU}(Z^{[2]}) = 12
$$

And there you go — your neural network has predicted that 5 quarters and 6 nickels are worth **12 dollars**.

---

## Backward Pass

Assume your loss function is just the difference between true and predicted value. (Note: This is a toy loss for demonstration, and in practice would be a very poor loss function due to non-differentiability):

$$
L(T, A) = T - A
$$

$$
\frac{dL}{dA} = -1
$$

For ReLU:

$$
\frac{dA}{dZ} =
\begin{cases}
1, & Z > 0 \\
0, & Z \leq 0
\end{cases}
$$

---

### Final Layer Derivatives

For $Z^{[2]} = A^{[1]} \cdot W^{[2]} + b^{[2]}$:

$$
\frac{dZ^{[2]}}{dW^{[2]}} = (A^{[1]})^\top,
\quad
\frac{dZ^{[2]}}{db^{[2]}} = 1,
\quad
\frac{dZ^{[2]}}{dA^{[1]}} = (W^{[2]})^\top
$$

Since $Z^{[2]} = 12 > 0$, $\frac{dA^{[2]}}{dZ^{[2]}} = 1$:

$$
\frac{dL}{dW^{[2]}} = -
\begin{bmatrix}
17 \\
29
\end{bmatrix},
\quad
\frac{dL}{db^{[2]}} = -1,
\quad
\frac{dL}{dA^{[1]}} = \begin{bmatrix} -0.5 & 0 \end{bmatrix}
$$

---

### Layer 1 Backpropagation

$$
\frac{dL}{dZ^{[1]}} = \begin{bmatrix} -0.5 & 0 \end{bmatrix}
$$

$$
\frac{dL}{dW^{[1]}} =
\begin{bmatrix}
-2.5 & 0 \\
-3.0 & 0
\end{bmatrix},
\quad
\frac{dL}{db^{[1]}} =
\begin{bmatrix}
-0.5 & 0
\end{bmatrix}
$$

$$
\frac{dL}{dI} = \frac{dL}{dZ^{[1]}} \cdot (W^{[1]})^\top
$$

---

### Parameter Update Example

With learning rate $\eta = 0.01$:

$$
\begin{aligned}
W^{[2]}_{\text{new}} &= W^{[2]} - 0.01 \cdot \frac{dL}{dW^{[2]}} \\
&= \begin{bmatrix}0.5  \\ 0\end{bmatrix}
   {-} 0.01 \begin{bmatrix} -17 \\ -29 \end{bmatrix} \\
&= \begin{bmatrix}0.5  \\ 0\end{bmatrix}
   {-} \begin{bmatrix}-0.17  \\ -0.29 \end{bmatrix} \\
&= \begin{bmatrix}0.67  \\ 0.29\end{bmatrix}
\end{aligned}$$

And we would do this for the rest of the parameters before another forward pass. 


---

## Conclusion

This document has provided an in-depth explanation of both how Multi-Layer Perceptrons function and how deep learning works. It builds towards more complicated models (Transformers, Convolutional Neural Networks, etc.) and has the math illustrated and explained clearly. If you understand MLPs, you’ve already taken a large step toward being able to comprehend the beauty and scale of modern AI models.

### Preface to Code

The code examples I published here (the script and the Colab notebook) both have an MLP constructed from several different layers that is trained in a regression style (even though the problem lends itself better to classification). It is a simple example where each layer of the model is treated as a separate unit and backpropagation is illustrated in the training loop for it. Referring back to this document will be helpful for understanding the code (the code is also heavily commented). PyTorch is used to create arrays and tensors (which are just arrays/vectors with other functions).

## Sources:
   Sanderson, Grant. “Neural Networks.” 3Blue1Brown, https://www.3blue1brown.com/topics/neural-networks.

   “Multi-Layer Perceptron Learning in Tensorflow.” GeeksforGeeks, 23 July 2025, https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/.

   Baladram, Samy. “Multilayer Perceptron, Explained: A Visual Guide with Mini 2D Dataset.” Towards Data Science, 25 Oct. 2024, https://towardsdatascience.com/multilayer-perceptron-explained-a-visual-guide-with-mini-2d-dataset-0ae8100c5d1c/.

   Li, Z., Yang, W., Peng, S., & Liu, F. (2024). A Survey on State-of-the-art Deep Learning Applications and Techniques. arXiv preprint arXiv:2403.17561. Retrieved from https://arxiv.org/abs/2403.17561
