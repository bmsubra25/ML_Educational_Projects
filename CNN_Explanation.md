# Convolutional Neural Networks
  # Table of Contents
  - [Conceptual Explanation](#conceptual-explanation)
  - [Pooling](#pooling)
  - [Convolution](#convolution)
  - [Flattening and Processing](#flattening-and-processing)
  - [Notes on Backpropagation](#notes-on-back-propagation)
  - [Preface to Code](#preface-to-code)
  - [Sources](#sources)
    
## Conceptual Explanation

The following assumes you have some knowledge of:

- Calculus
- Derivatives in context
- Matrix–vector products
- A basic idea of Machine Learning
- Multi-Layer Perceptrons
- PyTorch tensors and their operations

![CNN Example](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQFkUWl-FqoWZZokQDo_2Xakb4eSpY8mEKrnQ&s)  
*Image from [ExplainThatStuff.com](https://www.explainthatstuff.com/how-convolutional-neural-networks-work.html), used for educational purposes.*


Convolutional Neural Networks are the most dominant modern architecture for processing images. They make use of linear layers and activation functions like MLPs (Multi-Layer Perceptrons), but need additional features in order to process data in the form of images. These main features are **Pooling Layers** and **Convolutional Layers**.

Images are represented as 3D tensors with shape **color channels × height × width** (often with an extra batch dimension at the front in training situations). The height and width values are given by the dimensions of the image, with each color channel corresponding to red, blue, or green (values in RGB coloring). These values are between 0 and 255 (often normalized to 1 during preprocessing), with colors represented as “mixtures” of these values (e.g., brown = (165,42,42), green = (0,255,0)). This is high-dimensional data, rich in features that cannot immediately be represented one-dimensionally, so many operations have to be performed to extract features and compress them until we can get a flat tensor.

---

## Pooling

![Max-Pooling Example](https://upload.wikimedia.org/wikipedia/commons/e/e9/Max_pooling.png)  
*Image by [Wikipedia](https://upload.wikimedia.org/wikipedia/commons/e/e9/Max_pooling.png), used under CC BY-SA.*

Pooling layers are layers that compress images into lower-dimensional form. There are three main kinds of pooling layers: Max-Pooling, Min-Pooling, and Average-Pooling. Max pooling layers split an image into equally sized sections, choosing the maximum value within each chunk and discarding the rest, reducing the spatial dimensionality of the image. Min-Pooling is the exact opposite of this process. Average-Pooling, on the other hand, averages out all the values within a chunk, replacing the chunk with the average value.

This allows for images to be compressed while keeping only their most distinct features, or to get an average of the sharpness of features in an area, all of which can be useful in many contexts.

---

## Convolution

![Convolutional Neural Network Example](https://upload.wikimedia.org/wikipedia/commons/b/bd/Convolutional_neural_network%2C_convolution_worked_example.png)  
*Image by [Wikipedia](https://upload.wikimedia.org/wikipedia/commons/b/bd/Convolutional_neural_network%2C_convolution_worked_example.png), used under CC BY-SA.*


Convolution is the main distinctive feature of CNNs. Convolution in CNNs is a process where a specific matrix slides over an image and multiplies overlapping values together. This gives a new set of values where features are emphasized based on the kernel’s structure, with the resulting values representing the degree of overlap of the kernel and the space it slides over.

In practice, the kernels you use can sharpen or blur images. In a Convolutional Neural Network, these are learnable, with a model learning what kernels are best applied to an image. Several kernels are used at once within the same layer, spanning all channels and producing unique feature maps (each with a separate matrix of values like the one above).

This is a simplified explanation of convolution, meant specifically for applications in Machine Learning. For more about convolution, 3Blue1Brown has a beautiful video, which is linked in the sources for this document.

![Feature Maps Example](https://www.tensorflow.org/static/tutorials/images/cnn_files/output_K3PAELE2eSU9_0.png)  
*Image by [TensorFlow](https://www.tensorflow.org/static/tutorials/images/cnn_files/output_K3PAELE2eSU9_0.png), used under CC BY-SA.*

---

## Flattening and Processing

Once pooling operations and convolutions are performed, the image is then flattened into a 1D vector of **channels × height × width** (excluding the batch dimension), and fed through MLP-style layers, producing outputs. Feedforward MLP layers need 1D input tensors and are still an extremely important component of a CNN for making predictions. The processes before this can be considered ways to make the data processable for the MLP layers. After this, backpropagation is performed.

Some modern architectures skip flattening and go directly to predictions. The way explained here, with flattening and MLP layers, is the standard method but not the only one.

---

## Notes on Back-Propagation

Backpropagation for this kind of network can be difficult to compute from scratch, so PyTorch’s automatic differentiation is typically used. PyTorch’s automatic differentiation keeps a computational graph, which allows you to easily access gradients for each parameter with respect to the loss without having to calculate them by hand. Even then, you can only use autograd with these operations if you use specific PyTorch functions (no manual indexing for pooling or convolution). It is possible to do manual backpropagation with convolutional neural networks, but unless one is attempting to specialize in image processing or wants to deeply understand CNNs, it is not typically done.

---

## Preface to Code

The code in this directory creates several classes that compose a CNN, wraps them in a model object, and trains it on the MNIST image dataset (a dataset consisting of 28 × 28 grayscale images with digits drawn from 0–9). This is a standard and simple example of a CNN application, with the code walked through line by line. Recreating classes from here and using them on more complex image processing problems could be a worthwhile learning endeavor.

Note: The `selections` list in the model validation section isn’t saved, even though it is mentioned in the notebook. This is intentional, to give the reader flexibility. If you’d like to save the model’s predictions for readability, you can simply add a `self.` before it so the predictions are stored as part of the model.

---

## Sources
  3Blue1Brown. “But What Is a Convolution?” YouTube, uploaded by 3Blue1Brown, 28 Oct. 2017, https://www.youtube.com/watch?v=KuXjwB4LzSA
  
  Futurology. “Convolutional Neural Networks Explained (CNN Visualized).” YouTube, uploaded by Futurology, 20 Jan. 2021, https://www.youtube.com/watch?v=pj9-rr1wDhM

  LeCun, Y., & Bengio, Y. (2015). An introduction to convolutional neural networks. arXiv preprint arXiv:1511.08458. Retrieved from https://arxiv.org/abs/1511.08458

  Li, Zewen, Wenjie Yang, Shouheng Peng, and Fan Liu. “A Survey of Convolutional Neural Networks: Analysis, Applications, and Prospects.” arXiv, 6 Apr. 2020, https://arxiv.org/abs/2004.02806

