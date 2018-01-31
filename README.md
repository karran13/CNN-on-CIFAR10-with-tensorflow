# CNN-on-CIFAR10-with-tensorflow
Work in progress implementation of alex-net on cifar10 with tensorflow and python

Implementation of Alex-net on CIFAR 10

**Data processing** 

**Methodology:**

-Images are first unpacked using pickle into arrays from the CIFAR10 files

-Pixel values are normalized to be from 0 to 1, from 0 to 255

-Converted into required format of Batch size x pixel x pixel x num_channels

-Passed through a function to perform randomized distortions on the image, on parameters such as orientation, brightness,hue,saturation etc

**Neural Network design and Loss Function**

The network architecture has: 

2 convolution layers interspersed with 2 max-pooling layers and 2 normalization layers and finally 2 ReLu layers and a fully connected layer to get the logits

The loss function is calculated using tensorflow's softmax_cross_entropy_with_logits function, which calculates the softmax from the logits and then calculates a cross entropy function using the softmax terms.

**Optimizer**

The optimizer used is an Adam Optimizer. The optimize function selects a random batch of training examples in order to reach convergence on the best possible values of the weights and biases, while printing batch training accuracy and loss for each batch.

It's found that the optimizer converges after around 8 hours in 150k iterations with an accuracy ~80%.


