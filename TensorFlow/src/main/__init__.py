'''
Created on Dec 22, 2017

@author: Binki
'''

import tensorflow as tf
import numpy as np
import time

path = "C:\\Users\\Binki\\Downloads\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py\\data_batch_1"

# fix up data

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float32
    var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    return var


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data_dict = unpickle(path)

# for key,val in data_dict.items():
#     print("{} = {}".format(key, val))

data = 'data'
labels = 'labels'

data_key = data.encode(encoding='utf_8', errors='strict')
label_key = labels.encode(encoding='utf_8', errors='strict')


data_set_train = data_dict[data_key]

data_set_train = np.array(data_set_train)
data_set_train = np.reshape(data_set_train, [-1, 3, 32, 32])
data_set_train = data_set_train.transpose([0, 2, 3, 1])
print(len(data_set_train))

data_set_labels = np.array(data_dict[label_key])

num_classes = np.max(data_set_labels) + 1

data_set_labels = np.eye(num_classes, dtype=float)[data_set_labels]



# make tensorflow graph



def pre_process_image(image,training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.
    
    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
#        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        
        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    return image

def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)

    return images

def inference(images):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
      kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0)
      conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv1 = tf.nn.relu(pre_activation, name=scope.name)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[5, 5, 64, 64],
                                           stddev=5e-2,
                                           wd=0.0)
      conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv2 = tf.nn.relu(pre_activation, name=scope.name)
    
  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [64, -1])
    dim = reshape.get_shape()[1]
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  # local4
  with tf.variable_scope('local4') as scope:
      weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
      biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
      local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
      weights = _variable_with_weight_decay('weights', [192, num_classes],
                                          stddev=1 / 192.0, wd=0.0)
      biases = _variable_on_cpu('biases', [num_classes],
                              tf.constant_initializer(0.0))
      softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
  
  return softmax_linear


def create_network():
    
    images=x
    images=pre_process(images,training=True)
    logits=inference(images)
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits))
    y_pred=tf.nn.softmax(logits=logits)
    return y_pred,loss



def random_batch():
    # Number of images in the training-set.
    num_images = len(data_set_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=64,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = data_set_train[idx, :, :, :]
    y_batch = data_set_labels[idx, :]

    return x_batch, y_batch


def optimize(num_iterations):
    # Start-time used for printing time-usage below.

    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch,y_batch=random_batch()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true:y_batch }

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations (and last).
        if (i_global % 5 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))
       




x = tf.placeholder(tf.float32, shape=[64, 32, 32, 3], name='x')

y_true = tf.placeholder(tf.float32, shape=[64, num_classes])

y_true_cls = tf.arg_max(y_true, dimension=1)

global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

y_pred,loss=create_network()

y_pred_cls= tf.arg_max(y_pred,dimension=1)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss=loss, global_step=global_step)

correct_prediction=tf.equal(y_pred_cls, y_true_cls)

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

session=tf.Session()

session.run(tf.global_variables_initializer())

optimize(1000)





