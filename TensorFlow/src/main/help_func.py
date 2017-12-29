'''
Created on Dec 30, 2017

@author: Binki
'''

import tensorflow as tf
import numpy as np

class processHelp(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        
        
    def _variable_on_cpu(self,name, shape, initializer):
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
            return var


    def _variable_with_weight_decay(self,name, shape, stddev, wd):
        dtype = tf.float32
        var = self._variable_on_cpu(
                               name,
                               shape,
                               tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        return var

    def pre_process_image(self,image, training):
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

    def pre_process(self,images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
        images = tf.map_fn(lambda image: self.pre_process_image(image, training), images)

        return images
