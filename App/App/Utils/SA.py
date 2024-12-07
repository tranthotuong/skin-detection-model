#Soft Attention

from keras import backend as K
from keras.layers import Layer,InputSpec
from tensorflow.keras.utils import register_keras_serializable
import keras.layers as kl
import tensorflow as tf
print(tf.__version__)

class SoftAttention(Layer):
    def __init__(self,ch,m,concat_with_x=False,aggregate=False,**kwargs):
        self.channels=int(ch)
        self.multiheads = m
        self.aggregate_channels = aggregate
        self.concat_input_with_scaled = concat_with_x

        
        super(SoftAttention,self).__init__(**kwargs)

    def build(self,input_shape):

        self.i_shape = input_shape

        kernel_shape_conv3d = (self.channels, 3, 3) + (1, self.multiheads) # DHWC
    
        self.out_attention_maps_shape = input_shape[0:1]+(self.multiheads,)+input_shape[1:-1]
        
        if self.aggregate_channels==False:

            self.out_features_shape = input_shape[:-1]+(input_shape[-1]+(input_shape[-1]*self.multiheads),)
        else:
            if self.concat_input_with_scaled:
                self.out_features_shape = input_shape[:-1]+(input_shape[-1]*2,)
            else:
                self.out_features_shape = input_shape
        

        self.kernel_conv3d = self.add_weight(shape=kernel_shape_conv3d,
                                        initializer='he_uniform',
                                        name='kernel_conv3d')
        self.bias_conv3d = self.add_weight(shape=(self.multiheads,),
                                      initializer='zeros',
                                      name='bias_conv3d')

        super(SoftAttention, self).build(input_shape)

    def call(self, x):
        # Get input dimensions
        batch_size = tf.shape(x)[0]
        height, width, channels = self.i_shape[1], self.i_shape[2], self.i_shape[3]

        # Expand input tensor for 3D convolution
        exp_x = tf.expand_dims(x, axis=-1)  # Shape: (batch_size, height, width, channels, 1)

        # Apply 3D convolution
        c3d = tf.nn.conv3d(
            input=exp_x,
            filters=self.kernel_conv3d,
            strides=[1, 1, 1, 1, 1],
            padding='SAME',
            data_format='NDHWC'
        )
        conv3d = tf.nn.bias_add(c3d, self.bias_conv3d)
        conv3d = tf.nn.relu(conv3d)  # Shape: (batch_size, height, width, channels, multiheads)

        # Reshape to match input dimensions
        conv3d = tf.transpose(conv3d, perm=[0, 1, 2, 4, 3])  # Move multiheads to the channel dimension
        conv3d = tf.reshape(conv3d, shape=(batch_size, height, width, channels, self.multiheads))  # Shape: (batch_size, height, width, channels, multiheads)

        # Compute attention maps
        softmax_alpha = tf.nn.softmax(conv3d, axis=-1)  # Shape: (batch_size, height, width, channels, multiheads)

        # Aggregation logic
        if self.aggregate_channels:
            exp_softmax_alpha = tf.reduce_sum(softmax_alpha, axis=-1, keepdims=True)  # Aggregate across multiheads
            exp_softmax_alpha = tf.squeeze(exp_softmax_alpha, axis=-1)  # Remove the last axis for shape compatibility

            # Ensure shapes match for multiplication
            if exp_softmax_alpha.shape != x.shape:
                exp_softmax_alpha = tf.broadcast_to(exp_softmax_alpha, x.shape)

            u = tf.multiply(exp_softmax_alpha, x)  # Element-wise multiplication
        else:
            exp_softmax_alpha = tf.expand_dims(softmax_alpha, axis=-1)  # Shape: (batch_size, height, width, channels, multiheads, 1)
            x_exp = tf.expand_dims(x, axis=-2)  # Shape: (batch_size, height, width, 1, channels)

            # Ensure alignment of shapes for multiplication
            u = tf.multiply(exp_softmax_alpha, x_exp)  # Element-wise multiplication
            u = tf.reshape(u, shape=(batch_size, height, width, -1))  # Flatten multiheads into channels

        # Optionally concatenate input with scaled output
        if self.concat_input_with_scaled:
            o = tf.concat([u, x], axis=-1)  # Concatenate along channel axis
        else:
            o = u

        return [o, softmax_alpha]


    def compute_output_shape(self, input_shape): 
        return [self.out_features_shape, self.out_attention_maps_shape]

    
    def get_config(self):
        return super(SoftAttention,self).get_config()
 