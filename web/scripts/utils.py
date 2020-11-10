import tensorflow as tf

def LRN2D(x):
    # adapted from https://github.com/TessFerrandez/research-papers/tree/prod/facenet
    return tf.nn.local_response_normalization(x, depth_radius=5, bias=1, alpha=1e-4, beta=0.75)

def conv2d_bn(x,layer=None, cv1_out=None, cv1_filter=(1, 1), cv1_strides=(1, 1), cv2_out=None, cv2_filter=(3, 3),
              cv2_strides=(1, 1), padding=None):
    # adapted from https://github.com/TessFerrandez/research-papers/tree/prod/facenet

    num = '' if cv2_out == None else '1'
    tensor = tf.keras.layers.Conv2D(cv1_out, cv1_filter, strides=cv1_strides, name=layer+'_conv'+num)(x)
    tensor = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name=layer+'_bn'+num)(tensor)
    tensor = tf.keras.layers.Activation('relu')(tensor)
    if padding == None:
        return tensor

    tensor = tf.keras.layers.ZeroPadding2D(padding=padding)(tensor)

    if cv2_out == None:
        return tensor

    tensor = tf.keras.layers.Conv2D(cv2_out, cv2_filter, strides=cv2_strides, name=layer+'_conv'+'2')(tensor)
    tensor = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name=layer+'_bn'+'2')(tensor)
    tensor = tf.keras.layers.Activation('relu')(tensor)

    return tensor
