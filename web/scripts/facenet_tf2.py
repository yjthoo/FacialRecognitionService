import tensorflow as tf
from scripts.utils import LRN2D, conv2d_bn

def facenet_tf2():

    # adapted from https://github.com/TessFerrandez/research-papers/tree/prod/facenet

    modelInput = tf.keras.Input(shape=(96, 96, 3))
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(modelInput)
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(x)
    x = tf.keras.layers.Lambda(LRN2D, name='lrn_1')(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), name='conv2')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = tf.keras.layers.Conv2D(192, (3, 3), name='conv3')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Lambda(LRN2D, name='lrn_2')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(x)

    # Inception3a
    inception_3a_3x3 = tf.keras.layers.Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
    inception_3a_3x3 = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
    inception_3a_3x3 = tf.keras.layers.Activation('relu')(inception_3a_3x3)
    inception_3a_3x3 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
    inception_3a_3x3 = tf.keras.layers.Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
    inception_3a_3x3 = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
    inception_3a_3x3 = tf.keras.layers.Activation('relu')(inception_3a_3x3)

    inception_3a_5x5 = tf.keras.layers.Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
    inception_3a_5x5 = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
    inception_3a_5x5 = tf.keras.layers.Activation('relu')(inception_3a_5x5)
    inception_3a_5x5 = tf.keras.layers.ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
    inception_3a_5x5 = tf.keras.layers.Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
    inception_3a_5x5 = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
    inception_3a_5x5 = tf.keras.layers.Activation('relu')(inception_3a_5x5)

    inception_3a_pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(x)
    inception_3a_pool = tf.keras.layers.Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
    inception_3a_pool = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)
    inception_3a_pool = tf.keras.layers.Activation('relu')(inception_3a_pool)
    inception_3a_pool = tf.keras.layers.ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)
    inception_3a_1x1 = tf.keras.layers.Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
    inception_3a_1x1 = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
    inception_3a_1x1 = tf.keras.layers.Activation('relu')(inception_3a_1x1)

    inception_3a = tf.keras.layers.concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)


    # Inception3b
    inception_3b_3x3 = tf.keras.layers.Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
    inception_3b_3x3 = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
    inception_3b_3x3 = tf.keras.layers.Activation('relu')(inception_3b_3x3)
    inception_3b_3x3 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
    inception_3b_3x3 = tf.keras.layers.Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
    inception_3b_3x3 = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
    inception_3b_3x3 = tf.keras.layers.Activation('relu')(inception_3b_3x3)

    inception_3b_5x5 = tf.keras.layers.Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
    inception_3b_5x5 = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
    inception_3b_5x5 = tf.keras.layers.Activation('relu')(inception_3b_5x5)
    inception_3b_5x5 = tf.keras.layers.ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
    inception_3b_5x5 = tf.keras.layers.Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
    inception_3b_5x5 = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
    inception_3b_5x5 = tf.keras.layers.Activation('relu')(inception_3b_5x5)

    inception_3b_pool = tf.keras.layers.Lambda(lambda x: x**2, name='power2_3b')(inception_3a)
    inception_3b_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
    inception_3b_pool = tf.keras.layers.Lambda(lambda x: x*9, name='mult9_3b')(inception_3b_pool)
    inception_3b_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sqrt(x), name='sqrt_3b')(inception_3b_pool)
    inception_3b_pool = tf.keras.layers.Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
    inception_3b_pool = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)
    inception_3b_pool = tf.keras.layers.Activation('relu')(inception_3b_pool)
    inception_3b_pool = tf.keras.layers.ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

    inception_3b_1x1 = tf.keras.layers.Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
    inception_3b_1x1 = tf.keras.layers.BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
    inception_3b_1x1 = tf.keras.layers.Activation('relu')(inception_3b_1x1)

    inception_3b = tf.keras.layers.concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)


    # Inception3c
    inception_3c_3x3 = conv2d_bn(inception_3b,
                                 layer='inception_3c_3x3',
                                 cv1_out=128,
                                 cv1_filter=(1, 1),
                                 cv2_out=256,
                                 cv2_filter=(3, 3),
                                 cv2_strides=(2, 2),
                                 padding=(1, 1))

    inception_3c_5x5 = conv2d_bn(inception_3b,
                                 layer='inception_3c_5x5',
                                 cv1_out=32,
                                 cv1_filter=(1, 1),
                                 cv2_out=64,
                                 cv2_filter=(5, 5),
                                 cv2_strides=(2, 2),
                                 padding=(2, 2))

    inception_3c_pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(inception_3b)
    inception_3c_pool = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

    inception_3c = tf.keras.layers.concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)


    #inception 4a
    inception_4a_3x3 = conv2d_bn(inception_3c,
                                 layer='inception_4a_3x3',
                                 cv1_out=96,
                                 cv1_filter=(1, 1),
                                 cv2_out=192,
                                 cv2_filter=(3, 3),
                                 cv2_strides=(1, 1),
                                 padding=(1, 1))
    inception_4a_5x5 = conv2d_bn(inception_3c,
                                 layer='inception_4a_5x5',
                                 cv1_out=32,
                                 cv1_filter=(1, 1),
                                 cv2_out=64,
                                 cv2_filter=(5, 5),
                                 cv2_strides=(1, 1),
                                 padding=(2, 2))
    inception_4a_pool = tf.keras.layers.Lambda(lambda x: x**2, name='power2_4a')(inception_3c)
    inception_4a_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
    inception_4a_pool = tf.keras.layers.Lambda(lambda x: x*9, name='mult9_4a')(inception_4a_pool)
    inception_4a_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sqrt(x), name='sqrt_4a')(inception_4a_pool)
    inception_4a_pool = conv2d_bn(inception_4a_pool,
                                  layer='inception_4a_pool',
                                  cv1_out=128,
                                  cv1_filter=(1, 1),
                                  padding=(2, 2))
    inception_4a_1x1 = conv2d_bn(inception_3c,
                                 layer='inception_4a_1x1',
                                 cv1_out=256,
                                 cv1_filter=(1, 1))
    inception_4a = tf.keras.layers.concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)


    #inception4e
    inception_4e_3x3 = conv2d_bn(inception_4a,
                                 layer='inception_4e_3x3',
                                 cv1_out=160,
                                 cv1_filter=(1, 1),
                                 cv2_out=256,
                                 cv2_filter=(3, 3),
                                 cv2_strides=(2, 2),
                                 padding=(1, 1))
    inception_4e_5x5 = conv2d_bn(inception_4a,
                                 layer='inception_4e_5x5',
                                 cv1_out=64,
                                 cv1_filter=(1, 1),
                                 cv2_out=128,
                                 cv2_filter=(5, 5),
                                 cv2_strides=(2, 2),
                                 padding=(2, 2))
    inception_4e_pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(inception_4a)
    inception_4e_pool = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

    inception_4e = tf.keras.layers.concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)


    #inception5a
    inception_5a_3x3 = conv2d_bn(inception_4e,
                                 layer='inception_5a_3x3',
                                 cv1_out=96,
                                 cv1_filter=(1, 1),
                                 cv2_out=384,
                                 cv2_filter=(3, 3),
                                 cv2_strides=(1, 1),
                                 padding=(1, 1))

    inception_5a_pool = tf.keras.layers.Lambda(lambda x: x**2, name='power2_5a')(inception_4e)
    inception_5a_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)
    inception_5a_pool = tf.keras.layers.Lambda(lambda x: x*9, name='mult9_5a')(inception_5a_pool)
    inception_5a_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sqrt(x), name='sqrt_5a')(inception_5a_pool)
    inception_5a_pool = conv2d_bn(inception_5a_pool,
                                  layer='inception_5a_pool',
                                  cv1_out=96,
                                  cv1_filter=(1, 1),
                                  padding=(1, 1))
    inception_5a_1x1 = conv2d_bn(inception_4e,
                                 layer='inception_5a_1x1',
                                 cv1_out=256,
                                 cv1_filter=(1, 1))

    inception_5a = tf.keras.layers.concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)


    #inception_5b
    inception_5b_3x3 = conv2d_bn(inception_5a,
                                 layer='inception_5b_3x3',
                                 cv1_out=96,
                                 cv1_filter=(1, 1),
                                 cv2_out=384,
                                 cv2_filter=(3, 3),
                                 cv2_strides=(1, 1),
                                 padding=(1, 1))
    inception_5b_pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(inception_5a)
    inception_5b_pool = conv2d_bn(inception_5b_pool,
                                  layer='inception_5b_pool',
                                  cv1_out=96,
                                  cv1_filter=(1, 1))
    inception_5b_pool = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

    inception_5b_1x1 = conv2d_bn(inception_5a,
                                 layer='inception_5b_1x1',
                                 cv1_out=256,
                                 cv1_filter=(1, 1))
    inception_5b = tf.keras.layers.concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)

    av_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
    reshape_layer = tf.keras.layers.Flatten()(av_pool)
    dense_layer = tf.keras.layers.Dense(128, name='dense_layer')(reshape_layer)
    norm_layer = tf.keras.layers.Lambda(lambda  x: tf.keras.backend.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)

    model = tf.keras.Model(inputs=[modelInput], outputs=norm_layer)

    return model
