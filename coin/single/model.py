from tensorflow import keras
#import keras
#from keras_self_attention import SeqSelfAttention
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential


def buildModel():
    model_ = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=(4, 8), padding='valid', input_shape=[4, 3000, 1]),
        keras.layers.BatchNormalization(momentum=0.99, scale=False),
        keras.layers.ReLU(),
        keras.layers.Conv2D(64, kernel_size=(1, 8), padding='same'),
        keras.layers.BatchNormalization(momentum=0.99, scale=False),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(1, 8), strides=(1, 8), padding='same'),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(128, kernel_size=(1, 8), padding='same'),
        keras.layers.BatchNormalization(momentum=0.99, scale=False),
        keras.layers.ReLU(),
        keras.layers.Conv2D(128, kernel_size=(1, 8), padding='same'),
        keras.layers.BatchNormalization(momentum=0.99, scale=False),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(1, 8), strides=(1, 8), padding='same'),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(64, kernel_size=(1, 8), padding='same'),
        keras.layers.BatchNormalization(momentum=0.99, scale=False),
        keras.layers.ReLU(),

        keras.layers.Conv2D(64, kernel_size=(1, 8), padding='same'),
        keras.layers.BatchNormalization(momentum=0.99, scale=False),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(1, 8), strides=(1, 8), padding='same'),
        keras.layers.Dropout(0.3),

        keras.layers.Reshape((3, 128)),
        keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))
    ])
    # model_.add(SeqSelfAttention(attention_width=100,
    #                             attention_activation='sigmoid'
    #                             ))
    # model_.add(keras.layers.Flatten())
    # model_.add(keras.layers.Dropout(0.219900))
    # model_.add(keras.layers.Dense(64, activation='relu'))
    # model_.add(keras.layers.Dense(1, activation='linear'))
    # model = keras.Model(inputs=model_.inputs,
    #                     outputs=model_.outputs)

    l = keras.layers.Attention()([model_.output, model_.output])
    l = keras.layers.Flatten()(l)
    l = keras.layers.Dropout(0.3)(l)#0.332263
    l = keras.layers.Dense(64, activation='relu')(l)
    l = keras.layers.Dense(1, activation='linear')(l)
    model = keras.Model(inputs=model_.inputs,
                        outputs=l)
    return model


def buildModel1(d1, d2, d3, d4):
    model_ = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=(4, 8), padding='valid', input_shape=[4, 3002, 1]),
        keras.layers.BatchNormalization(momentum=0.99, scale=False),
        keras.layers.ReLU(),
        keras.layers.Conv2D(64, kernel_size=(1, 8), padding='same'),
        keras.layers.BatchNormalization(momentum=0.99, scale=False),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(1, 8), strides=(1, 8), padding='same'),
        keras.layers.Dropout(d1),

        keras.layers.Conv2D(128, kernel_size=(1, 8), padding='same'),
        keras.layers.BatchNormalization(momentum=0.99, scale=False),
        keras.layers.ReLU(),
        keras.layers.Conv2D(128, kernel_size=(1, 8), padding='same'),
        keras.layers.BatchNormalization(momentum=0.99, scale=False),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(1, 8), strides=(1, 8), padding='same'),
        keras.layers.Dropout(d2),

        keras.layers.Conv2D(64, kernel_size=(1, 8), padding='same'),
        keras.layers.BatchNormalization(momentum=0.99, scale=False),
        keras.layers.ReLU(),

        keras.layers.Conv2D(64, kernel_size=(1, 8), padding='same'),
        keras.layers.BatchNormalization(momentum=0.99, scale=False),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D(pool_size=(1, 8), strides=(1, 8), padding='same'),
        keras.layers.Dropout(d3),

        keras.layers.Reshape((3, 128)),
        keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))
    ])

    l = keras.layers.Attention()([model_.output, model_.output])
    l = keras.layers.Flatten()(l)
    l = keras.layers.Dropout(d4)(l)
    l = keras.layers.Dense(64, activation='relu')(l)
    l = keras.layers.Dense(1, activation='linear')(l)
    model = keras.Model(inputs=model_.inputs,
                        outputs=l)
    return model
