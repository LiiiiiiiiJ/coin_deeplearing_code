# -*- coding: utf-8 -*-
'''
@ author LiJ
@ version 1.0
@ date 2022 / 7 / 7 17: 28
@ Description: CBA model use on inteaction
'''

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda,RepeatVector

def buildModel():

    model_a = keras.Sequential([
            keras.layers.Conv2D(64, kernel_size=(4, 8), padding='valid', input_shape=[4,3000,1]),
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
    model_b = keras.Sequential([
            keras.layers.Conv2D(64, kernel_size=(4, 8), padding='valid', input_shape=[4,3000,1]),
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
    model_ = keras.Model(inputs=[model_a.input,model_b.input],
                         outputs=keras.layers.Concatenate(axis=1)([model_a.output,model_b.output]))
    
    l = keras.layers.Attention()([model_.output,model_.output])
    l = keras.layers.Flatten()(l)
    l = keras.layers.Dropout(0.3)(l)
    l = keras.layers.Dense(64,activation='relu')(l)
    l = keras.layers.Dense(1,activation='linear')(l)
    model = keras.Model(inputs=model_.inputs,
                         outputs=l)
    return model




















