import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from model import buildModel
import numpy as np
import pandas as pd
import pickle
from utils import show_train_history
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        tf.config.experimental
        # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.67)
        # config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        # session = tf.compat.v1.Session(config=config)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")


def train_and_pre(mode,zu_name):
    # -----------------load data----------------#
    np.random.seed(1)
    dna1 = np.expand_dims(pickle.load(open('../Data/encode/dna1_B73' + zu_name + '_500_single', 'rb')), 3)
    print(dna1.shape)

    file = pd.read_csv('../Data/exp_data/B73(' + zu_name + ')values_500_single.csv',encoding='gbk')
    ann = file['tracking_id'].values.astype('str')
    exp=file['value'].values.astype('float')

    # ------------build model & train-----------#

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def pearson(y_true, y_pred):
        x = y_true
        y = y_pred
        mx = K.mean(x, axis=0)
        my = K.mean(y, axis=0)
        xm, ym = x - mx, y - my
        r_num = K.sum(xm * ym)
        x_square_sum = K.sum(xm * xm)
        y_square_sum = K.sum(ym * ym)
        r_den = K.sqrt(x_square_sum * y_square_sum)
        r = r_num / r_den
        return K.mean(r)

    slide1 = int(len(ann) * 0.6)
    slide2 = int(len(ann) * 0.8)

    model = buildModel()
    #model.summary()
    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(1e-3),
                  metrics=['accuracy','mse', pearson])

    ckpt = keras.callbacks.ModelCheckpoint('../Model/'+mode+'/single_'+ zu_name + '.h5',
                                           monitor='val_mse',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='min',
                                           period=1)

    history = model.fit(dna1[:slide1],
                        exp[:slide1],
                        epochs=500,
                        batch_size=64,
                        validation_data=(dna1[slide1:slide2], exp[slide1:slide2]),
                        callbacks=[ckpt])

    show_train_history(history.history, s=zu_name + '_single', locate='../Result/pic/'+mode+'/')
   #    ROC(model, [dna1[slide1:], dna2[slide1:]], lab[slide1:],)

    # --------------------predict----------------------------#
    test_data = dna1[slide2:]
    test_label = exp[slide2:]

    rslt = model.predict(test_data)
    array = np.array([ann[slide2:],test_label, rslt[:, 0]]).T
    df = pd.DataFrame(array, columns=['Annotation','exp', 'pre'])
    df.to_csv('../Result/' + mode + '/single_' + zu_name + '.csv', index=False)

# # -----------------load data----------------#
#
# np.random.seed(1)
# dna1 = np.expand_dims(np.transpose(pickle.load(open('../Data/dna1', 'rb')), [0, 2, 1]), 3)
# dna2 = np.expand_dims(np.transpose(pickle.load(open('../Data/dna2', 'rb')), [0, 2, 1]), 3)
# print(dna1.shape)
#
# train = pd.read_csv('../Data/train.csv', encoding='utf-8')
# lab = train['label'].values.astype('int')
# lab = np.array(lab)
#
# # ------------build model & train-----------#
#
# def precision(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
#
# slide1 = int(len(lab) * 0.6)
# slide2 = int(len(lab) * 0.8)
#
# model = buildModel()
# model.summary()
# model.compile(loss='binary_crossentropy',
#               optimizer=keras.optimizers.Adam(1e-3),
#               metrics=['accuracy',precision])
#
# ckpt = keras.callbacks.ModelCheckpoint('../Model/cba.h5',
#                                        monitor='val_precision',
#                                        verbose=1,
#                                        save_best_only=True,
#                                        save_weights_only=False,
#                                        mode='max',
#                                        period=1)
#
# history = model.fit([dna1[:slide1], dna2[:slide1]],
#                     lab[:slide1],
#                     epochs=25,
#                     batch_size=64,
#                     validation_data=([dna1[slide1:slide2], dna2[slide1:slide2]], lab[slide1:slide2]),
#                     callbacks=[ckpt])
#
# show_train_history(history.history, s='test', locate='../Result/')
# ROC(model, [dna1[slide1:], dna2[slide1:]], lab[slide1:])
#
# #--------------------predict----------------------------#
# test_data=[dna1[slide2:], dna2[slide2:]]
# test_label=lab[slide2:]
#
# rslt=model.predict()
# array=np.array([test_label,rslt[:,0]]).T
# df = pd.DataFrame(array, columns=['label', 'result'])
# df.to_csv('output.csv', index=False)
