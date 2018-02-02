# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'


SIZE_OF_INPUT = 327680


def convBlock(inp, filter_size, kernel_size, num_conv=1, dropout=0.0):
    from keras.layers import Conv1D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.core import Dropout

    c1 = Conv1D(filter_size, kernel_size, padding='same', activation="relu")(inp)
    c1 = BatchNormalization()(c1)
    if num_conv > 1:
        c1 = Conv1D(filter_size, kernel_size, padding='same', activation="relu")(c1)
        c1 = BatchNormalization()(c1)
    if num_conv > 2:
        c1 = Conv1D(filter_size, kernel_size, padding='same', activation="relu")(c1)
        c1 = BatchNormalization()(c1)
    c1 = Dropout(dropout)(c1)
    return c1


def get_zf_simple_audio_model_v1():
    from keras.models import Model
    from keras.layers import Input, MaxPooling1D
    from keras.layers.core import Dense, Dropout, Flatten

    fltr = 16
    pool_size = 16
    kernel_size = 9

    inputs = Input((SIZE_OF_INPUT, 1))
    conv = convBlock(inputs, fltr, kernel_size, 1)
    pool = MaxPooling1D(pool_size)(conv)
    fltr *= 2

    conv = convBlock(pool, fltr, kernel_size, 2)
    pool = MaxPooling1D(pool_size)(conv)
    fltr *= 2

    conv = convBlock(pool, fltr, kernel_size, 2)
    pool = MaxPooling1D(pool_size)(conv)
    fltr *= 2

    conv = convBlock(pool, fltr, kernel_size, 3)
    pool = MaxPooling1D(pool_size)(conv)
    fltr *= 2

    flatten = Flatten()(pool)
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dropout(0.5)(dense1)
    dense3 = Dense(128, activation='relu')(dense2)
    dense4 = Dropout(0.5)(dense3)
    final = Dense(24, activation='sigmoid')(dense4)

    model = Model(inputs=inputs, outputs=final)

    return model
