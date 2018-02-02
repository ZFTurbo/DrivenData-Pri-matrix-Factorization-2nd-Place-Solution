# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'


def preprocess_batch(batch):
    batch /= 256.0
    batch -= 0.5
    return batch


def VGG_3D_24_56_56(class_num):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense
    from keras.layers.core import Activation, Dropout

    if K.image_dim_ordering() == 'th':
        inputs = Input((3, 24, 56, 56))
    else:
        inputs = Input((24, 56, 56, 3))

    filters = 8
    x_24_56_56 = Conv3D(2 * filters, (3, 3, 3), padding='same', activation='relu')(inputs)
    x_24_56_56 = Conv3D(2 * filters, (3, 3, 3), padding='same', activation='relu')(x_24_56_56)
    x_12_28_28 = MaxPooling3D(pool_size=(2, 2, 2))(x_24_56_56)

    x_12_28_28 = Conv3D(4 * filters, (3, 3, 3), padding='same', activation='relu')(x_12_28_28)
    x_12_28_28 = Conv3D(4 * filters, (3, 3, 3), padding='same', activation='relu')(x_12_28_28)
    x_6_14_14 = MaxPooling3D(pool_size=(2, 2, 2))(x_12_28_28)

    x_6_14_14 = Conv3D(8 * filters, (3, 3, 3), padding='same', activation='relu')(x_6_14_14)
    x_6_14_14 = Conv3D(8 * filters, (3, 3, 3), padding='same', activation='relu')(x_6_14_14)
    x_3_7_7 = MaxPooling3D(pool_size=(2, 2, 2))(x_6_14_14)

    x = Flatten()(x_3_7_7)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(class_num, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def VGG_3D_28_16_16_nano(class_num):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense
    from keras.layers.core import Activation, Dropout

    if K.image_dim_ordering() == 'th':
        inputs = Input((3, 27, 16, 16))
    else:
        inputs = Input((27, 16, 16, 3))

    filters = 16
    x_28_16_16 = Conv3D(2 * filters, (3, 3, 3), padding='same', activation='relu')(inputs)
    x_28_16_16 = Conv3D(2 * filters, (3, 3, 3), padding='same', activation='relu')(x_28_16_16)
    x_9_8_8 = MaxPooling3D(pool_size=(3, 2, 2))(x_28_16_16)

    x_9_8_8 = Conv3D(4 * filters, (3, 3, 3), padding='same', activation='relu')(x_9_8_8)
    x_9_8_8 = Conv3D(4 * filters, (3, 3, 3), padding='same', activation='relu')(x_9_8_8)
    x_3_4_4 = MaxPooling3D(pool_size=(3, 2, 2))(x_9_8_8)

    x_3_4_4 = Conv3D(8 * filters, (3, 3, 3), padding='same', activation='relu')(x_3_4_4)
    x_3_4_4 = Conv3D(8 * filters, (3, 3, 3), padding='same', activation='relu')(x_3_4_4)
    x_3_4_4 = Conv3D(8 * filters, (3, 3, 3), padding='same', activation='relu')(x_3_4_4)
    x_1_2_2 = MaxPooling3D(pool_size=(3, 2, 2))(x_3_4_4)

    x = Flatten()(x_1_2_2)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(class_num, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def VGG_3D_28_16_16_nano_LSTM(class_num):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, TimeDistributed, LSTM
    from keras.layers.core import Activation, Dropout

    if K.image_dim_ordering() == 'th':
        inputs = Input((28, 3, 16, 16))
    else:
        inputs = Input((28, 16, 16, 3))

    filters = 16

    x_28_16_16 = TimeDistributed(Conv2D(2 * filters, (3, 3), padding='same', activation='relu'))(inputs)
    x_28_16_16 = TimeDistributed(Conv2D(2 * filters, (3, 3), padding='same', activation='relu'))(x_28_16_16)
    x_28_8_8 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x_28_16_16)

    x_28_8_8 = TimeDistributed(Conv2D(4 * filters, (3, 3), padding='same', activation='relu'))(x_28_8_8)
    x_28_8_8 = TimeDistributed(Conv2D(4 * filters, (3, 3), padding='same', activation='relu'))(x_28_8_8)
    x_28_4_4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x_28_8_8)

    x = TimeDistributed(Flatten())(x_28_4_4)
    x = LSTM(112, return_sequences=True, dropout=0.5)(x)
    x = Dense(112, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(class_num, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs, outputs=x)
    print(model.summary())
    return model


def VGG_3D_28_16_16_nano_LSTM_v2(class_num):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, TimeDistributed, LSTM
    from keras.layers.core import Activation, Dropout

    if K.image_dim_ordering() == 'th':
        inputs = Input((28, 3, 16, 16))
    else:
        inputs = Input((28, 16, 16, 3))

    filters = 16

    x_28_16_16 = TimeDistributed(Conv2D(2 * filters, (3, 3), padding='same', activation='relu'))(inputs)
    x_28_16_16 = TimeDistributed(Conv2D(2 * filters, (3, 3), padding='same', activation='relu'))(x_28_16_16)
    x_28_8_8 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x_28_16_16)

    x_28_8_8 = TimeDistributed(Conv2D(4 * filters, (3, 3), padding='same', activation='relu'))(x_28_8_8)
    x_28_8_8 = TimeDistributed(Conv2D(4 * filters, (3, 3), padding='same', activation='relu'))(x_28_8_8)
    x_28_4_4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x_28_8_8)

    x_28_4_4 = TimeDistributed(Conv2D(8 * filters, (3, 3), padding='same', activation='relu'))(x_28_4_4)
    x_28_4_4 = TimeDistributed(Conv2D(8 * filters, (3, 3), padding='same', activation='relu'))(x_28_4_4)
    x_28_2_2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x_28_4_4)

    x_28_2_2 = TimeDistributed(Conv2D(16 * filters, (3, 3), padding='same', activation='relu'))(x_28_2_2)
    x_28_2_2 = TimeDistributed(Conv2D(16 * filters, (3, 3), padding='same', activation='relu'))(x_28_2_2)
    x_28_1_1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x_28_2_2)

    x = TimeDistributed(Flatten())(x_28_1_1)
    x = LSTM(112, return_sequences=False, dropout=0.5)(x)
    x = Dense(112, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(class_num, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs, outputs=x)
    print(model.summary())
    return model


def VGG_3D_28_16_16_nano_ConvLSTM(class_num):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling3D, Flatten, Dense, ConvLSTM2D
    from keras.layers.core import Activation, Dropout

    if K.image_dim_ordering() == 'th':
        inputs = Input((28, 3, 16, 16))
    else:
        inputs = Input((28, 16, 16, 3))

    filters = 16

    x_28_16_16 = ConvLSTM2D(2 * filters, (3, 3), padding='same', activation='relu', return_sequences=True)(inputs)
    x_28_16_16 = ConvLSTM2D(2 * filters, (3, 3), padding='same', activation='relu', return_sequences=True)(x_28_16_16)
    x_28_8_8 = MaxPooling3D(pool_size=(1, 2, 2))(x_28_16_16)

    x_28_8_8 = ConvLSTM2D(4 * filters, (3, 3), padding='same', activation='relu', return_sequences=True)(x_28_8_8)
    x_28_8_8 = ConvLSTM2D(4 * filters, (3, 3), padding='same', activation='relu', return_sequences=True)(x_28_8_8)
    x_28_4_4 = MaxPooling3D(pool_size=(1, 2, 2))(x_28_8_8)

    x_28_4_4 = ConvLSTM2D(8 * filters, (3, 3), padding='same', activation='relu', return_sequences=True)(x_28_4_4)
    x_28_4_4 = ConvLSTM2D(8 * filters, (3, 3), padding='same', activation='relu', return_sequences=True)(x_28_4_4)
    x_28_2_2 = MaxPooling3D(pool_size=(1, 2, 2))(x_28_4_4)

    x_28_2_2 = ConvLSTM2D(16 * filters, (3, 3), padding='same', activation='relu', return_sequences=True)(x_28_2_2)
    x_28_2_2 = ConvLSTM2D(16 * filters, (3, 3), padding='same', activation='relu', return_sequences=True)(x_28_2_2)
    x_28_1_1 = MaxPooling3D(pool_size=(1, 2, 2))(x_28_2_2)

    x = Flatten()(x_28_1_1)
    x = Dense(128)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(class_num, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs, outputs=x)
    print(model.summary())
    return model


def VGG_3D_24_56_56_large(class_num):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense
    from keras.layers.core import Activation, Dropout

    if K.image_dim_ordering() == 'th':
        inputs = Input((3, 24, 56, 56))
    else:
        inputs = Input((24, 56, 56, 3))

    filters = 16
    x_24_56_56 = Conv3D(2 * filters, (3, 3, 3), padding='same', activation='relu')(inputs)
    x_24_56_56 = Conv3D(2 * filters, (3, 3, 3), padding='same', activation='relu')(x_24_56_56)
    x_12_28_28 = MaxPooling3D(pool_size=(2, 2, 2))(x_24_56_56)

    x_12_28_28 = Conv3D(4 * filters, (3, 3, 3), padding='same', activation='relu')(x_12_28_28)
    x_12_28_28 = Conv3D(4 * filters, (3, 3, 3), padding='same', activation='relu')(x_12_28_28)
    x_6_14_14 = MaxPooling3D(pool_size=(2, 2, 2))(x_12_28_28)

    x_6_14_14 = Conv3D(8 * filters, (3, 3, 3), padding='same', activation='relu')(x_6_14_14)
    x_6_14_14 = Conv3D(8 * filters, (3, 3, 3), padding='same', activation='relu')(x_6_14_14)
    x_3_7_7 = MaxPooling3D(pool_size=(2, 2, 2))(x_6_14_14)

    x = Flatten()(x_3_7_7)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(class_num, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def VGG_3D_56_224_224_large(class_num):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense
    from keras.layers.core import Activation, Dropout

    if K.image_dim_ordering() == 'th':
        inputs = Input((3, 56, 224, 224))
    else:
        inputs = Input((56, 224, 224, 3))

    filters = 4
    x_56_224_224 = Conv3D(1 * filters, (3, 3, 3), padding='same', activation='relu')(inputs)
    # x_56_224_224 = Conv3D(1 * filters, (3, 3, 3), padding='same', activation='relu')(x_56_224_224)
    x_28_112_112 = MaxPooling3D(pool_size=(3, 2, 2))(x_56_224_224)

    x_28_112_112 = Conv3D(4 * filters, (3, 3, 3), padding='same', activation='relu')(x_28_112_112)
    # x_28_112_112 = Conv3D(4 * filters, (3, 3, 3), padding='same', activation='relu')(x_28_112_112)
    x_14_56_56 = MaxPooling3D(pool_size=(2, 2, 2))(x_28_112_112)

    x_14_56_56 = Conv3D(8 * filters, (3, 3, 3), padding='same', activation='relu')(x_14_56_56)
    x_14_56_56 = Conv3D(8 * filters, (3, 3, 3), padding='same', activation='relu')(x_14_56_56)
    x_7_28_28 = MaxPooling3D(pool_size=(2, 2, 2))(x_14_56_56)

    x_7_28_28 = Conv3D(16 * filters, (3, 3, 3), padding='same', activation='relu')(x_7_28_28)
    x_7_28_28 = Conv3D(16 * filters, (3, 3, 3), padding='same', activation='relu')(x_7_28_28)
    x_7_14_14 = MaxPooling3D(pool_size=(1, 2, 2))(x_7_28_28)

    x_7_14_14 = Conv3D(32 * filters, (3, 3, 3), padding='same', activation='relu')(x_7_14_14)
    x_7_14_14 = Conv3D(32 * filters, (3, 3, 3), padding='same', activation='relu')(x_7_14_14)
    x_7_14_14 = Conv3D(32 * filters, (3, 3, 3), padding='same', activation='relu')(x_7_14_14)
    x_7_7_7 = MaxPooling3D(pool_size=(1, 2, 2))(x_7_14_14)

    x = Flatten()(x_7_7_7)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(class_num, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def VGG_3D_54_224_224_large(class_num):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense
    from keras.layers.core import Activation, Dropout

    if K.image_dim_ordering() == 'th':
        inputs = Input((3, 54, 224, 224))
    else:
        inputs = Input((54, 224, 224, 3))

    filters = 4
    x_54_224_224 = Conv3D(1 * filters, (3, 3, 3), padding='same', activation='relu')(inputs)
    # x_54_224_224 = Conv3D(1 * filters, (3, 3, 3), padding='same', activation='relu')(x_54_224_224)
    x_18_112_112 = MaxPooling3D(pool_size=(3, 2, 2))(x_54_224_224)

    x_18_112_112 = Conv3D(8 * filters, (3, 3, 3), padding='same', activation='relu')(x_18_112_112)
    x_18_112_112 = Conv3D(8 * filters, (3, 3, 3), padding='same', activation='relu')(x_18_112_112)
    x_6_56_56 = MaxPooling3D(pool_size=(3, 2, 2))(x_18_112_112)

    x_6_56_56 = Conv3D(16 * filters, (3, 3, 3), padding='same', activation='relu')(x_6_56_56)
    x_6_56_56 = Conv3D(16 * filters, (3, 3, 3), padding='same', activation='relu')(x_6_56_56)
    x_3_28_28 = MaxPooling3D(pool_size=(2, 2, 2))(x_6_56_56)

    x_3_28_28 = Conv3D(32 * filters, (3, 3, 3), padding='same', activation='relu')(x_3_28_28)
    x_3_28_28 = Conv3D(32 * filters, (3, 3, 3), padding='same', activation='relu')(x_3_28_28)
    x_1_14_14 = MaxPooling3D(pool_size=(3, 2, 2))(x_3_28_28)

    x_1_14_14 = Conv3D(64 * filters, (3, 3, 3), padding='same', activation='relu')(x_1_14_14)
    x_1_14_14 = Conv3D(64 * filters, (3, 3, 3), padding='same', activation='relu')(x_1_14_14)
    x_1_14_14 = Conv3D(64 * filters, (3, 3, 3), padding='same', activation='relu')(x_1_14_14)
    x_1_7_7 = MaxPooling3D(pool_size=(1, 2, 2))(x_1_14_14)

    x = Flatten()(x_1_7_7)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(class_num, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def VGG_3D_54_112_112(class_num):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense
    from keras.layers.core import Activation, Dropout

    if K.image_dim_ordering() == 'th':
        inputs = Input((3, 54, 112, 112))
    else:
        inputs = Input((54, 112, 112, 3))

    filters = 4

    x_54_112_112 = Conv3D(4 * filters, (3, 3, 3), padding='same', activation='relu')(inputs)
    # x_54_112_112 = Conv3D(4 * filters, (3, 3, 3), padding='same', activation='relu')(x_54_112_112)
    x_18_56_56 = MaxPooling3D(pool_size=(3, 2, 2))(x_54_112_112)

    x_18_56_56 = Conv3D(8 * filters, (3, 3, 3), padding='same', activation='relu')(x_18_56_56)
    x_18_56_56 = Conv3D(8 * filters, (3, 3, 3), padding='same', activation='relu')(x_18_56_56)
    x_9_28_28 = MaxPooling3D(pool_size=(2, 2, 2))(x_18_56_56)

    x_9_28_28 = Conv3D(16 * filters, (3, 3, 3), padding='same', activation='relu')(x_9_28_28)
    x_9_28_28 = Conv3D(16 * filters, (3, 3, 3), padding='same', activation='relu')(x_9_28_28)
    x_3_14_14 = MaxPooling3D(pool_size=(3, 2, 2))(x_9_28_28)

    x_3_14_14 = Conv3D(32 * filters, (3, 3, 3), padding='same', activation='relu')(x_3_14_14)
    x_3_14_14 = Conv3D(32 * filters, (3, 3, 3), padding='same', activation='relu')(x_3_14_14)
    x_3_14_14 = Conv3D(32 * filters, (3, 3, 3), padding='same', activation='relu')(x_3_14_14)
    x_1_7_7 = MaxPooling3D(pool_size=(3, 2, 2))(x_3_14_14)

    x = Flatten()(x_1_7_7)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(class_num, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs, outputs=x)
    return model
