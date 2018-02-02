# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Train neural net models based on recurrent GRU layers and imagehashes for each frame of videos (phash and dhash)
'''


from a00_common_functions import *
from a00_augmentation_functions import *
import datetime
import random


TRAIN_TABLE = None
IMAGE_HASHES = None


def read_image_hashes_train():
    global IMAGE_HASHES
    start_time = time.time()
    print('Reading image hashes...')
    d0 = load_from_file(OUTPUT_PATH + 'image_hashes_train_part_0.pklz')
    d1 = load_from_file(OUTPUT_PATH + 'image_hashes_train_part_1.pklz')
    d2 = load_from_file(OUTPUT_PATH + 'image_hashes_train_part_2.pklz')
    d3 = load_from_file(OUTPUT_PATH + 'image_hashes_train_part_3.pklz')
    print('Merge 1')
    d01 = dict(d0, **d1)
    print('Merge 2')
    d23 = dict(d2, **d3)
    print('Merge 3')
    d = dict(d01, **d23)
    print('Reading image hashes complete... {} sec'.format(time.time() - start_time))
    return d


@threadsafe_generator
def batch_generator_train(files, batch_size, augment=True):
    global TRAIN_TABLE, IMAGE_HASHES

    if TRAIN_TABLE is None:
        TRAIN_TABLE = get_labels_dict()
    if IMAGE_HASHES is None:
        IMAGE_HASHES = read_image_hashes_train()

    sz_0 = 330

    while True:
        batch_files = np.random.choice(files, batch_size)

        video_list = []
        labels_list = []
        for i in range(len(batch_files)):
            label = TRAIN_TABLE[batch_files[i]]
            vd = IMAGE_HASHES[batch_files[i]]

            start_sh0 = random.randint(0, vd.shape[0] - sz_0)
            vd = vd[start_sh0:start_sh0+sz_0, :]
            # vd = vd.flatten()
            # vd = np.expand_dims(vd, axis=0)
            video_list.append(vd)
            labels_list.append(label)
        video_list = np.array(video_list, dtype=np.float32)
        labels_list = np.array(labels_list)

        yield video_list, labels_list


# Simple 1 LSTM layer
def ZF_full_keras_model_GRU_ImageHash():
    from keras.models import Model
    from keras.layers import Input, Dense, GRU
    from keras.layers.core import Dropout

    inputs1 = Input((330, 128))
    x = GRU(128, return_sequences=False, recurrent_dropout=0.5)(inputs1)
    x = Dropout(0.5)(x)
    x = Dense(24, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs1, outputs=x)
    print(model.summary())
    return model


def train_single_classification_model(num_fold, train_files, valid_files):
    from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
    from keras.optimizers import Adam, SGD, RMSprop

    restore = 1
    # K.set_image_dim_ordering('th')
    cnn_type = 'ZF_full_keras_model_GRU_ImageHash'
    print('Creating and compiling model [{}]...'.format(cnn_type))
    model = ZF_full_keras_model_GRU_ImageHash()

    final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
    cache_model_path = MODELS_PATH + '{}_temp_fold_{}.h5'.format(cnn_type, num_fold)
    cache_model_path_detailed = MODELS_PATH + '{}_temp_fold_{}_'.format(cnn_type, num_fold) + 'ep_{epoch:03d}_loss_{val_loss:.4f}.h5'
    if os.path.isfile(cache_model_path) and restore:
        print('Load model from last point: ', cache_model_path)
        model.load_weights(cache_model_path)
    else:
        print('Start training from begining')

    print('Fitting model...')

    optim_name = 'Adam'
    batch_size = 48
    learning_rate = 0.0001
    epochs = 2000
    patience = 50
    print('Batch size: {}'.format(batch_size))
    print('Model memory usage: {} GB'.format(get_model_memory_usage(batch_size, model)))
    print('Learning rate: {}'.format(learning_rate))
    steps_per_epoch = 200
    validation_steps = 200
    print('Samples train: {}, Samples valid: {}'.format(steps_per_epoch, validation_steps))

    if optim_name == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    elif optim_name == 'Adam':
        optim = Adam(lr=learning_rate)
    else:
        optim = RMSprop(lr=learning_rate)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint(cache_model_path, monitor='val_loss', save_best_only=True, verbose=0),
        # ModelCheckpoint(cache_model_path_detailed, monitor='val_loss', save_best_only=False, save_weights_only=True, verbose=0),
        CSVLogger(HISTORY_FOLDER_PATH + 'history_fold_{}_{}_lr_{}_optim_{}.csv'.format(num_fold, cnn_type, learning_rate, optim_name), append=True)
    ]

    history = model.fit_generator(generator=batch_generator_train(train_files, batch_size),
                  epochs=epochs,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=batch_generator_train(valid_files, batch_size),
                  validation_steps=validation_steps,
                  verbose=2,
                  max_queue_size=16,
                  callbacks=callbacks)

    min_loss = min(history.history['val_loss'])
    print('Minimum loss for given fold: ', min_loss)
    model.load_weights(cache_model_path)
    model.save(final_model_path)
    now = datetime.datetime.now()
    filename = HISTORY_FOLDER_PATH + 'history_{}_{}_{:.4f}_lr_{}_{}_weather.csv'.format(cnn_type, num_fold, min_loss, learning_rate, now.strftime("%Y-%m-%d-%H-%M"))
    pd.DataFrame(history.history).to_csv(filename, index=False)
    return min_loss


def run_cross_validation_create_models(nfolds):
    global FOLD_TO_CALC

    files, kfold_images_split = get_kfold_split(nfolds)
    files = np.array(files)
    num_fold = 0
    sum_score = 0
    for train_index, test_index in kfold_images_split:
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split frames train: ', len(train_index))
        print('Split frames valid: ', len(test_index))

        if 'FOLD_TO_CALC' in globals():
            if num_fold not in FOLD_TO_CALC:
                continue

        train_files = files[train_index]
        valid_files = files[test_index]
        score = train_single_classification_model(num_fold, train_files, valid_files)
        sum_score += score

    print('Avg loss: {}'.format(sum_score / nfolds))


if __name__ == '__main__':
    IMAGE_HASHES = read_image_hashes_train()
    run_cross_validation_create_models(5)


'''
Fold 1: 0.0657681702822
Fold 2: 0.0639512884058
Fold 3: 0.0656863708794
Fold 4: 0.0641085110977
Fold 5: 0.0656456260569
'''