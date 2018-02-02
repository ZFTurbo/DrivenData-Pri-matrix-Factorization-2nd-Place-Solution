# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Train neural net models based on Conv3D and micro dataset
'''

from a00_common_functions import *
from a00_augmentation_functions import *
import datetime
import shutil
import random
from a02_3D_model import *


TRAIN_TABLE = None


def batch_generator_train(files, batch_size, augment=True, dim_ordering = 'th'):
    global TRAIN_TABLE

    if TRAIN_TABLE is None:
        TRAIN_TABLE = get_labels_dict()

    while True:
        batch_files = np.random.choice(files, batch_size)

        video_list = []
        labels_list = []
        for i in range(len(batch_files)):
            f = INPUT_PATH + 'micro/' + batch_files[i]
            full_video = read_video(f)
            label = TRAIN_TABLE[batch_files[i]]

            start_sh0 = random.randint(0, full_video.shape[0] - 24)
            start_sh1 = random.randint(0, full_video.shape[1] - 56)
            start_sh2 = random.randint(0, full_video.shape[2] - 56)
            video = full_video[start_sh0:start_sh0+24, start_sh1:start_sh1+56, start_sh2:start_sh2+56, :]

            if augment:
                # random left-right flip
                if random.randint(0, 1) == 0:
                    video = video[:, :, ::-1, :]
                video = random_intensity_change_3D(video, 3)

            video_list.append(video)
            labels_list.append(label)
        video_list = np.array(video_list, dtype=np.float32)
        if dim_ordering == 'th':
            video_list = video_list.transpose((0, 4, 1, 2, 3))
        video_list = preprocess_batch(video_list)
        labels_list = np.array(labels_list)

        yield video_list, labels_list


def train_single_classification_model_full_frame(num_fold, train_files, valid_files, restore, optim_name='Adam'):
    from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
    from keras.optimizers import Adam, SGD
    import keras.backend as K

    K.set_image_dim_ordering('th')
    cnn_type = 'VGG_3D_24_56_56_v2'
    print('Creating and compiling model [{}]...'.format(cnn_type))
    model = VGG_3D_24_56_56(24)

    final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
    cache_model_path = MODELS_PATH + '{}_temp_fold_{}.h5'.format(cnn_type, num_fold)
    cache_model_path_detailed = MODELS_PATH + '{}_temp_fold_{}_'.format(cnn_type, num_fold) + 'ep_{epoch:03d}_loss_{val_loss:.4f}.h5'
    if os.path.isfile(cache_model_path) and restore:
        print('Load model from last point: ', cache_model_path)
        model.load_weights(cache_model_path)
    else:
        print('Start training from begining')

    print('Fitting model...')

    batch_size = 32
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
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint(cache_model_path, monitor='val_loss', save_best_only=True, verbose=0),
        ModelCheckpoint(cache_model_path_detailed, monitor='val_loss', save_best_only=False, save_weights_only=True, verbose=0),
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
    filename = HISTORY_FOLDER_PATH + 'history_{}_{}_{:.4f}_lr_{}_{}.csv'.format(cnn_type, num_fold, min_loss, learning_rate, now.strftime("%Y-%m-%d-%H-%M"))
    pd.DataFrame(history.history).to_csv(filename, index=False)
    return min_loss


def run_cross_validation_create_models(nfolds, restore=1, optim='Adam'):
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
        score = train_single_classification_model_full_frame(num_fold, train_files, valid_files, restore, optim)
        sum_score += score

    print('Avg loss: {}'.format(sum_score / nfolds))


if __name__ == '__main__':
    num_folds = 5
    run_cross_validation_create_models(num_folds, 0, 'Adam')
    run_cross_validation_create_models(num_folds, 1, 'SGD')

'''
v1 (Adam: 0.0001)
Fold 1: 0.036211631686
Fold 2: 0.036375067237
Fold 3: 0.036035662205
Fold 4: 0.036324784893
Fold 5: 0.035504640939

v2 (Additional training with SGD)
Fold 1: 0.0309606694896
Fold 2: 0.0341039754543
Fold 3: 0.0343223309331
Fold 4: 0.0328618826927
Fold 5: 0.0334771562601
'''