# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Train neural net models based on Conv1D and audio data
'''


from a00_common_functions import *
from a00_augmentation_functions import *
import datetime
import random
from a02_audio_models import *


TRAIN_TABLE = None


def batch_generator_train(files, batch_size):
    global TRAIN_TABLE

    if TRAIN_TABLE is None:
        TRAIN_TABLE = get_labels_dict()

    while True:
        batch_files = np.random.choice(files, batch_size)

        input_data = []
        output_data = []
        for i in range(len(batch_files)):
            f = OUTPUT_FULL_AUDIO_WAV_PATH + batch_files[i] + '.wav'
            if os.path.isfile(f):
                single_data = read_audio(f)
            else:
                single_data = np.zeros((SIZE_OF_INPUT, 1))
            label = TRAIN_TABLE[batch_files[i]]

            # get random part
            if single_data.shape[0] > SIZE_OF_INPUT:
                start_0 = random.randint(0, single_data.shape[0] - SIZE_OF_INPUT)
                single_data = single_data[start_0:start_0 + SIZE_OF_INPUT]
            else:
                tmp_arr = np.zeros((SIZE_OF_INPUT, 1))
                start_0 = random.randint(0, SIZE_OF_INPUT - single_data.shape[0])
                tmp_arr[start_0:start_0 + single_data.shape[0]] = single_data
                single_data = tmp_arr.copy()

            input_data.append(single_data)
            output_data.append(label)

        input_data = np.array(input_data, dtype=np.float32)
        output_data = np.array(output_data, dtype=np.float32)

        yield input_data, output_data


def train_single_classification_model_full_frame(num_fold, train_files, valid_files):
    from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
    from keras.optimizers import Adam, SGD
    import keras.backend as K

    restore = 0
    # K.set_image_dim_ordering('th')
    cnn_type = 'get_zf_simple_audio_model_v1'
    print('Creating and compiling model [{}]...'.format(cnn_type))
    model = get_zf_simple_audio_model_v1()

    final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
    cache_model_path = MODELS_PATH + '{}_temp_fold_{}.h5'.format(cnn_type, num_fold)
    cache_model_path_detailed = MODELS_PATH + '{}_temp_fold_{}_'.format(cnn_type, num_fold) + 'ep_{epoch:03d}_loss_{val_loss:.4f}.h5'
    if os.path.isfile(cache_model_path) and restore:
        print('Load model from last point: ', cache_model_path)
        model.load_weights(cache_model_path)
    else:
        print('Start training from begining')

    print('Fitting model...')

    optim_name = 'SGD'
    batch_size = 48
    learning_rate = 0.001
    epochs = 200
    patience = 50
    print('Batch size: {}'.format(batch_size))
    print('Model memory usage: {} GB'.format(get_model_memory_usage(batch_size, model)))
    print('Learning rate: {}'.format(learning_rate))
    steps_per_epoch = 400
    validation_steps = 400
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
                  max_queue_size=10,
                  callbacks=callbacks)

    min_loss = min(history.history['val_loss'])
    print('Minimum loss for given fold: ', min_loss)
    model.load_weights(cache_model_path)
    model.save(final_model_path)
    now = datetime.datetime.now()
    filename = HISTORY_FOLDER_PATH + 'history_{}_{}_{:.4f}_lr_{}_{}_weather.csv'.format(cnn_type, num_fold, min_loss, learning_rate, now.strftime("%Y-%m-%d-%H-%M"))
    pd.DataFrame(history.history).to_csv(filename, index=False)
    return min_loss


def run_cross_validation_create_audio_models(nfolds):
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
        score = train_single_classification_model_full_frame(num_fold, train_files, valid_files)
        sum_score += score

    print('Avg loss: {}'.format(sum_score / nfolds))


if __name__ == '__main__':
    run_cross_validation_create_audio_models(5)


'''
Fold 1: 0.0781437471323
Fold 2: 0.0755217015091
Fold 3: 0.0768511359766
Fold 4: 0.0753326794133
Fold 5: 0.0769
'''