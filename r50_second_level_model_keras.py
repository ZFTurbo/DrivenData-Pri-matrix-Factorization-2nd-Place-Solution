# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Second level model, which uses all previously generated features, based on Keras classifier
Run 2 times with different parameters to increase solution stability
'''

import math
import datetime
from a00_common_functions import *


def batch_generator_train_blender_random_sample(X, y, batch_size):
    rng = list(range(X.shape[0]))

    while True:
        index1 = random.sample(rng, batch_size)
        input1 = X[index1, :]
        output1 = y[index1, :]
        yield input1, output1


def ZF_keras_blender_v2(input_features):
    from keras.models import Model
    from keras.layers import Input, Dense
    from keras.layers.core import Dropout

    inputs1 = Input((input_features,))
    x = Dense(input_features, activation='tanh')(inputs1)
    x = Dropout(0.5)(x)
    x = Dense(input_features, activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(24, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs1, outputs=x)
    print(model.summary())
    return model


def ZF_keras_blender_v3(input_features):
    from keras.models import Model
    from keras.layers import Input, Dense
    from keras.layers.core import Dropout

    inputs1 = Input((input_features,))
    x = Dense(input_features // 2, activation='sigmoid')(inputs1)
    x = Dropout(0.5)(x)
    x = Dense(24, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs1, outputs=x)
    print(model.summary())
    return model


def create_keras_blender_model(train, real_labels, features, model_type):
    from keras import __version__
    import keras.backend as K
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.optimizers import Adam, SGD

    print('Keras version: {}'.format(__version__))
    start_time = time.time()
    restore = 0
    files, ret = get_kfold_split(5)
    files = np.array(files)

    model_list = []
    full_preds = np.zeros((len(files), 24), dtype=np.float32)
    counts = np.zeros((len(files), 24), dtype=np.float32)
    num_fold = 0
    for train_index, valid_index in ret:
        num_fold += 1
        print('Start fold {}'.format(num_fold))
        train_files = files[train_index]
        valid_files = files[valid_index]
        X_train = train.loc[train_files, features].copy()
        X_valid = train.loc[valid_files, features].copy()
        y_train = real_labels.loc[train_files, ANIMAL_TYPE].copy()
        y_valid = real_labels.loc[valid_files, ANIMAL_TYPE].copy()

        print('Train data:', X_train.shape, y_train.shape)
        print('Valid data:', X_valid.shape, y_valid.shape)

        # K.set_image_dim_ordering('th')
        if model_type == 1:
            cnn_type = 'ZF_keras_blender_v2'
            print('Creating and compiling model [{}]...'.format(cnn_type))
            final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
            cache_model_path = MODELS_PATH + '{}_temp_fold_{}.h5'.format(cnn_type, num_fold)
            model = ZF_keras_blender_v2(len(features))
        else:
            cnn_type = 'ZF_keras_blender_v3'
            print('Creating and compiling model [{}]...'.format(cnn_type))
            final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
            cache_model_path = MODELS_PATH + '{}_temp_fold_{}.h5'.format(cnn_type, num_fold)
            model = ZF_keras_blender_v3(len(features))

        optim_name = 'Adam'
        batch_size = 48
        learning_rate = 0.00005
        epochs = 10000
        patience = 50
        print('Batch size: {}'.format(batch_size))
        print('Model memory usage: {} GB'.format(get_model_memory_usage(batch_size, model)))
        print('Learning rate: {}'.format(learning_rate))
        steps_per_epoch = (X_train.shape[0] // batch_size)
        validation_steps = 2*(X_valid.shape[0] // batch_size)
        print('Steps train: {}, Steps valid: {}'.format(steps_per_epoch, validation_steps))

        if optim_name == 'SGD':
            optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            optim = Adam(lr=learning_rate)
        model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
            ModelCheckpoint(cache_model_path, monitor='val_loss', save_best_only=True, verbose=0),
        ]

        if restore == 1 and os.path.isfile(final_model_path):
            print('Skip training!')
            model.load_weights(final_model_path)
        else:
            history = model.fit_generator(generator=batch_generator_train_blender_random_sample(X_train.as_matrix().copy(), y_train.as_matrix().copy(), batch_size),
                                      epochs=epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      validation_data=batch_generator_train_blender_random_sample(X_valid.as_matrix().copy(), y_valid.as_matrix().copy(), batch_size),
                                      validation_steps=validation_steps,
                                      verbose=2,
                                      max_queue_size=16,
                                      callbacks=callbacks)

            min_loss = min(history.history['val_loss'])
            print('Minimum loss for given fold: ', min_loss)
            model.load_weights(cache_model_path)
            model.save(final_model_path)

            if 0:
                now = datetime.datetime.now()
                filename = HISTORY_FOLDER_PATH + 'history_{}_{}_{:.4f}_lr_{}_{}_weather.csv'.format(cnn_type, num_fold,
                                                                                                min_loss, learning_rate,
                                                                                                now.strftime(
                                                                                                    "%Y-%m-%d-%H-%M"))
                pd.DataFrame(history.history).to_csv(filename, index=False)

        pred = model.predict(X_valid.as_matrix().copy())
        full_preds[valid_index, :] += pred
        counts[valid_index, :] += 1
        score = get_score(pred, y_valid.as_matrix())
        print('Fold {} score: {}'.format(num_fold, score))
        model_list.append(model)

    full_preds /= counts
    real = real_labels[ANIMAL_TYPE].as_matrix()
    score = get_score(full_preds, real)
    print('Score: {}'.format(score))
    print('Time: {} sec'.format(time.time() - start_time))

    s = pd.DataFrame(files, columns=['filename'])
    for a in ANIMAL_TYPE:
        s[a] = 0.0
    s[ANIMAL_TYPE] = full_preds
    s.to_csv(SUBM_PATH + 'subm_{}_{}_train.csv'.format('keras_blender', model_type), index=False)

    return score, full_preds, model_list


def predict_with_keras_model(test, features, models_list):
    dtest = test[features].as_matrix().copy()
    full_preds = []
    for m in models_list:
        preds = m.predict(dtest)
        full_preds.append(preds)
    preds = np.array(full_preds).mean(axis=0)
    return preds


def get_readable_date(dt):
    return datetime.datetime.fromtimestamp(dt).strftime('%Y-%m-%d %H:%M:%S')


def remove_some_features(train, features):
    new_features = []
    for f in features:
        if train[f].min() < -1 or train[f].max() > 1:
            print('Feature {} removed...'.format(f))
        else:
            new_features.append(f)
    return new_features


def replace_columns(table, suffix):
    for a in ANIMAL_TYPE:
        new_name = a + '_' + suffix
        new_name = new_name.replace(' ', '_')
        new_name = new_name.replace('(', '_')
        new_name = new_name.replace(')', '_')
        table.rename(columns={a: new_name}, inplace=True)
    return table


def read_tables():
    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename')
    meta_1 = pd.read_csv(OUTPUT_PATH + 'train_metadata.csv', index_col='filename')
    meta_2 = pd.read_csv(OUTPUT_PATH + 'train_audio_metadata.csv', index_col='filename')
    meta_mini = pd.read_csv(OUTPUT_PATH + 'metadata_mini_dataset.csv', index_col='filename')
    meta_mini['modification_time_mini'] = meta_mini['modification_time']
    oof_3dvgg = pd.read_csv(FEATURES_PATH + 'VGG_3D_24_56_56_v2_train.csv', index_col='filename')
    oof_3dvgg = replace_columns(oof_3dvgg, '3dvgg')
    oof_resnet = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_for_resnet_v4_train.csv', index_col='filename')
    oof_resnet = replace_columns(oof_resnet, 'resnet')
    oof_inception = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_inception_v1_train.csv', index_col='filename')
    oof_inception = replace_columns(oof_inception, 'inception')
    oof_inception_lstm = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_inception_LSTM_v2_train.csv',
                                     index_col='filename')
    oof_inception_lstm = replace_columns(oof_inception_lstm, 'inception_lstm')
    oof_inception_gru = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_inception_GRU_v4_train.csv',
                                    index_col='filename')
    oof_inception_gru = replace_columns(oof_inception_gru, 'inception_gru')
    oof_audio = pd.read_csv(FEATURES_PATH + 'zf_simple_audio_model_v1_train.csv', index_col='filename')
    oof_audio = replace_columns(oof_audio, 'audio')
    image_hashes = pd.read_csv(OUTPUT_PATH + 'image_hashes_data_train.csv', index_col='filename')
    oof_im_hash_gru = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_GRU_ImageHash_train.csv', index_col='filename')
    oof_im_hash_gru = replace_columns(oof_im_hash_gru, 'im_hash_gru')
    oof_vgg16_gru = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_GRU_1024_VGG16_v11_train.csv',
                                index_col='filename')
    oof_vgg16_gru = replace_columns(oof_vgg16_gru, 'vgg16_gru_v11')

    train = pd.concat([train_labels, meta_1, meta_2], axis=1)

    train = pd.merge(train, meta_mini[['modification_time_mini']], left_index=True, right_index=True, how='left')
    train = pd.merge(train, oof_3dvgg, left_index=True, right_index=True, how='left')
    train = pd.merge(train, oof_resnet, left_index=True, right_index=True, how='left')
    train = pd.merge(train, oof_inception, left_index=True, right_index=True, how='left')
    train = pd.merge(train, oof_inception_lstm, left_index=True, right_index=True, how='left')
    train = pd.merge(train, oof_inception_gru, left_index=True, right_index=True, how='left')
    train = pd.merge(train, oof_audio, left_index=True, right_index=True, how='left')
    train = pd.merge(train, image_hashes, left_index=True, right_index=True, how='left')
    train = pd.merge(train, oof_im_hash_gru, left_index=True, right_index=True, how='left')
    train = pd.merge(train, oof_vgg16_gru, left_index=True, right_index=True, how='left')

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50]:
        neigbour_average = pd.read_csv(FEATURES_PATH + 'neighbours_inception_{}_average_pred_train.csv'.format(2*i), index_col='filename')
        neigbour_average = replace_columns(neigbour_average, '{}_inception_neighbour_avg'.format(i))
        train = pd.merge(train, neigbour_average, left_index=True, right_index=True, how='left')

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50]:
        neigbour_average = pd.read_csv(FEATURES_PATH + 'neighbours_VGG16_{}_average_pred_train.csv'.format(2*i), index_col='filename')
        neigbour_average = replace_columns(neigbour_average, '{}_VGG16_neighbour_avg'.format(i))
        train = pd.merge(train, neigbour_average, left_index=True, right_index=True, how='left')

    features = sorted(list(set(train.columns.values) - set(ANIMAL_TYPE)))
    features.remove('fps')
    features.remove('sample_rate')

    real_labels = train_labels[ANIMAL_TYPE].copy()
    train['target'] = -1
    for i in range(len(ANIMAL_TYPE)):
        a = ANIMAL_TYPE[i]
        train.loc[train[a] == 1, 'target'] = i
    train.fillna(-1, inplace=True)

    return train, real_labels, features


def read_tst_table():
    subm = pd.read_csv(INPUT_PATH + 'submission_format.csv', index_col='filename')
    meta_1 = pd.read_csv(OUTPUT_PATH + 'test_metadata.csv', index_col='filename')
    meta_2 = pd.read_csv(OUTPUT_PATH + 'test_audio_metadata.csv', index_col='filename')
    meta_mini = pd.read_csv(OUTPUT_PATH + 'metadata_mini_dataset.csv', index_col='filename')
    meta_mini['modification_time_mini'] = meta_mini['modification_time']
    oof_3dvgg = pd.read_csv(FEATURES_PATH + 'VGG_3D_24_56_56_v2_test.csv', index_col='filename')
    oof_3dvgg = replace_columns(oof_3dvgg, '3dvgg')
    oof_resnet = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_for_resnet_v4_test.csv', index_col='filename')
    oof_resnet = replace_columns(oof_resnet, 'resnet')
    oof_inception = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_inception_v1_test.csv', index_col='filename')
    oof_inception = replace_columns(oof_inception, 'inception')
    oof_inception_lstm = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_inception_LSTM_v2_test.csv',
                                     index_col='filename')
    oof_inception_lstm = replace_columns(oof_inception_lstm, 'inception_lstm')
    oof_inception_gru = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_inception_GRU_v4_test.csv',
                                    index_col='filename')
    oof_inception_gru = replace_columns(oof_inception_gru, 'inception_gru')
    oof_audio = pd.read_csv(FEATURES_PATH + 'zf_simple_audio_model_v1_test.csv', index_col='filename')
    oof_audio = replace_columns(oof_audio, 'audio')
    image_hashes = pd.read_csv(OUTPUT_PATH + 'image_hashes_data_test.csv', index_col='filename')
    oof_im_hash_gru = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_GRU_ImageHash_test.csv', index_col='filename')
    oof_im_hash_gru = replace_columns(oof_im_hash_gru, 'im_hash_gru')
    oof_vgg16_gru = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_GRU_1024_VGG16_v11_test.csv', index_col='filename')
    oof_vgg16_gru = replace_columns(oof_vgg16_gru, 'vgg16_gru_v11')

    test = pd.concat([subm, meta_1, meta_2], axis=1)

    test = pd.merge(test, meta_mini[['modification_time_mini']], left_index=True, right_index=True, how='left')
    test = pd.merge(test, oof_3dvgg, left_index=True, right_index=True, how='left')
    test = pd.merge(test, oof_resnet, left_index=True, right_index=True, how='left')
    test = pd.merge(test, oof_inception, left_index=True, right_index=True, how='left')
    test = pd.merge(test, oof_inception_lstm, left_index=True, right_index=True, how='left')
    test = pd.merge(test, oof_inception_gru, left_index=True, right_index=True, how='left')
    test = pd.merge(test, oof_audio, left_index=True, right_index=True, how='left')
    test = pd.merge(test, image_hashes, left_index=True, right_index=True, how='left')
    test = pd.merge(test, oof_im_hash_gru, left_index=True, right_index=True, how='left')
    test = pd.merge(test, oof_vgg16_gru, left_index=True, right_index=True, how='left')

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50]:
        neigbour_average = pd.read_csv(FEATURES_PATH + 'neighbours_inception_{}_average_pred_test.csv'.format(2*i), index_col='filename')
        neigbour_average = replace_columns(neigbour_average, '{}_inception_neighbour_avg'.format(i))
        test = pd.merge(test, neigbour_average, left_index=True, right_index=True, how='left')

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50]:
        neigbour_average = pd.read_csv(FEATURES_PATH + 'neighbours_VGG16_{}_average_pred_test.csv'.format(2*i), index_col='filename')
        neigbour_average = replace_columns(neigbour_average, '{}_VGG16_neighbour_avg'.format(i))
        test = pd.merge(test, neigbour_average, left_index=True, right_index=True, how='left')

    test.fillna(-1, inplace=True)
    return subm, test


def remade_table(train, table, features):

    if 1:
        train['unique_w_h'] = train['width']*10000 + train['height']
        w_h_uni = list(train['unique_w_h'].value_counts().index)
        table['unique_w_h'] = table['width'] * 10000 + table['height']
        for i, w in enumerate(w_h_uni):
            nm = 'size_w_h_{}'.format(i)
            table['size_w_h_{}'.format(i)] = 0
            table.loc[table['unique_w_h'] == w, nm] = 1
            features.append(nm)

    if 1:
        min_mod_time = train['modification_time'].min()
        max_mod_time = train['modification_time'].max()
        print(min_mod_time, max_mod_time, max_mod_time - min_mod_time, math.log2(max_mod_time - min_mod_time))
        count = int(math.ceil(math.log2(max_mod_time - min_mod_time + 1000)))
        for i in range(count):
            nm = 'modification_time_{}'.format(i)
            table[nm] = ((table['modification_time'] - min_mod_time + 1000) // (2 ** i)) % 2
            features.append(nm)

        min_mod_time = train['modification_time_mini'].min()
        max_mod_time = train['modification_time_mini'].max()
        print(min_mod_time, max_mod_time, max_mod_time - min_mod_time, math.log2(max_mod_time - min_mod_time))
        count = int(math.ceil(math.log2(max_mod_time - min_mod_time + 1000)))
        for i in range(count):
            nm = 'modification_time_mini{}'.format(i)
            table[nm] = ((table['modification_time_mini'] - min_mod_time + 1000) // (2 ** i)) % 2
            print(table[nm].value_counts())
            features.append(nm)

    if 1:
        min_mod_time = train['filesize'].min()
        max_mod_time = train['filesize'].max()
        print(min_mod_time, max_mod_time, max_mod_time - min_mod_time, math.log2(max_mod_time - min_mod_time))
        f = ['filesize']
        STORE_NUM = 50
        for i in range(STORE_NUM):
            nm = 'filesize_{}'.format(i)
            table[nm] = 0
            border1 = (i*20000000) / STORE_NUM
            border2 = ((i+1) * 20000000) / STORE_NUM
            if i == STORE_NUM-1:
                border2 = 10000000000
            table.loc[(table['filesize'] >= border1) & (table['filesize'] < border2), nm] = 1
            features.append(nm)
            f.append(nm)

    if 1:
        min_mod_time = train['filesize_audio'].min()
        max_mod_time = train['filesize_audio'].max()
        print(min_mod_time, max_mod_time, max_mod_time - min_mod_time, math.log2(max_mod_time - min_mod_time))
        f = ['filesize_audio']
        STORE_NUM = 50
        for i in range(STORE_NUM):
            nm = 'filesize_audio_{}'.format(i)
            table[nm] = 0
            border1 = (i*250000) / STORE_NUM
            border2 = ((i+1) * 250000) / STORE_NUM
            if i == STORE_NUM-1:
                border2 = 10000000000
            table.loc[(table['filesize'] >= border1) & (table['filesize'] < border2), nm] = 1
            features.append(nm)
            f.append(nm)

    return table, features


def run_keras(iter1):
    train, real_labels, features = read_tables()
    train, features = remade_table(train, train, features)

    features = remove_some_features(train, features)
    print('Features [{}]: {}'.format(len(features), features))
    gbm_type = 'keras_blender'
    score, valid_pred, model_list = create_keras_blender_model(train, real_labels, features, iter1)

    subm, test = read_tst_table()
    test, _ = remade_table(train, test, features.copy())
    preds = predict_with_keras_model(test, features, model_list)

    subm[ANIMAL_TYPE] = preds
    subm.to_csv(SUBM_PATH + 'subm_{}_{}.csv'.format(gbm_type, iter1))


if __name__ == '__main__':
    start_time = time.time()
    run_keras(1)
    run_keras(2)
    print("Elapsed time overall: %s seconds" % (time.time() - start_time))


'''
v1
Fold 1 score: 0.015657899015900822
Fold 2 score: 0.016037633911393928
Fold 3 score: 0.016532430151098742
Fold 4 score: 0.01616162141784269
Fold 5 score: 0.016577789515137307
Score: 0.016193443821 LB: 0.014632 (Success)

v2
Fold 1 score: 0.015074018978742549
Fold 2 score: 0.01555255359101372
Fold 3 score: 0.015957094486122876
Fold 4 score: 0.01556895309846875
Fold 5 score: 0.015519083961205887
Score: 0.015534

v3 Lower param network:
Fold 1 score: 0.014718619955914135
Fold 2 score: 0.015302019796005347
Fold 3 score: 0.015723252563834587
Fold 4 score: 0.015347507767085855
Fold 5 score: 0.015447161299913829
Score: 0.015307
'''