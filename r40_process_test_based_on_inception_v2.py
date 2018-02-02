# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Process test data with LSTM neural network for inception_v3 data and create test part for 2nd level features
'''

from a00_common_functions import *
from r21_create_keras_models_based_on_inception_v2 import ZF_full_keras_model_inception_LSTM_v2
import math


AUGMENTATION_SIZE = 3


def predict_generator(files, batch_size):

    sz_0 = 56
    count = 0
    start_time = time.time()
    log_point = batch_size * (1000 // batch_size)

    while True:
        if count + batch_size > len(files):
            batch_files = files[count:]
        else:
            batch_files = files[count:count + batch_size]

        video_list = []
        for i in range(len(batch_files)):
            f = OUTPUT_PATH + 'inception_v3_preds/' + batch_files[i] + '.pklz'
            try:
                arr = load_from_file(f)
            except:
                print('Error in {}'.format(f))
                continue
            for j in range(AUGMENTATION_SIZE):
                delta = (arr.shape[0] - sz_0) // (AUGMENTATION_SIZE - 1)
                start_sh0 = j*delta
                vd = arr[start_sh0:start_sh0 + sz_0, :]
                if vd.shape != (56, 2048):
                    print('Problem with: {}'.format(batch_files[i]))
                # vd = vd.flatten()
                # vd = np.expand_dims(vd, axis=0)
                video_list.append(vd.copy())

        if len(video_list) > 0:
            try:
                video_list = np.array(video_list, dtype=np.float32)
            except:
                print(len(video_list), video_list[0].shape)
                exit()

        count += batch_size
        if count % log_point == 0:
            print('Processed {} from {} Elapsed time: {}'.format(count, len(files), round(time.time() - start_time, 2)))
        yield video_list


def process_tst(nfolds):
    FOLD_TO_CALC = [1, 2, 3, 4, 5]

    recalc = 1
    batch_size = 8
    augmentation_size = AUGMENTATION_SIZE
    cnn_type = 'ZF_full_keras_model_inception_LSTM_v2'
    subm = pd.read_csv(INPUT_PATH + 'submission_format.csv', index_col='filename')
    files = list(subm.index.values)
    # files = ['xeJ73n1gu6.mp4']
    print('Files to process: {}'.format(len(files)))

    pred_list = []
    for num_fold in range(1, nfolds + 1):
        print('Start KFold number {} from {}'.format(num_fold, nfolds))

        if num_fold not in FOLD_TO_CALC:
            continue

        cache_path = CACHE_PATH + cnn_type + '_fold_{}.pklz'.format(num_fold)
        if not os.path.isfile(cache_path) or recalc == 1:
            # import keras.backend as K
            # K.set_image_dim_ordering('th')
            model = ZF_full_keras_model_inception_LSTM_v2()
            final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
            model.load_weights(final_model_path)

            steps = math.ceil(len(files) / batch_size)
            print('Steps: {}'.format(steps))
            full_preds = model.predict_generator(
                generator=predict_generator(files, batch_size),
                steps=steps,
                max_queue_size=10,
                verbose=2
            )

            if full_preds.shape[0] != augmentation_size * len(files):
                print('Check predictions shape: {} != {}'.format(full_preds.shape, augmentation_size * len(files)))
            print('Predictions shape: {}'.format(full_preds.shape))

            preds = []
            for j in range(len(files)):
                p = full_preds[j * augmentation_size:(j + 1) * augmentation_size].copy().mean(axis=0)
                preds.append(p.copy())
            preds = np.array(preds)

            save_in_file(preds, cache_path)
        else:
            preds = load_from_file(cache_path)

        pred_list.append(preds)
    pred_list = np.array(pred_list)
    print('Full predictions shape: {}'.format(pred_list.shape))
    preds = pred_list.mean(axis=0)
    print('Averaged predictions shape: {}'.format(preds.shape))

    for i in range(preds.shape[0]):
        subm.loc[files[i], ANIMAL_TYPE] = preds[i]

    subm.to_csv(FEATURES_PATH + 'ZF_full_keras_model_inception_LSTM_v2_test.csv')


def process_other_files(nfolds):
    FOLD_TO_CALC = [1, 2, 3, 4, 5]

    recalc = 1
    batch_size = 8
    augmentation_size = AUGMENTATION_SIZE
    cnn_type = 'ZF_full_keras_model_inception_LSTM_v2'
    files = get_other_filelist()
    files_filtered = []
    for pf in files:
        f = OUTPUT_PATH + 'inception_v3_preds/' + pf + '.pklz'
        try:
            arr = load_from_file(f)
            if arr.shape[0] < 56 or arr.shape[1] != 2048:
                print('Shape error in {}: {}'.format(f, arr.shape))
                continue
        except:
            print('Error in {}'.format(f))
            continue
        files_filtered.append(pf)
    files = files_filtered
    print('Files to process: {}'.format(len(files)))

    pred_list = []
    for num_fold in range(1, nfolds + 1):
        print('Start KFold number {} from {}'.format(num_fold, nfolds))

        if num_fold not in FOLD_TO_CALC:
            continue

        cache_path = CACHE_PATH + cnn_type + '_other_fold_{}.pklz'.format(num_fold)
        if not os.path.isfile(cache_path) or recalc == 1:
            # import keras.backend as K
            # K.set_image_dim_ordering('th')
            model = ZF_full_keras_model_inception_LSTM_v2()
            final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
            model.load_weights(final_model_path)

            steps = math.ceil(len(files) / batch_size)
            print('Steps: {}'.format(steps))
            full_preds = model.predict_generator(
                generator=predict_generator(files, batch_size),
                steps=steps,
                max_queue_size=10,
                verbose=2
            )

            if full_preds.shape[0] != augmentation_size * len(files):
                print('Check predictions shape: {} != {}'.format(full_preds.shape, augmentation_size * len(files)))
            print('Predictions shape: {}'.format(full_preds.shape))

            preds = []
            for j in range(len(files)):
                p = full_preds[j * augmentation_size:(j + 1) * augmentation_size].copy().mean(axis=0)
                preds.append(p.copy())
            preds = np.array(preds)

            save_in_file(preds, cache_path)
        else:
            preds = load_from_file(cache_path)

        pred_list.append(preds)
    pred_list = np.array(pred_list)
    print('Full predictions shape: {}'.format(pred_list.shape))
    preds = pred_list.mean(axis=0)
    print('Averaged predictions shape: {}'.format(preds.shape))

    # Save raw predictions
    file_path = FEATURES_PATH + 'ZF_full_keras_model_inception_LSTM_v2_other.csv'
    files = np.expand_dims(np.array(files), axis=1)
    full_preds = np.concatenate((files, preds), axis=1)
    tbl = pd.DataFrame(full_preds, columns=['filename'] + ANIMAL_TYPE)
    tbl.to_csv(file_path, index=False)


if __name__ == '__main__':
    num_folds = 5
    process_tst(num_folds)
    process_other_files(num_folds)

'''
LS: 0.029055 LB: 0.025962
'''