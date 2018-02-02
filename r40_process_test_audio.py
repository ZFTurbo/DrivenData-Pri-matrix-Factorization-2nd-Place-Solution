# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Process test data with Conv1D neural network for audio and create test part for 2nd level features
'''


from a00_common_functions import *
from a02_audio_models import *
import math


AUGMENTATION_SIZE = 3


def predict_generator(files, batch_size):

    count = 0
    start_time = time.time()
    log_point = batch_size * (1000 // batch_size)

    while True:
        if count + batch_size > len(files):
            batch_files = files[count:]
        else:
            batch_files = files[count:count + batch_size]

        data_list = []
        for i in range(len(batch_files)):
            f = OUTPUT_FULL_AUDIO_WAV_PATH + batch_files[i] + '.wav'
            if os.path.isfile(f):
                single_data = read_audio(f)
            else:
                single_data = np.zeros((SIZE_OF_INPUT, 1))

            for j in range(AUGMENTATION_SIZE):
                if single_data.shape[0] > SIZE_OF_INPUT:
                    delta = (single_data.shape[0] - SIZE_OF_INPUT) // (AUGMENTATION_SIZE - 1)
                    start_0 = j * delta
                    single_data = single_data[start_0:start_0 + SIZE_OF_INPUT]
                else:
                    tmp_arr = np.zeros((SIZE_OF_INPUT, 1))
                    delta = (SIZE_OF_INPUT - single_data.shape[0]) // (AUGMENTATION_SIZE - 1)
                    start_0 = j * delta
                    tmp_arr[start_0:start_0 + single_data.shape[0]] = single_data
                    single_data = tmp_arr.copy()

                data_list.append(single_data.copy())

        if len(data_list) > 0:
            data_list = np.array(data_list, dtype=np.float32)

        count += batch_size
        if count % log_point == 0:
            print('Processed {} from {} Elapsed time: {}'.format(count, len(files), round(time.time() - start_time, 2)))
        yield data_list


def process_tst(nfolds):
    FOLD_TO_CALC = [1, 2, 3, 4, 5]

    recalc = 1
    batch_size = 8
    augmentation_size = AUGMENTATION_SIZE
    cnn_type = 'get_zf_simple_audio_model_v1'
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
            import keras.backend as K
            K.set_image_dim_ordering('th')
            model = get_zf_simple_audio_model_v1()
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

    subm.to_csv(FEATURES_PATH + 'get_zf_simple_audio_model_v1_test.csv')


if __name__ == '__main__':
    num_folds = 5
    process_tst(num_folds)


'''
LS: 0.077082 LB: 0.075533
'''