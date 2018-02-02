# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Process test data with Conv3D neural network and create test part for 2nd level features
'''

from a00_common_functions import *
from a02_3D_model import *
import math

AUGMENTATION_SIZE = 27


def get_video_augm_v1(tf):
    im_big = np.zeros((AUGMENTATION_SIZE, 24, 56, 56, 3), dtype=np.float32)
    # P1
    im_big[0] = tf[0:24, 0:56, 0:56, :]
    # P2
    im_big[1] = tf[tf.shape[0]-24:tf.shape[0], tf.shape[1]-56:tf.shape[1], tf.shape[2]-56:tf.shape[2], :]
    return im_big


def get_video_augm(tf):
    sz_0 = 24
    sz_1 = 56
    sz_2 = 56

    im_big = np.zeros((AUGMENTATION_SIZE, sz_0, sz_1, sz_2, 3), dtype=np.float32)
    total = 0
    for i in [0, (tf.shape[0]-sz_0) // 2, tf.shape[0]-sz_0]:
        for j in [0, (tf.shape[1] - sz_1) // 2, tf.shape[1] - sz_1]:
            for k in [0, (tf.shape[2] - sz_2) // 2, tf.shape[2] - sz_2]:
                im_big[total] = tf[i:i+sz_0, j:j+sz_1, k:k+sz_2, :]
                total += 1

    return im_big


def predict_generator(files, batch_size, dim_ordering='th'):
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
            f = INPUT_PATH + 'micro/' + batch_files[i]
            full_video = read_video(f)
            video_list.append(get_video_augm(full_video))
        if len(video_list) > 0:
            video_list = np.concatenate(video_list)
            if dim_ordering == 'th':
                video_list = video_list.transpose((0, 4, 1, 2, 3))
            video_list = preprocess_batch(video_list)

        count += batch_size
        if count % log_point == 0:
            print('Processed {} from {} Elapsed time: {}'.format(count, len(files), round(time.time() - start_time, 2)))
        yield video_list


def process_tst(nfolds):
    FOLD_TO_CALC = [1, 2, 3, 4, 5]

    recalc = 1
    batch_size = 4
    augmentation_size = AUGMENTATION_SIZE
    cnn_type = 'VGG_3D_24_56_56_v2'
    subm = pd.read_csv(INPUT_PATH + 'submission_format.csv', index_col='filename')
    files = list(subm.index.values)
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
            model = VGG_3D_24_56_56(24)
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

    # This file can be used as submission as well
    subm.to_csv(FEATURES_PATH + 'VGG_3D_24_56_56_v2_test.csv')


if __name__ == '__main__':
    num_folds = 5
    process_tst(num_folds)

'''
LS: 0.032391 LB: 0.030836
'''