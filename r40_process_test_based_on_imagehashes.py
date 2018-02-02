# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Process test data with GRU neural network for imagehashes and create test part for 2nd level features
'''

from a00_common_functions import *
from r21_create_keras_models_based_on_imagehashes import ZF_full_keras_model_GRU_ImageHash
import math


AUGMENTATION_SIZE = 3
IMAGE_HASHES = None


def predict_generator(files, batch_size):
    global IMAGE_HASHES

    sz_0 = 330
    count = 0
    start_time = time.time()
    log_point = batch_size * (1000 // batch_size)

    if IMAGE_HASHES is None:
        IMAGE_HASHES = load_from_file(OUTPUT_PATH + 'image_hashes_test.pklz')

    while True:
        if count + batch_size > len(files):
            batch_files = files[count:]
        else:
            batch_files = files[count:count + batch_size]

        video_list = []
        for i in range(len(batch_files)):
            arr = IMAGE_HASHES[batch_files[i]]

            for j in range(AUGMENTATION_SIZE):
                delta = (arr.shape[0] - sz_0) // (AUGMENTATION_SIZE - 1)
                start_sh0 = j*delta
                vd = arr[start_sh0:start_sh0 + sz_0, :]
                # vd = vd.flatten()
                # vd = np.expand_dims(vd, axis=0)
                video_list.append(vd.copy())

        if len(video_list) > 0:
            video_list = np.array(video_list, dtype=np.float32)

        count += batch_size
        if count % log_point == 0:
            print('Processed {} from {} Elapsed time: {}'.format(count, len(files), round(time.time() - start_time, 2)))
        yield video_list


def process_tst(nfolds):
    FOLD_TO_CALC = [1, 2, 3, 4, 5]

    recalc = 1
    batch_size = 8
    augmentation_size = AUGMENTATION_SIZE
    cnn_type = 'ZF_full_keras_model_GRU_ImageHash'
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
            model = ZF_full_keras_model_GRU_ImageHash()
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

    subm.to_csv(FEATURES_PATH + 'ZF_full_keras_model_GRU_ImageHash_test.csv')


if __name__ == '__main__':
    num_folds = 5
    process_tst(num_folds)


'''
LS: 0.066830 LB: 0.065041
'''