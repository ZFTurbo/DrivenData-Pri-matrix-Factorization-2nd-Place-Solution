# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Validate GRU neural network for imagehashes and create train part for 2nd level features
'''


from a00_common_functions import *
from keras import __version__
print('Keras version: {}'.format(__version__))
from r21_create_keras_models_based_on_imagehashes import ZF_full_keras_model_GRU_ImageHash
import math


AUGMENTATION_SIZE = 3

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


def predict_generator(files, batch_size):
    global TRAIN_TABLE, IMAGE_HASHES

    if TRAIN_TABLE is None:
        TRAIN_TABLE = get_labels_dict()
    if IMAGE_HASHES is None:
        IMAGE_HASHES = read_image_hashes_train()

    sz_0 = 330
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


def validate_single_model(num_fold, valid_files, real_labels):
    global LABELS

    recalc = 0
    batch_size = 16
    augmentation_size = AUGMENTATION_SIZE
    cnn_type = 'ZF_full_keras_model_GRU_ImageHash'

    start_time = time.time()
    cache_path = CACHE_PATH_VALID + cnn_type + '_fold_{}.pklz'.format(num_fold)
    if not os.path.isfile(cache_path) or recalc == 1:
        import keras.backend as K
        print('Creating and compiling model [{}]...'.format(cnn_type))
        K.set_image_dim_ordering('th')
        model = ZF_full_keras_model_GRU_ImageHash()
        final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
        model.load_weights(final_model_path)

        steps = math.ceil(len(valid_files) / batch_size)
        print('Steps: {}'.format(steps))
        full_preds = model.predict_generator(
            generator=predict_generator(valid_files, batch_size),
            steps=steps,
            max_queue_size=10,
            verbose=2
        )
        save_in_file(full_preds, cache_path)
    else:
        print('Restore from cache: {}'.format(cache_path))
        full_preds = load_from_file(cache_path)

    if full_preds.shape[0] != augmentation_size * len(valid_files):
        print('Check predictions shape: {} != {}'.format(full_preds.shape, augmentation_size * len(valid_files)))
    print('Predictions shape: {}'.format(full_preds.shape))

    preds = []
    for j in range(len(valid_files)):
        p = full_preds[j * augmentation_size:(j + 1) * augmentation_size].copy().mean(axis=0)
        preds.append(p.copy())
    preds = np.array(preds)
    print(real_labels)
    print('Averaged predictions shape: {}'.format(preds.shape))
    print('Real labels shape: {}'.format(real_labels.shape))

    score = get_score(preds, real_labels)
    print('Fold {} score: {}'.format(num_fold, score))
    print('Time: {} sec'.format(time.time() - start_time))

    return score, preds


def run_validation_func(nfolds):
    FOLD_TO_CALC = [1, 2, 3, 4, 5]

    labels = get_labels_dict()
    files, kf = get_kfold_split(nfolds)
    num_fold = 0
    sum_score = 0
    total = 0
    full_preds = np.zeros((len(files), 24))
    for train_index, valid_index in kf:
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split valid: ', len(valid_index))

        if num_fold not in FOLD_TO_CALC:
            continue

        real_labels = []
        valid_files = np.array(files)[valid_index]
        for v in valid_files:
            real_labels.append(labels[v])
        real_labels = np.array(real_labels)

        score, preds = validate_single_model(num_fold, valid_files, real_labels)
        print(preds.shape)
        full_preds[valid_index, :] = preds
        sum_score += score
        total += 1

    # Calc score
    real_labels = []
    for v in files:
        real_labels.append(labels[v])
    real_labels = np.array(real_labels)
    score = get_score(full_preds, real_labels)

    # Save raw predictions
    file_path = FEATURES_PATH + 'ZF_full_keras_model_GRU_ImageHash_train.csv'
    files = np.expand_dims(np.array(files), axis=1)
    full_preds = np.concatenate((files, full_preds), axis=1)
    tbl = pd.DataFrame(full_preds, columns=['filename'] + ANIMAL_TYPE)
    tbl.to_csv(file_path, index=False)

    return score


if __name__ == '__main__':
    num_folds = 5
    score = run_validation_func(num_folds)
    print('Score: {}'.format(round(score, 6)))


'''
v1 
Fold 1 score: 0.06744854765971763
Fold 2 score: 0.06578884590569155
Fold 3 score: 0.06756453203809344
Fold 4 score: 0.06602620434268382
Fold 5 score: 0.06732205329384636
Score: 0.06683
'''