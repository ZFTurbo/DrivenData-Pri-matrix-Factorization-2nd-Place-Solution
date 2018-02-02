# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Validate Conv3D neural network and create train part for 2nd level features
'''


from a00_common_functions import *
from a02_3D_model import *
import math

AUGMENTATION_SIZE = 27


def get_video_augm_2(tf):
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


def validate_single_model(num_fold, valid_files, real_labels):
    global LABELS

    recalc = 1
    batch_size = 4
    augmentation_size = AUGMENTATION_SIZE
    cnn_type = 'VGG_3D_24_56_56_v2'

    start_time = time.time()
    cache_path = CACHE_PATH_VALID + cnn_type + '_fold_{}.pklz'.format(num_fold)
    if not os.path.isfile(cache_path) or recalc == 1:
        import keras.backend as K
        print('Creating and compiling model [{}]...'.format(cnn_type))
        K.set_image_dim_ordering('th')
        model = VGG_3D_24_56_56(24)
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

    # Save raw predictions (features)
    file_path = FEATURES_PATH + 'VGG_3D_24_56_56_v2_train.csv'
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
Fold 1 score: 0.045816375130474964
Fold 2 score: 0.04915124410256261
Fold 3 score: 0.04862028764342383
Fold 4 score: 0.047641518782681726
Fold 5 score: 0.09065013019852058
Score: 0.056376

v2
Fold 1 score: 0.033428753518684455
Fold 2 score: 0.034480034683949244
Fold 3 score: 0.03461519877074555
Fold 4 score: 0.034234021514233305
Fold 5 score: 0.033434585020989054
Score: 0.034039

v3
Fold 1 score: 0.029561852663646495
Fold 2 score: 0.03367055439121324
Fold 3 score: 0.033722480841632964
Fold 4 score: 0.032459142990508706
Fold 5 score: 0.03254286149364171
Score: 0.032391
'''