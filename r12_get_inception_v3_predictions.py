# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Extract features for every 6th frame of video with Inception pretrained net
It will generate around 190 GB of additional files.
Note: Optimized to use with Theano backend
'''

import platform
import sys
import os

# gpu_use = 0
# if platform.processor() == 'Intel64 Family 6 Model 63 Stepping 2, GenuineIntel' or platform.processor() == 'Intel64 Family 6 Model 79 Stepping 1, GenuineIntel':
#     os.environ["THEANO_FLAGS"] = "device=gpu{},lib.cnmem=0.81,base_compiledir='C:\\\\Users\\\\user\\\\AppData\\\\Local\\\\Theano{}'".format(gpu_use, gpu_use)
#     if sys.version_info[1] > 4:
#         os.environ["KERAS_BACKEND"] = "tensorflow"
#         os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

from a00_common_functions import *


OUTPUT_INCEPTION_V3_FOLDER = OUTPUT_PATH + 'inception_v3_preds/'
if not os.path.isdir(OUTPUT_INCEPTION_V3_FOLDER):
    os.mkdir(OUTPUT_INCEPTION_V3_FOLDER)


def read_video_incep(f, sz=598, dim_ordering='th'):
    frame_mod = 6
    cap = cv2.VideoCapture(f)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    video = None
    while (cap.isOpened()):
        ret, frame = cap.read()
        if current_frame % frame_mod != 0:
            current_frame += 1
            continue
        if ret is False:
            break
        if video is None:
            video = np.zeros((1 + ((length - 1) // frame_mod), sz, sz, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (sz, sz), cv2.INTER_LANCZOS4)
        video[current_frame // frame_mod, :, :, :] = frame
        current_frame += 1

    if dim_ordering == 'th':
        video = np.transpose(video, (0, 3, 1, 2))
    return video


def get_inception_v3_predictions_for_train(reverse=False):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    import keras.backend as K
    K.set_image_dim_ordering('th')

    model = InceptionV3(include_top=False, input_shape=(3, 598, 598), weights='imagenet', pooling='avg')

    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename').index
    if reverse is True:
        train_labels = train_labels[::-1]
    # train_labels = ['FkJeaRnQg1.mp4']
    for t in train_labels:
        f = FULL_VIDEO_PATH + t
        if os.path.isfile(f):
            print('Go for {}...'.format(t))
            out_file = OUTPUT_INCEPTION_V3_FOLDER + t + '.pklz'
            if os.path.isfile(out_file) and 0:
                print('Already exists. Skip!')
                continue
            v = read_video_incep(f, 598)
            print(v.shape)
            v = preprocess_input(v.astype(np.float32))
            preds = model.predict(v)
            save_in_file(preds, out_file)


def get_inception_v3_predictions_for_test(reverse=False):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    import keras.backend as K
    K.set_image_dim_ordering('th')

    model = InceptionV3(include_top=False, input_shape=(3, 598, 598), weights='imagenet', pooling='avg')

    subm = pd.read_csv(INPUT_PATH + 'submission_format.csv', index_col='filename')
    test_files = list(subm.index.values)
    if reverse is True:
        test_files = test_files[::-1]
    # test_files = ['xeJ73n1gu6.mp4']
    for t in test_files:
        f = FULL_VIDEO_PATH + t
        if os.path.isfile(f):
            print('Go for {}...'.format(t))
            out_file = OUTPUT_INCEPTION_V3_FOLDER + t + '.pklz'
            if os.path.isfile(out_file):
                print('Already exists. Skip!')
                continue
            v = read_video_incep(f, 598)
            print(v.shape)
            v = preprocess_input(v.astype(np.float32))
            preds = model.predict(v)
            save_in_file(preds, out_file)


def get_inception_v3_predictions_for_other(reverse=False):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    import keras.backend as K
    K.set_image_dim_ordering('th')

    model = InceptionV3(include_top=False, input_shape=(3, 598, 598), weights='imagenet', pooling='avg')

    files = get_other_filelist()
    if reverse is True:
        files = files[::-1]
    # test_files = ['xeJ73n1gu6.mp4']
    for t in files:
        f = FULL_VIDEO_PATH + t
        if os.path.isfile(f):
            print('Go for {}...'.format(t))
            out_file = OUTPUT_INCEPTION_V3_FOLDER + t + '.pklz'
            if os.path.isfile(out_file):
                print('Already exists. Skip!')
                continue
            try:
                v = read_video_incep(f, 598)
                print(v.shape)
            except:
                print('Bad video...')
                continue
            v = preprocess_input(v.astype(np.float32))
            preds = model.predict(v)
            save_in_file(preds, out_file)


if __name__ == '__main__':
    get_inception_v3_predictions_for_train(False)
    get_inception_v3_predictions_for_test(False)
    get_inception_v3_predictions_for_other(True)
