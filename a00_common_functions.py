# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

import numpy as np
import gzip
import bz2
import pickle
import os
import glob
import time
import cv2
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold
from collections import Counter, defaultdict
from multiprocessing import Process, Manager
from sklearn.metrics import accuracy_score, log_loss
import random
from scipy.io import wavfile

random.seed(2016)
np.random.seed(2016)

INPUT_PATH = '../input/'
OUTPUT_PATH = '../modified_data/'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
FEATURES_PATH = '../features/'
if not os.path.isdir(FEATURES_PATH):
    os.mkdir(FEATURES_PATH)
CACHE_PATH = "../cache_test/"
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)
CACHE_PATH_VALID = "../cache_valid/"
if not os.path.isdir(CACHE_PATH_VALID):
    os.mkdir(CACHE_PATH_VALID)
MODELS_PATH = '../models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
HISTORY_FOLDER_PATH = "../models/history/"
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH)
SUBM_PATH = '../subm/'
if not os.path.isdir(SUBM_PATH):
    os.mkdir(SUBM_PATH)

# FULL_VIDEO_PATH = INPUT_PATH + 'raw/'
FULL_VIDEO_PATH = 'D:/Projects/2017_10_DrivenData_Chimps/input/raw/'
OUTPUT_FULL_VIDEO_PATH = OUTPUT_PATH + 'full_video_250_250/'
OUTPUT_FULL_AUDIO_PATH = OUTPUT_PATH + 'full_audio/'
if not os.path.isdir(OUTPUT_FULL_VIDEO_PATH):
    os.mkdir(OUTPUT_FULL_VIDEO_PATH)
if not os.path.isdir(OUTPUT_FULL_AUDIO_PATH):
    os.mkdir(OUTPUT_FULL_AUDIO_PATH)
OUTPUT_FULL_VIDEO_PATH_125_60 = OUTPUT_PATH + 'full_video_125_60/'
if not os.path.isdir(OUTPUT_FULL_VIDEO_PATH_125_60):
    os.mkdir(OUTPUT_FULL_VIDEO_PATH_125_60)
OUTPUT_FULL_AUDIO_WAV_PATH = OUTPUT_PATH + 'full_audio_wav/'
if not os.path.isdir(OUTPUT_FULL_AUDIO_WAV_PATH):
    os.mkdir(OUTPUT_FULL_AUDIO_WAV_PATH)

ANIMAL_TYPE = ['bird', 'blank', 'cattle', 'chimpanzee', 'elephant', 'forest buffalo',
               'gorilla', 'hippopotamus', 'human', 'hyena', 'large ungulate', 'leopard',
               'lion', 'other (non-primate)', 'other (primate)', 'pangolin', 'porcupine',
               'reptile', 'rodent', 'small antelope', 'small cat', 'wild dog', 'duiker', 'hog']


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3), protocol=4)


def save_in_file_uncompessed(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=0), protocol=4)


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def get_labels_dict():
    cache_path = OUTPUT_PATH + 'labels_dict.pklz'
    if not os.path.isfile(cache_path):
        ret = dict()
        tt = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename')
        for index, row in tt.iterrows():
            ret[index] = list(row.values)
        save_in_file(ret, cache_path)
    else:
        ret = load_from_file(cache_path)
    return ret


def get_per_label_files(files, labels):
    ret = dict()
    for a in ANIMAL_TYPE:
        ret[a] = []
    for f in files:
        lbl = labels[f]
        for i in range(len(lbl)):
            if lbl[i] > 0:
                ret[ANIMAL_TYPE[i]].append(f)
    # for a in ANIMAL_TYPE:
    #    print(len(ret[a]))
    return ret


def read_video(f):
    cap = cv2.VideoCapture(f)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    video = None
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break
        if video is None:
            video = np.zeros((length, frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        video[current_frame, :, :, :] = frame
        current_frame += 1
    if current_frame != length and 0:
        print('Check some problem {} != {} ({})'.format(current_frame, length, os.path.basename(f)))

    return video


def get_kfold_split(nfolds):
    from sklearn.model_selection import KFold

    tt = pd.read_csv(INPUT_PATH + 'train_labels.csv')
    kfold_cache_path = OUTPUT_PATH + 'kfold_cache_{}.pklz'.format(nfolds)
    if not os.path.isfile(kfold_cache_path):
        files = sorted(list(tt['filename'].values))
        print('Unique files found: {}'.format(len(files)))

        ret1 = []
        for i in range(nfolds):
            ret1.append([[], []])
        cols = sorted(list(tt.columns.values))
        cols.remove('filename')
        print('Columns: {}'.format(cols))
        stat = []
        for c in cols:
            count = len(tt[tt[c] == 1])
            stat.append((c, count))
        stat_sorted = sorted(stat, key=lambda tup: tup[1])
        print(stat_sorted)
        for c, count in stat_sorted:
            print(c, count)
            sub = tt[tt[c] == 1].copy()
            files_to_split = np.array(list(sub['filename'].values))
            if len(files_to_split) > nfolds:
                kf = KFold(n_splits=nfolds, shuffle=True, random_state=66)
                total = 0
                for train_file, test_file in kf.split(range(len(files_to_split))):
                    train_files_fixed = list(set(files_to_split[train_file]) - set(ret1[total][1]))
                    ret1[total][0] += train_files_fixed
                    print('Initial: {}'.format(len(test_file)))
                    test_files_fixed = list(set(files_to_split[test_file]) - set(ret1[total][0]))
                    print('Reduced: {}'.format(len(test_files_fixed)))
                    ret1[total][1] += test_files_fixed
                    total += 1
            else:
                print('Here')
                part_0 = [files_to_split[0]]
                part_1 = [files_to_split[1]]
                part_2 = []
                part_3 = []
                part_4 = []
                ret1[0][0] += part_1 + part_2 + part_3 + part_4
                ret1[0][1] += part_0
                ret1[1][0] += part_0 + part_2 + part_3 + part_4
                ret1[1][1] += part_1
                ret1[2][0] += part_0 + part_1 + part_3 + part_4
                ret1[2][1] += part_2
                ret1[3][0] += part_0 + part_1 + part_2 + part_4
                ret1[3][1] += part_3
                ret1[4][0] += part_0 + part_1 + part_2 + part_3
                ret1[4][1] += part_4

        lbl = get_labels_dict()
        ret = []
        for i in range(nfolds):
            ret.append([[], []])

        fdict = dict()
        for i in range(len(files)):
            fdict[files[i]] = i

        for total in range(nfolds):
            train_files = ret1[total][0]
            test_files = ret1[total][1]
            for i in range(len(train_files)):
                ret[total][0].append(fdict[train_files[i]])
            for i in range(len(test_files)):
                ret[total][1].append(fdict[test_files[i]])
            print(len(ret[total][0]), len(ret[total][1]))

        for total in range(nfolds):
            ret[total][0] = list(set(ret[total][0]))
            ret[total][1] = list(set(ret[total][1]))
            print(len(ret[total][0]), len(ret[total][1]))

        for train_file, test_file in ret:
            print(len(train_file))
            print(len(test_file))
            clss = dict()
            for i in range(len(test_file)):
                a = files[test_file[i]]
                animal = ANIMAL_TYPE[np.argmax(lbl[a])]
                if animal in clss:
                    clss[animal] += 1
                else:
                    clss[animal] = 1
            print(len(clss), clss)

        save_in_file((files, ret), kfold_cache_path)
    else:
        files, ret = load_from_file(kfold_cache_path)

    return files, ret


def get_single_stratified_split():
    # Use 95% / 5% split
    cache_path = OUTPUT_PATH + 'single_stratified_split_cache.pklz'
    if not os.path.isfile(cache_path):
        tt = pd.read_csv(INPUT_PATH + 'train_labels.csv')
        print('Initital table length: {}'.format(len(tt)))
        train_files = []
        valid_files = []
        cols = sorted(list(tt.columns.values))
        cols.remove('filename')
        print('Columns: {}'.format(cols))
        stat = []
        for c in cols:
            count = len(tt[tt[c] == 1])
            stat.append((c, count))
        stat_sorted = sorted(stat, key=lambda tup: tup[1])
        print(stat_sorted)
        for c, count in stat_sorted:
            sub = tt[tt[c] == 1].copy()
            valid_part = len(sub) // 20
            if valid_part == 0:
                valid_part = 1
            files_to_split = list(sub['filename'].values)
            random.shuffle(files_to_split)
            valid_files += files_to_split[:valid_part]
            train_files += files_to_split[valid_part:]
            tt = tt[tt[c] != 1]
            print('Current length: {}'.format(len(tt)))

        print('Length of train: {}'.format(len(train_files)))
        print('Length of valid: {}'.format(len(valid_files)))
        save_in_file((train_files, valid_files), cache_path)
    else:
        train_files, valid_files = load_from_file(cache_path)

    return train_files, valid_files

# Get files from non train and non test lists
def get_other_filelist():
    files_all = glob.glob(FULL_VIDEO_PATH + '*.mp4')
    print('Full videos: {}'.format(len(files_all)))
    other_labels = []
    for f in files_all:
        other_labels.append(os.path.basename(f))
    test_labels = pd.read_csv(INPUT_PATH + 'submission_format.csv', index_col='filename').index
    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename').index
    files = sorted(list(set(other_labels) - set(test_labels) - set(train_labels)))
    print('Files to process: {}'.format(len(files)))
    return files


def get_score(pred, real, print_stat=True):
    scr = 0.0
    for i in range(pred.shape[1]):
        r = real[:, i].astype(np.float64).copy()
        p = pred[:, i].astype(np.float64).copy()
        if np.sum(r) == 0:
            r = np.concatenate((r, np.array([1.])))
            p = np.concatenate((p, np.array([1.])))
        l = log_loss(r, p)
        if print_stat is True:
            print('Partial score for {}: {}'.format(ANIMAL_TYPE[i], l))
        scr += l
    scr /= pred.shape[1]
    return scr


def read_audio(f):
    rate, wf = wavfile.read(f)
    wf = wf.astype(np.float32) / 32768.0
    wf = np.expand_dims(wf, axis=1)
    return wf


def save_history_figure(history, path, columns=('loss', 'val_loss')):
    import matplotlib.pyplot as plt
    s = pd.DataFrame(history.history)
    plt.plot(s[list(columns)])
    plt.savefig(path)
    plt.close()


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes