# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Extract features for every 3rd frame of video with VGG16 pretrained net.
It will generate around 80 GB of additional files.
Note: Optimized to use with Theano backend
'''


from a00_common_functions import *


OUTPUT_VGG16_FOLDER = OUTPUT_PATH + 'vgg16_preds/'
if not os.path.isdir(OUTPUT_VGG16_FOLDER):
    os.mkdir(OUTPUT_VGG16_FOLDER)


def preproc_video_for_vgg16(x):
    x = x.astype(np.float32)
    x[:, 0, :, :] -= 103.939
    x[:, 1, :, :] -= 116.779
    x[:, 2, :, :] -= 123.68
    return x


def read_video_incep(f, sz=448):
    frame_mod = 3
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

    video = np.transpose(video, (0, 3, 1, 2))
    return video


def get_vgg16_predictions_for_train(reverse=False):
    from keras.applications.vgg16 import VGG16
    import keras.backend as K
    K.set_image_dim_ordering('th')

    model = VGG16(include_top=False, input_shape=(3, 448, 448), weights='imagenet', pooling='avg')

    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename').index
    if reverse is True:
        train_labels = train_labels[::-1]
    # train_labels = ['FkJeaRnQg1.mp4']
    for t in train_labels:
        f = FULL_VIDEO_PATH + t
        if os.path.isfile(f):
            print('Go for {}...'.format(t))
            out_file = OUTPUT_VGG16_FOLDER + t + '.pklz'
            if os.path.isfile(out_file):
                print('Already exists. Skip!')
                continue
            v = read_video_incep(f, 448)
            print(v.shape)
            v = preproc_video_for_vgg16(v.astype(np.float32))
            preds = model.predict(v)
            save_in_file(preds, out_file)


def get_vgg16_predictions_for_test(reverse=False):
    from keras.applications.vgg16 import VGG16
    import keras.backend as K
    K.set_image_dim_ordering('th')

    model = VGG16(include_top=False, input_shape=(3, 448, 448), weights='imagenet', pooling='avg')

    subm = pd.read_csv(INPUT_PATH + 'submission_format.csv', index_col='filename')
    test_files = list(subm.index.values)
    if reverse is True:
        test_files = test_files[::-1]
    # train_labels = ['FkJeaRnQg1.mp4']
    for t in test_files:
        f = FULL_VIDEO_PATH + t
        if os.path.isfile(f):
            print('Go for {}...'.format(t))
            out_file = OUTPUT_VGG16_FOLDER + t + '.pklz'
            if os.path.isfile(out_file):
                print('Already exists. Skip!')
                continue
            v = read_video_incep(f, 448)
            print(v.shape)
            v = preproc_video_for_vgg16(v.astype(np.float32))
            preds = model.predict(v)
            save_in_file(preds, out_file)


def get_vgg16_predictions_for_other(reverse=False, part=-1):
    from keras.applications.vgg16 import VGG16
    import keras.backend as K
    K.set_image_dim_ordering('th')

    model = VGG16(include_top=False, input_shape=(3, 448, 448), weights='imagenet', pooling='avg')

    files = get_other_filelist()
    if reverse is True:
        files = files[::-1]
    files_not_proc = []
    for t in files:
        out_file = OUTPUT_VGG16_FOLDER + t + '.pklz'
        if os.path.isfile(out_file):
            continue
        files_not_proc.append(t)
    print('Files left to process: {}'.format(len(files_not_proc)))
    if part != -1:
        if part != 3:
            files_not_proc = files_not_proc[(part*len(files_not_proc)) // 4:((part+1)*len(files_not_proc)) // 4]
        else:
            files_not_proc = files_not_proc[(part * len(files_not_proc)) // 4:]

    for t in files_not_proc:
        f = FULL_VIDEO_PATH + t
        if os.path.isfile(f):
            print('Go for {}...'.format(t))
            out_file = OUTPUT_VGG16_FOLDER + t + '.pklz'
            if os.path.isfile(out_file):
                print('Already exists. Skip!')
                continue
            try:
                v = read_video_incep(f, 448)
                print(v.shape)
            except:
                print('Bad video...')
                continue
            v = preproc_video_for_vgg16(v.astype(np.float32))
            preds = model.predict(v)
            save_in_file(preds, out_file)


if __name__ == '__main__':
    get_vgg16_predictions_for_train(False)
    get_vgg16_predictions_for_test(False)
    get_vgg16_predictions_for_other(False, -1)
