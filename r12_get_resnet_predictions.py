# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Extract features for every 6th frame of video with ResNet50 pretrained net.
It will generate around 95 GB of additional files.
Note: it's probably better to replace 224x224 on some higher resolution (448x448) and use full videos.
Note: Optimized to use with Theano backend
'''

from a00_common_functions import *

OUTPUT_RESNET50_FOLDER = OUTPUT_PATH + 'resnet50_preds/'
if not os.path.isdir(OUTPUT_RESNET50_FOLDER):
    os.mkdir(OUTPUT_RESNET50_FOLDER)


def read_video_224_224(f):
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
            video = np.zeros((1 + ((length - 1) // frame_mod), 224, 224, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (224, 224), cv2.INTER_LANCZOS4)
        video[current_frame // frame_mod, :, :, :] = frame
        current_frame += 1

    video = np.transpose(video, (0, 3, 1, 2))
    return video


def preproc_video_for_resnet50(x):
    x = x.astype(np.float32)
    x[:, 0, :, :] -= 103.939
    x[:, 1, :, :] -= 116.779
    x[:, 2, :, :] -= 123.68
    return x


def get_resnet_predictions_for_train(reverse=False):
    from keras.applications import ResNet50
    from keras.layers import Flatten
    from keras.models import Model
    import keras.backend as K
    K.set_image_dim_ordering('th')

    model = ResNet50(include_top=False, input_shape=(3, 224, 224), weights='imagenet')
    x = model.layers[-1].output
    x = Flatten()(x)
    model = Model(inputs=model.input, outputs=x)

    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename').index
    if reverse is True:
        train_labels = train_labels[::-1]
    # train_labels = ['JUb9w8FC2j.mp4']
    for t in train_labels:
        f = FULL_VIDEO_PATH + t
        if os.path.isfile(f):
            print('Go for {}...'.format(t))
            out_file = OUTPUT_RESNET50_FOLDER + t + '.pklz'
            if os.path.isfile(out_file):
                print('Already exists. Skip!')
                continue
            v = read_video_224_224(f)
            print(v.shape)
            v = preproc_video_for_resnet50(v)
            preds = model.predict(v)
            save_in_file(preds, out_file)


def get_resnet_predictions_for_test(reverse=False):
    from keras.applications import ResNet50
    from keras.layers import Flatten
    from keras.models import Model
    import keras.backend as K
    K.set_image_dim_ordering('th')

    model = ResNet50(include_top=False, input_shape=(3, 224, 224), weights='imagenet')
    x = model.layers[-1].output
    x = Flatten()(x)
    model = Model(inputs=model.input, outputs=x)

    subm = pd.read_csv(INPUT_PATH + 'submission_format.csv', index_col='filename')
    test_files = list(subm.index.values)
    if reverse is True:
        test_files = test_files[::-1]
    for t in test_files:
        f = FULL_VIDEO_PATH + t
        if os.path.isfile(f):
            print('Go for {}...'.format(t))
            out_file = OUTPUT_RESNET50_FOLDER + t + '.pklz'
            if os.path.isfile(out_file):
                print('Already exists. Skip!')
                continue
            v = read_video_224_224(f)
            print(v.shape)
            v = preproc_video_for_resnet50(v)
            preds = model.predict(v)
            save_in_file(preds, out_file)


if __name__ == '__main__':
    get_resnet_predictions_for_train(False)
    get_resnet_predictions_for_test(False)