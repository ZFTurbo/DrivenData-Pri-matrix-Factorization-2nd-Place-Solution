# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Create some videos with vectors visualization
'''

from a00_common_functions import *
import shutil
from PIL import Image
import math
import colorsys


def find_animal_videos_examples():
    np.random.seed(999)
    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename')
    out_directory = OUTPUT_PATH + 'example_videos_by_class/'
    if not os.path.isdir(out_directory):
        os.mkdir(out_directory)
    for a in ANIMAL_TYPE:
        if not os.path.isdir(out_directory + a):
            os.mkdir(out_directory + a)

    for a in ANIMAL_TYPE:
        print('Go for {}'.format(a))
        fls = train_labels[train_labels[a] == 1].index.values
        np.random.shuffle(fls)
        for f in fls[:10]:
            print('Copy {}'.format(f))
            in_path = FULL_VIDEO_PATH + f
            out_path = out_directory + a + '/' + f
            shutil.copy(in_path, out_path)


def convert_image_hash(hashes):
    hashes = np.array(hashes)
    s = []
    for i in range(hashes.shape[0]):
        p = list('{:064b}{:064b}'.format(int(str(hashes[i][0]), 16), int(str(hashes[i][1]), 16)))
        s.append(p)
    s = np.array(s, dtype=np.uint8)
    return s


def create_imagehash_videos(vid_motion, vid_blank):
    text_color = (32, 32, 192)
    from_top = 130

    blank = read_video(FULL_VIDEO_PATH + vid_blank)
    motion = read_video(FULL_VIDEO_PATH + vid_motion)

    im_hash_blank = load_from_file(OUTPUT_PATH + 'image_hashes/' + vid_blank + '.pklz')
    im_hash_motion = load_from_file(OUTPUT_PATH + 'image_hashes/' + vid_motion + '.pklz')
    im_hash_blank = convert_image_hash(im_hash_blank)
    im_hash_motion = convert_image_hash(im_hash_motion)
    print(im_hash_blank.shape)
    print(im_hash_motion.shape)

    print(blank.shape)
    # show_image(blank[0])
    # show_image(motion[0])

    full_frame = np.zeros((blank.shape[0], 720, 1280, 3), dtype=np.uint8)
    full_frame[...] = 255
    max_diff_blank = 0
    max_diff_motion = 0
    for i in range(blank.shape[0]):
        bl_resized = cv2.resize(blank[i], (540, 300), cv2.INTER_LANCZOS4)
        mo_resized = cv2.resize(motion[i], (540, 300), cv2.INTER_LANCZOS4)
        full_frame[i, from_top:from_top+300, 60:60+540, :] = bl_resized
        full_frame[i, from_top:from_top+300, -60-540:-60, :] = mo_resized

        # Text caption
        cv2.putText(full_frame[i], 'Motion detection with pHash and dHash', (250, 30),  cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(full_frame[i], 'Blank video: {}'.format(vid_blank), (70, from_top - 30), cv2.FONT_HERSHEY_TRIPLEX, 1, text_color, 1, cv2.LINE_AA)
        cv2.putText(full_frame[i], 'Animal video: {}'.format(vid_blank), (70 + 540 + 70, from_top - 30), cv2.FONT_HERSHEY_TRIPLEX, 1, text_color, 1, cv2.LINE_AA)

        # Vectors
        cv2.rectangle(full_frame[i], (72, from_top + 300 + 100 - 5), (75 + 128 * 4, from_top + 300 + 100 + 50 + 5), (0, 150, 0), 1)
        cv2.rectangle(full_frame[i], (72, from_top + 300 + 210 - 5), (75 + 128 * 4, from_top + 300 + 210 + 50 + 5), (0, 150, 0), 1)
        prev_diff = 0
        for j in range(im_hash_blank.shape[1]):
            if im_hash_blank[i, j] > 0:
                cv2.rectangle(full_frame[i], (75 + 4*j, from_top + 300 + 100), (75 + 4*(j + 1)-3, from_top + 300 + 100 + 50), (150, 0, 0), 1)
            if i > 0 and im_hash_blank[i, j] != im_hash_blank[i-1, j]:
                cv2.rectangle(full_frame[i], (75 + 4*j, from_top + 300 + 210), (75 + 4*(j + 1)-3, from_top + 300 + 210 + 50), (150, 0, 0), 1)
                prev_diff += 1
        cv2.putText(full_frame[i], 'Current image hash', (75, from_top + 300 + 80), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    text_color, 1, cv2.LINE_AA)
        cv2.putText(full_frame[i], 'Prev frame diff: {} Max: {}'.format(prev_diff, max_diff_blank), (75, from_top + 300 + 190), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    text_color, 1, cv2.LINE_AA)
        if prev_diff > max_diff_blank:
            max_diff_blank = prev_diff

        overall_shift = 620
        cv2.rectangle(full_frame[i], (72 + overall_shift, from_top + 300 + 100 - 5), (75 + 128 * 4 + overall_shift, from_top + 300 + 100 + 50 + 5),
                      (0, 150, 0), 1)
        cv2.rectangle(full_frame[i], (72 + overall_shift, from_top + 300 + 210 - 5),
                      (75 + 128 * 4 + overall_shift, from_top + 300 + 210 + 50 + 5), (0, 150, 0), 1)
        prev_diff = 0
        for j in range(im_hash_motion.shape[1]):
            if im_hash_motion[i, j] > 0:
                cv2.rectangle(full_frame[i], (75 + overall_shift + 4*j, from_top + 300 + 100), (75 + overall_shift + 4*(j + 1)-3, from_top + 300 + 100 + 50), (150, 0, 0), 1)
            if i > 0 and im_hash_motion[i, j] != im_hash_motion[i-1, j]:
                cv2.rectangle(full_frame[i], (75 + overall_shift + 4*j, from_top + 300 + 210), (75 + overall_shift + 4*(j + 1)-3, from_top + 300 + 210 + 50), (150, 0, 0), 1)
                prev_diff += 1
        cv2.putText(full_frame[i], 'Current image hash', (75 + overall_shift, from_top + 300 + 80), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    text_color, 1, cv2.LINE_AA)
        cv2.putText(full_frame[i], 'Prev frame diff: {} Max: {}'.format(prev_diff, max_diff_motion), (75 + overall_shift, from_top + 300 + 190), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    text_color, 1, cv2.LINE_AA)
        if prev_diff > max_diff_motion:
            max_diff_motion = prev_diff

        # show_image(full_frame[i])

    create_video(full_frame, 'debug_{}_{}.avi'.format(vid_motion, vid_blank))


def preproc_video_for_vgg16(x):
    x = x.astype(np.float32)
    x[:, 0, :, :] -= 103.939
    x[:, 1, :, :] -= 116.779
    x[:, 2, :, :] -= 123.68
    return x


def read_video_VGG16(f, sz=448):
    cap = cv2.VideoCapture(f)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    video = None
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break
        if video is None:
            video = np.zeros((length, sz, sz, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (sz, sz), cv2.INTER_LANCZOS4)
        video[current_frame, :, :, :] = frame
        current_frame += 1

    video = np.transpose(video, (0, 3, 1, 2))
    return video


def get_vgg16_predictions_for_video_full(t):
    OUTPUT_VGG16_FOLDER = OUTPUT_PATH + 'debug_vgg16/'
    if not os.path.isdir(OUTPUT_VGG16_FOLDER):
        os.mkdir(OUTPUT_VGG16_FOLDER)
    out_file = OUTPUT_VGG16_FOLDER + t + '.pklz'
    if not os.path.isfile(out_file):
        from keras.applications.vgg16 import VGG16
        import keras.backend as K
        K.set_image_dim_ordering('th')

        model = VGG16(include_top=False, input_shape=(3, 448, 448), weights='imagenet', pooling='avg')

        f = FULL_VIDEO_PATH + t
        print('Go for {}...'.format(t))
        out_file = OUTPUT_VGG16_FOLDER + t + '.pklz'
        v = read_video_VGG16(f)
        print(v.shape)
        v = preproc_video_for_vgg16(v.astype(np.float32))
        preds = model.predict(v)
        save_in_file(preds, out_file)
    else:
        preds = load_from_file(out_file)
    return preds


def hsv2rgb(h, s, v):
    return tuple(int((i * 255)) % 256 for i in colorsys.hsv_to_rgb(h, s, v))


def gen_hsv_palette(num):
    pal = []
    for i in range(360):
        pal.append(hsv2rgb(i/360, 1, 1))
        print(pal[-1])
    for j in range(360, num):
        pal.append(hsv2rgb((j-360)/360, 1, 1))
        print(pal[-1])
    return pal


def normalize_vectors(vec1, vec2):
    for i in range(vec1.shape[1]):
        mx = max(vec1[:, i].max(), vec2[:, i].max())
        if mx > 0:
            vec1[:, i] /= mx
            vec2[:, i] /= mx
    return vec1, vec2


def normalize_vectors_v2(vec1, vec2):
    mx = max(vec1.max(), vec2.max())
    vec1 /= vec1.max()
    vec2 /= vec2.max()
    return vec1, vec2


def create_video_VGG16_v1(vid_motion, vid_blank):
    text_color = (32, 32, 192)
    from_top = 130
    palette = gen_hsv_palette(512)

    blank = read_video(FULL_VIDEO_PATH + vid_blank)
    motion = read_video(FULL_VIDEO_PATH + vid_motion)

    im_hash_blank = get_vgg16_predictions_for_video_full(vid_blank)
    im_hash_motion = get_vgg16_predictions_for_video_full(vid_motion)
    print(im_hash_blank.shape, im_hash_blank.min(), im_hash_blank.max())
    print(im_hash_motion.shape, im_hash_motion.min(), im_hash_motion.max())
    im_hash_blank, im_hash_motion = normalize_vectors(im_hash_blank, im_hash_motion)

    full_frame = np.zeros((blank.shape[0], 720, 1280, 3), dtype=np.uint8)
    full_frame[...] = 255
    max_diff_blank = 0
    max_diff_motion = 0
    for i in range(blank.shape[0]):
        bl_resized = cv2.resize(blank[i], (540, 300), cv2.INTER_LANCZOS4)
        mo_resized = cv2.resize(motion[i], (540, 300), cv2.INTER_LANCZOS4)
        full_frame[i, from_top:from_top+300, 60:60+540, :] = bl_resized
        full_frame[i, from_top:from_top+300, -60-540:-60, :] = mo_resized

        # Text caption
        cv2.putText(full_frame[i], 'Image vectors from pretrained neural net VGG16', (200, 30),  cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(full_frame[i], 'Blank video: {}'.format(vid_blank), (70, from_top - 30), cv2.FONT_HERSHEY_TRIPLEX, 1, text_color, 1, cv2.LINE_AA)
        cv2.putText(full_frame[i], 'Animal video: {}'.format(vid_blank), (70 + 540 + 70, from_top - 30), cv2.FONT_HERSHEY_TRIPLEX, 1, text_color, 1, cv2.LINE_AA)

        # Vectors
        cv2.rectangle(full_frame[i], (72, from_top + 300 + 100 - 5), (75 + 128 * 4 + 5, from_top + 300 + 100 + 50 + 5), (0, 150, 0), 1)
        cv2.rectangle(full_frame[i], (72, from_top + 300 + 210 - 5), (75 + 128 * 4 + 5, from_top + 300 + 210 + 50 + 5), (0, 150, 0), 1)
        prev_diff = 0
        for j in range(im_hash_blank.shape[1]):
            if im_hash_blank[i, j] > 0:
                length = int(im_hash_blank[i, j]*50)
                cv2.rectangle(full_frame[i], (75 + j, from_top + 300 + 150 - length), (75 + (j + 1), from_top + 300 + 150), palette[j], 1)
            if i > 0 and im_hash_blank[i, j] != im_hash_blank[i-1, j]:
                diff_value = np.abs(im_hash_blank[i, j] - im_hash_blank[i-1, j])
                length = int(diff_value * 50)
                cv2.rectangle(full_frame[i], (75 + j, from_top + 300 + 260 - length), (75 + (j + 1), from_top + 300 + 260), palette[j], 1)
                prev_diff += diff_value
        cv2.putText(full_frame[i], 'Current image hash', (75, from_top + 300 + 80), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    text_color, 1, cv2.LINE_AA)
        cv2.putText(full_frame[i], 'Prev frame diff: {:.2f}'.format(prev_diff), (75, from_top + 300 + 190), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    text_color, 1, cv2.LINE_AA)
        if prev_diff > max_diff_blank:
            max_diff_blank = prev_diff

        overall_shift = 620
        cv2.rectangle(full_frame[i], (72 + overall_shift, from_top + 300 + 100 - 5), (75 + 5 + 128 * 4 + overall_shift, from_top + 300 + 100 + 50 + 5),
                      (0, 150, 0), 1)
        cv2.rectangle(full_frame[i], (72 + overall_shift, from_top + 300 + 210 - 5),
                      (75 + 5 + 128 * 4 + overall_shift, from_top + 300 + 210 + 50 + 5), (0, 150, 0), 1)
        prev_diff = 0
        for j in range(im_hash_motion.shape[1]):
            if im_hash_motion[i, j] > 0:
                length = int(im_hash_motion[i, j] * 50)
                cv2.rectangle(full_frame[i], (75 + overall_shift + j, from_top + 300 + 150 - length), (75 + overall_shift + (j + 1), from_top + 300 + 150), palette[j], 1)
            if i > 0:
                diff_value = np.abs(im_hash_motion[i, j] - im_hash_motion[i - 1, j])
                length = int(diff_value * 50)
                cv2.rectangle(full_frame[i], (75 + overall_shift + j, from_top + 300 + 260 - length), (75 + overall_shift + (j + 1), from_top + 300 + 260), palette[j], 1)
                prev_diff += diff_value
        cv2.putText(full_frame[i], 'Current image hash', (75 + overall_shift, from_top + 300 + 80), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    text_color, 1, cv2.LINE_AA)
        cv2.putText(full_frame[i], 'Prev frame diff: {:.2f}'.format(prev_diff, max_diff_motion), (75 + overall_shift, from_top + 300 + 190), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    text_color, 1, cv2.LINE_AA)
        if prev_diff > max_diff_motion:
            max_diff_motion = prev_diff

        # show_image(full_frame[i])

    create_video(full_frame, 'VGG16_v1_{}_{}.avi'.format(vid_motion, vid_blank))


def create_video_VGG16_v2(vid_motion, vid_blank):
    text_color = (32, 32, 192)
    from_top = 130
    palette = gen_hsv_palette(512)

    blank = read_video(FULL_VIDEO_PATH + vid_blank)
    motion = read_video(FULL_VIDEO_PATH + vid_motion)

    im_hash_blank = get_vgg16_predictions_for_video_full(vid_blank)
    im_hash_motion = get_vgg16_predictions_for_video_full(vid_motion)
    print(im_hash_blank.shape, im_hash_blank.min(), im_hash_blank.max())
    print(im_hash_motion.shape, im_hash_motion.min(), im_hash_motion.max())
    im_hash_blank, im_hash_motion = normalize_vectors_v2(im_hash_blank, im_hash_motion)

    full_frame = np.zeros((blank.shape[0], 720, 1280, 3), dtype=np.uint8)
    full_frame[...] = 255
    max_diff_blank = 0
    max_diff_motion = 0
    for i in range(blank.shape[0]):
        bl_resized = cv2.resize(blank[i], (540, 300), cv2.INTER_LANCZOS4)
        mo_resized = cv2.resize(motion[i], (540, 300), cv2.INTER_LANCZOS4)
        full_frame[i, from_top:from_top+300, 60:60+540, :] = bl_resized
        full_frame[i, from_top:from_top+300, -60-540:-60, :] = mo_resized

        # Text caption
        cv2.putText(full_frame[i], 'Image vectors from pretrained neural net VGG16', (200, 30),  cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(full_frame[i], 'Wild dog', (250, from_top - 30), cv2.FONT_HERSHEY_TRIPLEX, 1, text_color, 1, cv2.LINE_AA)
        cv2.putText(full_frame[i], 'Chimpanzee', (250 + 540 + 70, from_top - 30), cv2.FONT_HERSHEY_TRIPLEX, 1, text_color, 1, cv2.LINE_AA)

        # Vectors
        cv2.rectangle(full_frame[i], (72, from_top + 300 + 100 - 5), (75 + 128 * 4 + 5, from_top + 300 + 100 + 50 + 5), (0, 150, 0), 1)
        cv2.rectangle(full_frame[i], (72, from_top + 300 + 210 - 5), (75 + 128 * 4 + 5, from_top + 300 + 210 + 50 + 5), (0, 150, 0), 1)
        prev_diff = 0
        for j in range(im_hash_blank.shape[1]):
            if im_hash_blank[i, j] > 0:
                length = int(im_hash_blank[i, j]*50)
                cv2.rectangle(full_frame[i], (75 + j, from_top + 300 + 150 - length), (75 + (j + 1), from_top + 300 + 150), palette[j], 1)
            if i > 0:
                diff_value = np.abs(im_hash_blank[i, j] - im_hash_blank[i-1, j])
                try:
                    length = int(diff_value * 50)
                except:
                    print(i, j)
                    print(im_hash_blank[i, j])
                    print(im_hash_blank[i-1, j])
                    exit()
                cv2.rectangle(full_frame[i], (75 + j, from_top + 300 + 260 - length), (75 + (j + 1), from_top + 300 + 260), palette[j], 1)
                prev_diff += diff_value
        cv2.putText(full_frame[i], 'Current image hash', (75, from_top + 300 + 80), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    text_color, 1, cv2.LINE_AA)
        cv2.putText(full_frame[i], 'Prev frame diff: {:.2f}'.format(prev_diff), (75, from_top + 300 + 190), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    text_color, 1, cv2.LINE_AA)
        if prev_diff > max_diff_blank:
            max_diff_blank = prev_diff

        overall_shift = 620
        cv2.rectangle(full_frame[i], (72 + overall_shift, from_top + 300 + 100 - 5), (75 + 5 + 128 * 4 + overall_shift, from_top + 300 + 100 + 50 + 5),
                      (0, 150, 0), 1)
        cv2.rectangle(full_frame[i], (72 + overall_shift, from_top + 300 + 210 - 5),
                      (75 + 5 + 128 * 4 + overall_shift, from_top + 300 + 210 + 50 + 5), (0, 150, 0), 1)
        prev_diff = 0
        for j in range(im_hash_motion.shape[1]):
            if im_hash_motion[i, j] > 0:
                length = int(im_hash_motion[i, j] * 50)
                cv2.rectangle(full_frame[i], (75 + overall_shift + j, from_top + 300 + 150 - length), (75 + overall_shift + (j + 1), from_top + 300 + 150), palette[j], 1)
            if i > 0:
                diff_value = np.abs(im_hash_motion[i, j] - im_hash_motion[i - 1, j])
                length = int(diff_value * 50)
                cv2.rectangle(full_frame[i], (75 + overall_shift + j, from_top + 300 + 260 - length), (75 + overall_shift + (j + 1), from_top + 300 + 260), palette[j], 1)
                prev_diff += diff_value
        cv2.putText(full_frame[i], 'Current image hash', (75 + overall_shift, from_top + 300 + 80), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    text_color, 1, cv2.LINE_AA)
        cv2.putText(full_frame[i], 'Prev frame diff: {:.2f}'.format(prev_diff, max_diff_motion), (75 + overall_shift, from_top + 300 + 190), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    text_color, 1, cv2.LINE_AA)
        if prev_diff > max_diff_motion:
            max_diff_motion = prev_diff

        # show_image(full_frame[i])

    create_video(full_frame, 'VGG16_v2_{}_{}.avi'.format(vid_motion, vid_blank))


def create_video(image_list, out_file):
    height, width = image_list[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'DIB ')
    # fourcc = -1
    fps = 24.0
    video = cv2.VideoWriter(out_file, fourcc, fps, (width, height), True)
    for im in image_list:
        video.write(im.astype(np.uint8))
    cv2.destroyAllWindows()
    video.release()


def create_video_VGG16_v3(vid_motion, caption):
    text_color = (32, 32, 192)
    from_top = 130
    palette = gen_hsv_palette(512)

    motion = read_video(FULL_VIDEO_PATH + vid_motion)

    im_hash_motion = get_vgg16_predictions_for_video_full(vid_motion)
    print(im_hash_motion.shape, im_hash_motion.min(), im_hash_motion.max())
    im_hash_motion /= im_hash_motion.max()

    full_frame = np.zeros((motion.shape[0], 720, 1280, 3), dtype=np.uint8)
    full_frame[...] = 255
    max_diff_blank = 0
    max_diff_motion = 0
    prev_diff_avg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(motion.shape[0]):
        mo_resized = cv2.resize(motion[i], (720, 404), cv2.INTER_LANCZOS4)
        full_frame[i, from_top:from_top+404, -280-720:-280, :] = mo_resized

        # Text caption
        cv2.putText(full_frame[i], 'Image vectors from pretrained neural net VGG16', (200, 30),  cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(full_frame[i], caption, (530, from_top - 30), cv2.FONT_HERSHEY_TRIPLEX, 1, text_color, 1, cv2.LINE_AA)

        # Vectors
        overall_shift = 10
        overall_shift2 = 590
        cv2.rectangle(full_frame[i], (72 + overall_shift, from_top + 300 + 210 - 5), (75 + 5 + 128 * 4 + overall_shift, from_top + 300 + 210 + 50 + 5),
                      (0, 150, 0), 1)
        cv2.rectangle(full_frame[i], (72 + overall_shift2, from_top + 300 + 210 - 5),
                      (75 + 5 + 128 * 4 + overall_shift2, from_top + 300 + 210 + 50 + 5), (0, 150, 0), 1)
        prev_diff = 0
        for j in range(im_hash_motion.shape[1]):
            if im_hash_motion[i, j] > 0:
                length = int(im_hash_motion[i, j] * 50)
                cv2.rectangle(full_frame[i], (75 + overall_shift + j, from_top + 300 + 260 - length), (75 + overall_shift + (j + 1), from_top + 300 + 260), palette[j], 1)
            if i > 0:
                diff_value = np.abs(im_hash_motion[i, j] - im_hash_motion[i - 1, j])
                length = int(diff_value * 50 * 3)
                cv2.rectangle(full_frame[i], (75 + overall_shift2 + j, from_top + 300 + 260 - length), (75 + overall_shift2 + (j + 1), from_top + 300 + 260), palette[j], 1)
                prev_diff += diff_value
        cv2.putText(full_frame[i], 'Current image hash', (75 + overall_shift, from_top + 300 + 190), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    text_color, 1, cv2.LINE_AA)
        cv2.putText(full_frame[i], 'Avg frame diff: {:.0f}'.format(np.array(prev_diff_avg[-10:]).mean()), (75 + overall_shift2, from_top + 300 + 190), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    text_color, 1, cv2.LINE_AA)
        prev_diff_avg.append(prev_diff)
        if prev_diff > max_diff_motion:
            max_diff_motion = prev_diff

        # show_image(full_frame[i])

    create_video(full_frame, 'VGG16_v3_{}.avi'.format(vid_motion))


if __name__ == '__main__':
    find_animal_videos_examples()
    create_imagehash_videos('58LlzehObv.mp4', 'PfQKErLHJP.mp4')
    create_imagehash_videos('96D0fUjVB7.mp4', 'J2qknRE3cM.mp4')
    create_imagehash_videos('9Wmo89I0yL.mp4', 'S1nlaqL7XE.mp4')
    create_video_VGG16_v1('96D0fUjVB7.mp4', 'J2qknRE3cM.mp4')
    create_video_VGG16_v2('bvxaarMrpE.mp4', 'ApQq1avfua.mp4')
    create_video_VGG16_v3('1sdaVSUOWy.mp4', 'Hippopotamus')
    create_video_VGG16_v3('FgJpFLxSmH.mp4', '  Gorilla  ')
    create_video_VGG16_v3('bo5h4plgc5.mp4', '      Hog    ')
    create_video_VGG16_v3('JMM6P7AKcm.mp4', 'Primates (other)')
