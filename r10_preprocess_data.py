# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Convert full videos to lower resolution and extract audio. 
Note: it's probably better to use full videos everywhere on the next steps.
'''

from a00_common_functions import *
import ffmpy


def get_nolabel_data():
    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename').index
    test_labels = pd.read_csv(INPUT_PATH + 'submission_format.csv', index_col='filename').index
    train_labels = set(train_labels.values)
    test_labels = set(test_labels.values)
    all_files = glob.glob(INPUT_PATH + 'micro/*.mp4')
    print('Len train: {}'.format(len(train_labels)))
    print('Len test: {}'.format(len(test_labels)))
    print('All files: {}'.format(len(all_files)))

    af = []
    for f in all_files:
        af.append(os.path.basename(f))

    af = set(af)
    af = af - train_labels - test_labels
    print('No labels length: {}'.format(len(af)))
    af = list(af)
    nolabels = pd.DataFrame(af, columns=["filename"])
    print(nolabels)
    nolabels.to_csv(OUTPUT_PATH + 'nolabels.csv', index=False)


def check_nolabel_data():
    subm = pd.read_csv(OUTPUT_PATH + 'nolabels.csv', index_col='filename')
    files = list(subm.index.values)
    failed_files = []
    for f in files:
        f1 = INPUT_PATH + 'micro/' + f
        full_video = read_video(f1)
        if full_video.shape != (30, 64, 64, 3):
            print('Problem with video: {}: {}'.format(f, full_video.shape))
            failed_files.append(f)

    print('Initial: {}'.format(len(subm)))
    subm = subm[~subm.index.isin(failed_files)].copy()
    print('Fixed: {}'.format(len(subm)))
    # subm = subm.sort_index(axis=1)
    subm.to_csv(OUTPUT_PATH + 'nolabels_fixed.csv')


def convert_single_video_v2(name):

    f = FULL_VIDEO_PATH + name
    out_video = OUTPUT_FULL_VIDEO_PATH + name
    if os.path.isfile(out_video):
        print('Video already created. Skip it!')
        return
    out_audio = OUTPUT_FULL_AUDIO_PATH + name + '.mp3'

    try:
        ff = ffmpy.FFmpeg(
            inputs = {f: ''},
            outputs = {out_audio: '-vn -acodec copy -y'}
        )
        print(ff.cmd)
        ff.run()
    except:
        print('No audio')
        os.remove(out_audio)

    ff = ffmpy.FFmpeg(
        inputs={f: '-r 24'},
        outputs={out_video: '-r 4 -filter:v scale=250:250,select="not(mod(n-1\,6))" -sws_flags spline -vsync 0 -pix_fmt yuv420p -vcodec libx264 -level 41 -refs 4 -preset veryslow -trellis 2 -me_range 24 -x264opts crf=16:ratetol=100.0:psy=0 -threads 6 -an -f mp4 -y'}
    )
    print(ff.cmd)
    ff.run()


def convert_single_video_v2_125_60(name):

    f = FULL_VIDEO_PATH + name
    out_video = OUTPUT_FULL_VIDEO_PATH_125_60 + name
    if os.path.isfile(out_video):
        print('Video already created. Skip it!')
        return
    ff = ffmpy.FFmpeg(
        inputs={f: '-r 24'},
        outputs={out_video: '-r 4 -filter:v scale=126:126,select="not(mod(n-1\,6))" -sws_flags spline -vsync 0 -pix_fmt yuv420p -vcodec libx264 -level 41 -refs 4 -preset veryslow -trellis 2 -me_range 24 -x264opts crf=16:ratetol=100.0:psy=0 -threads 6 -an -f mp4 -y'}
    )
    # print(ff.cmd)
    ff.run(stdout=open('nul', 'w'), stderr=open('nul', 'w'))


def convert_all_train_videos():
    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename').index
    for t in train_labels[::-1]:
        f = FULL_VIDEO_PATH + t
        if os.path.isfile(f):
            print('Go for {}...'.format(t))
            convert_single_video_v2(t)


def convert_all_test_videos():
    subm = pd.read_csv(INPUT_PATH + 'submission_format.csv', index_col='filename')
    files = list(subm.index.values)
    for t in files:
        f = FULL_VIDEO_PATH + t
        if os.path.isfile(f):
            print('Go for {}...'.format(t))
            convert_single_video_v2(t)


def convert_all_other_videos():
    files = get_other_filelist()
    for t in files:
        f = FULL_VIDEO_PATH + t
        if os.path.isfile(f):
            print('Go for {}...'.format(t))
            try:
                convert_single_video_v2(t)
            except:
                print('Cant convert video {}'.format(t))


def convert_all_train_videos_125_60(reverse=False):
    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename').index
    if reverse is True:
        train_labels = train_labels[::-1]
    for t in train_labels:
        f = FULL_VIDEO_PATH + t
        if os.path.isfile(f):
            print('Go for {}...'.format(t))
            convert_single_video_v2_125_60(t)


def get_nano_dataset_cache():
    cache_folder = INPUT_PATH + 'nano_cache/'
    if not os.path.isdir(cache_folder):
        os.mkdir(cache_folder)
    cache_flag_path = cache_folder + 'train.txt'
    store_size = 10000
    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename').index
    if not os.path.isfile(cache_flag_path):
        for i in range(0, len(train_labels), store_size):
            print('Go for {}'.format(i))
            big_array = dict()
            if i+store_size > len(train_labels):
                files = train_labels[i:]
            else:
                files = train_labels[i:i + store_size]
            for t in files:
                f = INPUT_PATH + 'nano/' + t
                big_array[t] = read_video(f)
            save_in_file(big_array, cache_folder + 'train_{}.pklz'.format(i))
        f = open(cache_flag_path, 'w')
        f.close()

    big = dict()
    for i in range(0, len(train_labels), store_size):
        print('Load {}'.format(i))
        d = load_from_file(cache_folder + 'train_{}.pklz'.format(i))
        big = dict(big, **d)
    return big


def get_micro_dataset_cache():
    cache_folder = INPUT_PATH + 'micro_cache/'
    if not os.path.isdir(cache_folder):
        os.mkdir(cache_folder)
    cache_flag_path = cache_folder + 'train.txt'
    store_size = 10000
    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename').index
    if not os.path.isfile(cache_flag_path):
        for i in range(0, len(train_labels), store_size):
            print('Go for {}'.format(i))
            big_array = dict()
            if i+store_size > len(train_labels):
                files = train_labels[i:]
            else:
                files = train_labels[i:i + store_size]
            for t in files:
                f = INPUT_PATH + 'micro/' + t
                big_array[t] = read_video(f)
            save_in_file(big_array, cache_folder + 'train_{}.pklz'.format(i))
        f = open(cache_flag_path, 'w')
        f.close()

    big = dict()
    for i in range(0, len(train_labels), store_size):
        print('Load {}'.format(i))
        d = load_from_file(cache_folder + 'train_{}.pklz'.format(i))
        big = dict(big, **d)
    return big


def convert_audio_mp3_to_wav():
    files = glob.glob(OUTPUT_FULL_AUDIO_PATH + '*.mp3')
    for f in files:
        print('Go for {}'.format(os.path.basename(f)))
        in_path = f
        out_path = OUTPUT_FULL_AUDIO_WAV_PATH + os.path.basename(f)[:-4] + '.wav'

        if os.path.isfile(out_path):
            print('WAV already created. Skip it!')
            continue
        ff = ffmpy.FFmpeg(
            inputs={in_path: ''},
            outputs={out_path: '-vn -acodec pcm_s16le -ac 1 -ar 22050 -f wav -y'}
        )
        # print(ff.cmd)
        try:
            ff.run(stdout=open('nul', 'w'), stderr=open('nul', 'w'))
        except:
            try:
                os.remove(out_path)
            except:
                print('Cant remove...')


if __name__ == '__main__':
    get_nolabel_data()
    check_nolabel_data()
    convert_all_train_videos()
    # convert_all_train_videos_125_60(True)
    convert_all_test_videos()
    convert_all_other_videos()
    # get_nano_dataset_cache()
    # get_micro_dataset_cache()
    convert_audio_mp3_to_wav()