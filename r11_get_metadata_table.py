# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Get CSV files with metadata from all files
'''

from a00_common_functions import *


def get_metadata_table():
    '''
    Filesize
    AudioSize (later)
    FrameWidth
    FrameHeight
    Modification Date
    '''

    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename').index
    meta_arr = []
    for t in train_labels:
        print('Go for {}'.format(t))
        f = FULL_VIDEO_PATH + t
        file_size = os.path.getsize(f)
        mtime = os.path.getmtime(f)

        cap = cv2.VideoCapture(f)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        meta_arr.append((t, file_size, mtime, width, height, length, fps))

    tbl = pd.DataFrame(meta_arr,
                       columns=['filename', 'filesize', 'modification_time', 'width', 'height', 'length', 'fps'])
    tbl['modification_time'] = tbl['modification_time'].astype(np.uint64)

    tbl.to_csv(OUTPUT_PATH + 'train_metadata.csv', index=False)


def get_metadata_table_small_train():

    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename').index
    meta_arr = []
    for t in train_labels:
        print('Go for {}'.format(t))
        f = OUTPUT_FULL_VIDEO_PATH + t
        file_size = os.path.getsize(f)
        mtime = os.path.getmtime(f)

        cap = cv2.VideoCapture(f)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        meta_arr.append((t, file_size, mtime, width, height, length, fps))

    tbl = pd.DataFrame(meta_arr,
                       columns=['filename', 'filesize', 'modification_time', 'width', 'height', 'length', 'fps'])
    tbl['modification_time'] = tbl['modification_time'].astype(np.uint64)

    tbl.to_csv(OUTPUT_PATH + 'train_small_metadata.csv', index=False)


def get_metadata_table_audio():
    import librosa

    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename').index
    meta_arr = []
    for t in train_labels:
        print('Go for {}'.format(t))
        f = OUTPUT_FULL_AUDIO_PATH + t + '.mp3'
        if os.path.isfile(f):
            file_size = os.path.getsize(f)
            aud, sr = librosa.load(f, sr=None)
            length = aud.shape[0]
            shp = len(aud.shape)
            if shp > 1:
                print('Found')
                exit()
            min_aud = aud.min()
            max_aud = aud.max()
            avg_aud = aud.mean()

            meta_arr.append((t, file_size, sr, length, min_aud, max_aud, avg_aud))
        else:
            meta_arr.append((t, 0, 0, 0, 0, 0, 0))

    tbl = pd.DataFrame(meta_arr,
                       columns=['filename', 'filesize_audio', 'sample_rate', 'length_audio', 'min_aud', 'max_aud', 'avg_aud'])
    tbl['filesize'] = tbl['filesize'].astype(np.uint64)
    tbl['sample_rate'] = tbl['sample_rate'].astype(np.uint64)
    tbl['length'] = tbl['length'].astype(np.int64)

    tbl.to_csv(OUTPUT_PATH + 'train_audio_metadata.csv', index=False)


def get_metadata_table_test():

    subm = pd.read_csv(INPUT_PATH + 'submission_format.csv', index_col='filename')
    meta_arr = []
    files = list(subm.index.values)
    for t in files:
        print('Go for {}'.format(t))
        f = FULL_VIDEO_PATH + t
        file_size = os.path.getsize(f)
        mtime = os.path.getmtime(f)

        cap = cv2.VideoCapture(f)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        meta_arr.append((t, file_size, mtime, width, height, length, fps))

    tbl = pd.DataFrame(meta_arr,
                       columns=['filename', 'filesize', 'modification_time', 'width', 'height', 'length', 'fps'])
    tbl['modification_time'] = tbl['modification_time'].astype(np.uint64)

    tbl.to_csv(OUTPUT_PATH + 'test_metadata.csv', index=False)


def get_metadata_table_audio_test():
    import librosa

    subm = pd.read_csv(INPUT_PATH + 'submission_format.csv', index_col='filename')
    meta_arr = []
    files = list(subm.index.values)
    # files = ['e8JzAiHppZ.mp4']
    for t in files:
        print('Go for {}'.format(t))
        f = OUTPUT_FULL_AUDIO_PATH + t + '.mp3'
        if os.path.isfile(f):
            file_size = os.path.getsize(f)
            try:
                aud, sr = librosa.load(f, sr=None)
            except:
                print('Reading data error!')
                meta_arr.append((t, 0, 0, 0, 0, 0, 0))
                continue
            length = aud.shape[0]
            shp = len(aud.shape)
            if shp > 1:
                print('Found')
                exit()
            min_aud = aud.min()
            max_aud = aud.max()
            avg_aud = aud.mean()

            meta_arr.append((t, file_size, sr, length, min_aud, max_aud, avg_aud))
        else:
            meta_arr.append((t, 0, 0, 0, 0, 0, 0))

    tbl = pd.DataFrame(meta_arr,
                       columns=['filename', 'filesize_audio', 'sample_rate', 'length_audio', 'min_aud', 'max_aud', 'avg_aud'])
    tbl['filesize'] = tbl['filesize'].astype(np.uint64)
    tbl['sample_rate'] = tbl['sample_rate'].astype(np.uint64)
    tbl['length'] = tbl['length'].astype(np.int64)

    tbl.to_csv(OUTPUT_PATH + 'test_audio_metadata.csv', index=False)


def get_metadata_table_other():
    files = get_other_filelist()
    meta_arr = []
    for t in files:
        print('Go for {}'.format(t))
        f = FULL_VIDEO_PATH + t
        try:
            file_size = os.path.getsize(f)
            mtime = os.path.getmtime(f)

            cap = cv2.VideoCapture(f)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            meta_arr.append((t, file_size, mtime, width, height, length, fps))
        except:
            continue

    tbl = pd.DataFrame(meta_arr,
                       columns=['filename', 'filesize', 'modification_time', 'width', 'height', 'length', 'fps'])
    tbl['modification_time'] = tbl['modification_time'].astype(np.uint64)

    tbl.to_csv(OUTPUT_PATH + 'other_metadata.csv', index=False)


def get_metadata_table_audio_other():
    import librosa

    files = get_other_filelist()

    meta_arr = []
    # files = ['e8JzAiHppZ.mp4']
    for t in files:
        print('Go for {}'.format(t))
        f = OUTPUT_FULL_AUDIO_PATH + t + '.mp3'
        if os.path.isfile(f):
            file_size = os.path.getsize(f)
            try:
                aud, sr = librosa.load(f, sr=None)
            except:
                print('Reading data error!')
                meta_arr.append((t, 0, 0, 0, 0, 0, 0))
                continue
            length = aud.shape[0]
            shp = len(aud.shape)
            if shp > 1:
                print('Found')
                exit()
            min_aud = aud.min()
            max_aud = aud.max()
            avg_aud = aud.mean()

            meta_arr.append((t, file_size, sr, length, min_aud, max_aud, avg_aud))
        else:
            meta_arr.append((t, 0, 0, 0, 0, 0, 0))

    tbl = pd.DataFrame(meta_arr,
                       columns=['filename', 'filesize_audio', 'sample_rate', 'length_audio', 'min_aud', 'max_aud', 'avg_aud'])
    tbl['filesize'] = tbl['filesize'].astype(np.uint64)
    tbl['sample_rate'] = tbl['sample_rate'].astype(np.uint64)
    tbl['length'] = tbl['length'].astype(np.int64)

    tbl.to_csv(OUTPUT_PATH + 'other_audio_metadata.csv', index=False)


def get_modification_time_for_mini():
    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename').index
    files = glob.glob(INPUT_PATH + 'micro/*.mp4')

    meta_arr = []
    for f in files:
        print('Go for {}'.format(f))
        t = os.path.basename(f)
        file_size = os.path.getsize(f)
        mtime = os.path.getmtime(f)

        cap = cv2.VideoCapture(f)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        meta_arr.append((t, file_size, mtime, width, height, length))

    tbl = pd.DataFrame(meta_arr,
                       columns=['filename', 'filesize', 'modification_time', 'width', 'height', 'length'])
    tbl['modification_time'] = tbl['modification_time'].astype(np.uint64)
    tbl = tbl.sort_values('modification_time')

    tbl.to_csv(OUTPUT_PATH + 'metadata_mini_dataset.csv', index=False)


def create_all_sorted_videos_by_modification_time():
    use_cols = ['filename', 'modification_time', 'width', 'height', 'length']

    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename')
    meta_1_train = pd.read_csv(OUTPUT_PATH + 'train_metadata.csv', index_col='filename')
    train = pd.concat([train_labels, meta_1_train], axis=1)
    test = pd.read_csv(OUTPUT_PATH + 'test_metadata.csv', index_col='filename', usecols=use_cols)
    other = pd.read_csv(OUTPUT_PATH + 'other_metadata.csv', index_col='filename', usecols=use_cols)

    train['target'] = -1
    for i in range(len(ANIMAL_TYPE)):
        a = ANIMAL_TYPE[i]
        train.loc[train[a] == 1, 'target'] = i

    train = train.sort_values('modification_time')
    train['type'] = 'train'
    test = test.sort_values('modification_time')
    test['target'] = -1
    test['target'] = test['target'].astype(np.int16)
    test['type'] = 'testt'
    other['target'] = -1
    other['target'] = other['target'].astype(np.int16)
    other['type'] = 'other'

    feat_to_store = ['modification_time', 'target', 'type', 'width', 'height', 'length']
    full = pd.concat((train[feat_to_store], test[feat_to_store], other[feat_to_store]), axis=0)
    full = full.sort_values('modification_time')
    full[feat_to_store].to_csv(OUTPUT_PATH + 'full_videos_modification_time_sorted.csv')


if __name__ == '__main__':
    get_metadata_table()
    get_metadata_table_small_train()
    get_metadata_table_test()
    get_metadata_table_other()
    get_metadata_table_audio()
    get_metadata_table_audio_test()
    get_modification_time_for_mini()
    create_all_sorted_videos_by_modification_time()
