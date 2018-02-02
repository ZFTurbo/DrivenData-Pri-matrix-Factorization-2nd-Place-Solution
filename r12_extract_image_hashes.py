# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Extract phash and dhash for every frame of each video
'''

from a00_common_functions import *
import imagehash
from PIL import Image


OUTPUT_IMAGE_HASH_FOLDER = OUTPUT_PATH + 'image_hashes/'
if not os.path.isdir(OUTPUT_IMAGE_HASH_FOLDER):
    os.mkdir(OUTPUT_IMAGE_HASH_FOLDER)


def get_image_hashes_for_train(reverse=False):
    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename').index
    if reverse is True:
        train_labels = train_labels[::-1]
    for t in train_labels:
        f = FULL_VIDEO_PATH + t
        if os.path.isfile(f):
            print('Go for {}...'.format(t))
            out_file = OUTPUT_IMAGE_HASH_FOLDER + t + '.pklz'
            if os.path.isfile(out_file):
                print('Already exists. Skip!')
                continue
            v = read_video(f)
            hashes = []
            for i in range(v.shape[0]):
                image = Image.fromarray(cv2.cvtColor(v[i], cv2.COLOR_BGR2RGB))
                # show_image(v[i])
                # image.show()
                phash = imagehash.phash(image)
                dhash = imagehash.dhash(image)
                # print(phash, dhash)
                hashes.append((phash, dhash))
            save_in_file(hashes, out_file)


def get_image_hashes_for_test(reverse=False):
    subm = pd.read_csv(INPUT_PATH + 'submission_format.csv', index_col='filename')
    test_labels = list(subm.index.values)
    if reverse is True:
        test_labels = test_labels[::-1]
    for t in test_labels:
        f = FULL_VIDEO_PATH + t
        if os.path.isfile(f):
            print('Go for {}...'.format(t))
            out_file = OUTPUT_IMAGE_HASH_FOLDER + t + '.pklz'
            if os.path.isfile(out_file):
                print('Already exists. Skip!')
                continue
            v = read_video(f)
            hashes = []
            for i in range(v.shape[0]):
                image = Image.fromarray(cv2.cvtColor(v[i], cv2.COLOR_BGR2RGB))
                # show_image(v[i])
                # image.show()
                phash = imagehash.phash(image)
                dhash = imagehash.dhash(image)
                # print(phash, dhash)
                hashes.append((phash, dhash))
            save_in_file(hashes, out_file)


def get_image_hash_features_train():
    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename').index
    out = open(OUTPUT_PATH + 'image_hashes_data_train.csv', 'w')
    out.write('filename,max_cons_phash,max_cons_dhash,avg_cons_phash,avg_cons_dhash,max_overall_phash,max_overall_dhash,avg_overall_phash,avg_overall_dhash\n')
    for t in train_labels:
        f = OUTPUT_IMAGE_HASH_FOLDER + t + '.pklz'
        print('Go for {}'.format(f))
        hashes = load_from_file(f)
        out.write(t)

        max_cons_phash = -1
        max_cons_dhash = -1
        avg_cons_phash = 0
        avg_cons_dhash = 0
        for i in range(1, len(hashes)):
            h1_p, h1_d = hashes[i]
            h2_p, h2_d = hashes[i-1]
            diff_p = h1_p - h2_p
            diff_d = h1_d - h2_d
            avg_cons_phash += diff_p
            avg_cons_dhash += diff_d
            if diff_p > max_cons_phash:
                max_cons_phash = diff_p
            if diff_d > max_cons_dhash:
                max_cons_dhash = diff_d
        avg_cons_phash /= (len(hashes)-1)
        avg_cons_dhash /= (len(hashes) - 1)
        # print(max_cons_phash, max_cons_dhash, avg_cons_phash, avg_cons_dhash)
        out.write(',' + str(max_cons_phash))
        out.write(',' + str(max_cons_dhash))
        out.write(',' + str(avg_cons_phash))
        out.write(',' + str(avg_cons_dhash))

        avg_cons_phash = 0
        avg_cons_dhash = 0
        total = 0
        for i in range(0, len(hashes)):
            for j in range(i+1, len(hashes)):
                h1_p, h1_d = hashes[i]
                h2_p, h2_d = hashes[j]
                diff_p = h1_p - h2_p
                diff_d = h1_d - h2_d
                avg_cons_phash += diff_p
                avg_cons_dhash += diff_d
                if diff_p > max_cons_phash:
                    max_cons_phash = diff_p
                if diff_d > max_cons_dhash:
                    max_cons_dhash = diff_d
                total += 1
        avg_cons_phash /= total
        avg_cons_dhash /= total
        # print(max_cons_phash, max_cons_dhash, avg_cons_phash, avg_cons_dhash)
        out.write(',' + str(max_cons_phash))
        out.write(',' + str(max_cons_dhash))
        out.write(',' + str(avg_cons_phash))
        out.write(',' + str(avg_cons_dhash))
        out.write('\n')

    out.close()


def get_image_hash_features_test():
    subm = pd.read_csv(INPUT_PATH + 'submission_format.csv', index_col='filename')
    test_files = list(subm.index.values)
    out = open(OUTPUT_PATH + 'image_hashes_data_test.csv', 'w')
    out.write('filename,max_cons_phash,max_cons_dhash,avg_cons_phash,avg_cons_dhash,max_overall_phash,max_overall_dhash,avg_overall_phash,avg_overall_dhash\n')
    for t in test_files:
        f = OUTPUT_IMAGE_HASH_FOLDER + t + '.pklz'
        print('Go for {}'.format(f))
        hashes = load_from_file(f)
        out.write(t)

        max_cons_phash = -1
        max_cons_dhash = -1
        avg_cons_phash = 0
        avg_cons_dhash = 0
        for i in range(1, len(hashes)):
            h1_p, h1_d = hashes[i]
            h2_p, h2_d = hashes[i-1]
            diff_p = h1_p - h2_p
            diff_d = h1_d - h2_d
            avg_cons_phash += diff_p
            avg_cons_dhash += diff_d
            if diff_p > max_cons_phash:
                max_cons_phash = diff_p
            if diff_d > max_cons_dhash:
                max_cons_dhash = diff_d
        avg_cons_phash /= (len(hashes)-1)
        avg_cons_dhash /= (len(hashes) - 1)
        # print(max_cons_phash, max_cons_dhash, avg_cons_phash, avg_cons_dhash)
        out.write(',' + str(max_cons_phash))
        out.write(',' + str(max_cons_dhash))
        out.write(',' + str(avg_cons_phash))
        out.write(',' + str(avg_cons_dhash))

        avg_cons_phash = 0
        avg_cons_dhash = 0
        total = 0
        for i in range(0, len(hashes)):
            for j in range(i+1, len(hashes)):
                h1_p, h1_d = hashes[i]
                h2_p, h2_d = hashes[j]
                diff_p = h1_p - h2_p
                diff_d = h1_d - h2_d
                avg_cons_phash += diff_p
                avg_cons_dhash += diff_d
                if diff_p > max_cons_phash:
                    max_cons_phash = diff_p
                if diff_d > max_cons_dhash:
                    max_cons_dhash = diff_d
                total += 1
        avg_cons_phash /= total
        avg_cons_dhash /= total
        # print(max_cons_phash, max_cons_dhash, avg_cons_phash, avg_cons_dhash)
        out.write(',' + str(max_cons_phash))
        out.write(',' + str(max_cons_dhash))
        out.write(',' + str(avg_cons_phash))
        out.write(',' + str(avg_cons_dhash))
        out.write('\n')

    out.close()


def create_image_hashes_pklz_train(part):
    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename').index
    ret = dict()
    if part == 0:
        train_labels = train_labels[:len(train_labels) // 4]
    elif part == 1:
        train_labels = train_labels[len(train_labels) // 4:len(train_labels) // 2]
    elif part == 2:
        train_labels = train_labels[len(train_labels) // 2:3*len(train_labels) // 4]
    else:
        train_labels = train_labels[3*len(train_labels) // 4:]
    for t in train_labels:
        f = OUTPUT_IMAGE_HASH_FOLDER + t + '.pklz'
        print('Go for {}'.format(f))
        hashes = load_from_file(f)
        hashes = np.array(hashes)
        s = []
        for i in range(hashes.shape[0]):
            p = list('{:064b}{:064b}'.format(int(str(hashes[i][0]), 16), int(str(hashes[i][1]), 16)))
            s.append(p)
        s = np.array(s, dtype=np.uint8)
        ret[t] = s
    save_in_file(ret, OUTPUT_PATH + 'image_hashes_train_uncompressed_part_{}.pklz'.format(part))


def create_image_hashes_pklz_test():
    subm = pd.read_csv(INPUT_PATH + 'submission_format.csv', index_col='filename')
    test_files = list(subm.index.values)
    ret = dict()
    for t in test_files:
        f = OUTPUT_IMAGE_HASH_FOLDER + t + '.pklz'
        print('Go for {}'.format(f))
        hashes = load_from_file(f)
        hashes = np.array(hashes)
        s = []
        for i in range(hashes.shape[0]):
            p = list('{:064b}{:064b}'.format(int(str(hashes[i][0]), 16), int(str(hashes[1][1]), 16)))
            s.append(p)
        s = np.array(s, dtype=np.uint8)
        ret[t] = s
    save_in_file(ret, OUTPUT_PATH + 'image_hashes_test.pklz')


if __name__ == '__main__':
    get_image_hashes_for_train(True)
    get_image_hashes_for_test(True)
    get_image_hash_features_train()
    get_image_hash_features_test()
    create_image_hashes_pklz_train(0)
    create_image_hashes_pklz_train(1)
    create_image_hashes_pklz_train(2)
    create_image_hashes_pklz_train(3)
    create_image_hashes_pklz_test()