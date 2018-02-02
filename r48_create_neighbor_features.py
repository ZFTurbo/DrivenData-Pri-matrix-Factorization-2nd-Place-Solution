# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Calculate average preidctions for K nearest videos from "test" and "other" parts of dataset
'''

from a00_common_functions import *


def replace_columns(table, suffix):
    for a in ANIMAL_TYPE:
        new_name = a + '_' + suffix
        new_name = new_name.replace(' ', '_')
        new_name = new_name.replace('(', '_')
        new_name = new_name.replace(')', '_')
        table.rename(columns={a: new_name}, inplace=True)
    return table


def create_neighbour_features_from_preds_v2(store_num=50, use_data='inception'):
    # Use store_num*2 nearest predictions average
    order = pd.read_csv(OUTPUT_PATH + 'full_videos_modification_time_sorted.csv')
    if use_data == 'inception':
        subm_train = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_inception_GRU_v4_train.csv')
        subm_test = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_inception_GRU_v4_test.csv')
        subm_other = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_inception_GRU_v4_other.csv')
    if use_data == 'VGG16':
        subm_train = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_GRU_1024_VGG16_v11_train.csv')
        subm_test = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_GRU_1024_VGG16_v11_test.csv')
        subm_other = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_GRU_1024_VGG16_v11_other.csv')
    print('Len train:', len(subm_train))
    print('Len test:', len(subm_test))
    print('Len other:', len(subm_other))
    # subm = pd.concat((subm_train, subm_test, subm_other), axis=0)
    subm = pd.concat((subm_test, subm_other), axis=0)
    print('Len concat:', len(subm))

    accum = order[['filename']].copy()
    for v in ANIMAL_TYPE:
        accum[v] = 0.0
    accum['counter'] = 0

    for i in range(-store_num, store_num+1):
        if i == 0:
            continue
        print('Shift: {}'.format(i))
        order['filename_shift'] = order['filename'].shift(i)

        subm_copy = subm.copy()
        subm_copy.rename(columns={"filename": "filename_shift"}, inplace=True)
        prev = pd.merge(order[['filename', 'filename_shift']], subm_copy, on='filename_shift', how='left')
        prev.fillna(-1, inplace=True)
        prev.drop('filename_shift', axis=1, inplace=True)

        cond = (prev[ANIMAL_TYPE[0]] >= 0)
        accum.loc[cond, ANIMAL_TYPE] += prev.loc[cond, ANIMAL_TYPE]
        accum.loc[cond, 'counter'] += 1

    accum = accum[(accum['counter'] > 0)]
    for a in ANIMAL_TYPE:
        accum[a] = accum[a] / accum['counter']
    accum.drop('counter', axis=1, inplace=True)

    accum_train = accum[accum['filename'].isin(subm_train['filename'])]
    accum_train.to_csv(FEATURES_PATH + 'neighbours_{}_{}_average_pred_train.csv'.format(use_data, 2*store_num), index=False)
    accum_test = accum[accum['filename'].isin(subm_test['filename'])]
    accum_test.to_csv(FEATURES_PATH + 'neighbours_{}_{}_average_pred_test.csv'.format(use_data, 2 * store_num), index=False)
    accum_test = accum[accum['filename'].isin(subm_other['filename'])]
    accum_test.to_csv(FEATURES_PATH + 'neighbours_{}_{}_average_pred_other.csv'.format(use_data, 2 * store_num), index=False)


if __name__ == '__main__':
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50]:
        create_neighbour_features_from_preds_v2(i, 'inception')
        create_neighbour_features_from_preds_v2(i, 'VGG16')
