# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Second level model, which uses all previously generated features, based on XGBoost GBM classifier
Run 2 times with different parameters to increase solution stability
'''

import datetime
from operator import itemgetter
from a00_common_functions import *


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def create_xgboost_model(train, real_labels, features, eta_value, depth, iter1):
    import xgboost as xgb
    print('XGBoost version: {}'.format(xgb.__version__))
    start_time = time.time()

    num_folds = 5
    eta = eta_value
    max_depth = depth
    subsample = 0.95
    colsample_bytree = 0.95
    eval_metric = 'mlogloss'
    unique_target = np.array(sorted(train['target'].unique()))
    print('Target length: {}: {}'.format(len(unique_target), unique_target))

    for i in range(len(unique_target)):
        train.loc[train['target'] == unique_target[i], 'target'] = i

    log_str = 'XGBoost iter {}. FOLDS: {} METRIC: {} ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(0,
                                                                                                           num_folds,
                                                                                                           eval_metric,
                                                                                                           eta,
                                                                                                           max_depth,
                                                                                                           subsample,
                                                                                                           colsample_bytree)
    print(log_str)
    params = {
        "objective": "multi:softprob",
        "num_class": len(unique_target),
        "booster": "gbtree",
        "eval_metric": eval_metric,
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": 2017,
        "nthread": 6,
        # 'gpu_id': 0,
        # 'updater': 'grow_gpu_hist',
    }
    num_boost_round = 1500
    early_stopping_rounds = 50

    print('Train shape:', train.shape)
    # print('Features:', features)

    files, ret = get_kfold_split(5)
    files = np.array(files)

    model_list = []
    full_preds = np.zeros((len(files), 24), dtype=np.float32)
    counts = np.zeros((len(files), 24), dtype=np.float32)
    fold_num = 0
    for train_index, valid_index in ret:
        fold_num += 1
        print('Start fold {}'.format(fold_num))
        train_files = files[train_index]
        valid_files = files[valid_index]
        X_train = train.loc[train_files]
        X_valid = train.loc[valid_files]
        y_train = X_train['target']
        y_valid = X_valid['target']

        print('Train data:', X_train.shape)
        print('Valid data:', X_valid.shape)

        dtrain = xgb.DMatrix(X_train[features].as_matrix(), y_train)
        dvalid = xgb.DMatrix(X_valid[features].as_matrix(), y_valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                        early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        model_list.append(gbm)

        imp = get_importance(gbm, features)
        print('Importance: {}'.format(imp))

        print("Validating...")
        pred = gbm.predict(dvalid, ntree_limit=gbm.best_iteration + 1)
        full_preds[valid_index, :] += pred
        counts[valid_index, :] += 1

        real = real_labels.loc[valid_files, ANIMAL_TYPE].as_matrix()
        score = get_score(pred, real)
        print('Fold {} score: {}'.format(fold_num, score))

    full_preds /= counts
    real = real_labels[ANIMAL_TYPE].as_matrix()
    score = get_score(full_preds, real)
    print('Score: {}'.format(score))
    print('Time: {} sec'.format(time.time() - start_time))

    s = pd.DataFrame(files, columns=['filename'])
    for a in ANIMAL_TYPE:
        s[a] = 0.0
    s[ANIMAL_TYPE] = full_preds
    s.to_csv(SUBM_PATH + 'subm_{}_{}_train.csv'.format('xgboost', iter1), index=False)

    return score, full_preds, model_list


def predict_with_xgboost_model(test, features, models_list):
    import xgboost as xgb

    dtest = xgb.DMatrix(test[features].as_matrix())
    full_preds = []
    for m in models_list:
        preds = m.predict(dtest, ntree_limit=m.best_iteration + 1)
        full_preds.append(preds)
    preds = np.array(full_preds).mean(axis=0)
    return preds


def get_readable_date(dt):
    return datetime.datetime.fromtimestamp(dt).strftime('%Y-%m-%d %H:%M:%S')


def replace_columns(table, suffix):
    for a in ANIMAL_TYPE:
        new_name = a + '_' + suffix
        new_name = new_name.replace(' ', '_')
        new_name = new_name.replace('(', '_')
        new_name = new_name.replace(')', '_')
        table.rename(columns={a: new_name}, inplace=True)
    return table


def read_tables():
    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv', index_col='filename')
    meta_1 = pd.read_csv(OUTPUT_PATH + 'train_metadata.csv', index_col='filename')
    meta_2 = pd.read_csv(OUTPUT_PATH + 'train_audio_metadata.csv', index_col='filename')
    meta_mini = pd.read_csv(OUTPUT_PATH + 'metadata_mini_dataset.csv', index_col='filename')
    meta_mini['modification_time_mini'] = meta_mini['modification_time']
    oof_3dvgg = pd.read_csv(FEATURES_PATH + 'VGG_3D_24_56_56_v2_train.csv', index_col='filename')
    oof_3dvgg = replace_columns(oof_3dvgg, '3dvgg')
    oof_resnet = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_for_resnet_v4_train.csv', index_col='filename')
    oof_resnet = replace_columns(oof_resnet, 'resnet')
    oof_inception = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_inception_v1_train.csv', index_col='filename')
    oof_inception = replace_columns(oof_inception, 'inception')
    oof_inception_lstm = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_inception_LSTM_v2_train.csv', index_col='filename')
    oof_inception_lstm = replace_columns(oof_inception_lstm, 'inception_lstm')
    oof_inception_gru = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_inception_GRU_v4_train.csv', index_col='filename')
    oof_inception_gru = replace_columns(oof_inception_gru, 'inception_gru')
    oof_audio = pd.read_csv(FEATURES_PATH + 'zf_simple_audio_model_v1_train.csv', index_col='filename')
    oof_audio = replace_columns(oof_audio, 'audio')
    image_hashes = pd.read_csv(OUTPUT_PATH + 'image_hashes_data_train.csv', index_col='filename')
    oof_im_hash_gru = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_GRU_ImageHash_train.csv', index_col='filename')
    oof_im_hash_gru = replace_columns(oof_im_hash_gru, 'im_hash_gru')
    oof_vgg16_gru = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_GRU_1024_VGG16_v11_train.csv', index_col='filename')
    oof_vgg16_gru = replace_columns(oof_vgg16_gru, 'vgg16_gru_v11')

    train = pd.concat([train_labels, meta_1, meta_2], axis=1)

    train = pd.merge(train, meta_mini[['modification_time_mini']], left_index=True, right_index=True, how='left')
    train = pd.merge(train, oof_3dvgg, left_index=True, right_index=True, how='left')
    train = pd.merge(train, oof_resnet, left_index=True, right_index=True, how='left')
    train = pd.merge(train, oof_inception, left_index=True, right_index=True, how='left')
    train = pd.merge(train, oof_inception_lstm, left_index=True, right_index=True, how='left')
    train = pd.merge(train, oof_inception_gru, left_index=True, right_index=True, how='left')
    train = pd.merge(train, oof_audio, left_index=True, right_index=True, how='left')
    train = pd.merge(train, image_hashes, left_index=True, right_index=True, how='left')
    train = pd.merge(train, oof_im_hash_gru, left_index=True, right_index=True, how='left')
    train = pd.merge(train, oof_vgg16_gru, left_index=True, right_index=True, how='left')

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50]:
        neigbour_average = pd.read_csv(FEATURES_PATH + 'neighbours_inception_{}_average_pred_train.csv'.format(2*i), index_col='filename')
        neigbour_average = replace_columns(neigbour_average, '{}_inception_neighbour_avg'.format(i))
        train = pd.merge(train, neigbour_average, left_index=True, right_index=True, how='left')

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50]:
        neigbour_average = pd.read_csv(FEATURES_PATH + 'neighbours_VGG16_{}_average_pred_train.csv'.format(2*i), index_col='filename')
        neigbour_average = replace_columns(neigbour_average, '{}_VGG16_neighbour_avg'.format(i))
        train = pd.merge(train, neigbour_average, left_index=True, right_index=True, how='left')

    features = sorted(list(set(train.columns.values) - set(ANIMAL_TYPE)))
    features.remove('fps')
    features.remove('sample_rate')

    real_labels = train_labels[ANIMAL_TYPE].copy()
    print('Features [{}]: {}'.format(len(features), features))
    train['target'] = -1
    for i in range(len(ANIMAL_TYPE)):
        a = ANIMAL_TYPE[i]
        train.loc[train[a] == 1, 'target'] = i
    train.fillna(-1, inplace=True)

    return train, real_labels, features


def read_tst_table():
    subm = pd.read_csv(INPUT_PATH + 'submission_format.csv', index_col='filename')
    meta_1 = pd.read_csv(OUTPUT_PATH + 'test_metadata.csv', index_col='filename')
    meta_2 = pd.read_csv(OUTPUT_PATH + 'test_audio_metadata.csv', index_col='filename')
    meta_mini = pd.read_csv(OUTPUT_PATH + 'metadata_mini_dataset.csv', index_col='filename')
    meta_mini['modification_time_mini'] = meta_mini['modification_time']
    oof_3dvgg = pd.read_csv(FEATURES_PATH + 'VGG_3D_24_56_56_v2_test.csv', index_col='filename')
    oof_3dvgg = replace_columns(oof_3dvgg, '3dvgg')
    oof_resnet = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_for_resnet_v4_test.csv', index_col='filename')
    oof_resnet = replace_columns(oof_resnet, 'resnet')
    oof_inception = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_inception_v1_test.csv', index_col='filename')
    oof_inception = replace_columns(oof_inception, 'inception')
    oof_inception_lstm = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_inception_LSTM_v2_test.csv', index_col='filename')
    oof_inception_lstm = replace_columns(oof_inception_lstm, 'inception_lstm')
    oof_inception_gru = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_inception_GRU_v4_test.csv', index_col='filename')
    oof_inception_gru = replace_columns(oof_inception_gru, 'inception_gru')
    oof_audio = pd.read_csv(FEATURES_PATH + 'zf_simple_audio_model_v1_test.csv', index_col='filename')
    oof_audio = replace_columns(oof_audio, 'audio')
    image_hashes = pd.read_csv(OUTPUT_PATH + 'image_hashes_data_test.csv', index_col='filename')
    oof_im_hash_gru = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_GRU_ImageHash_test.csv', index_col='filename')
    oof_im_hash_gru = replace_columns(oof_im_hash_gru, 'im_hash_gru')
    oof_vgg16_gru = pd.read_csv(FEATURES_PATH + 'ZF_full_keras_model_GRU_1024_VGG16_v11_test.csv', index_col='filename')
    oof_vgg16_gru = replace_columns(oof_vgg16_gru, 'vgg16_gru_v11')

    test = pd.concat([subm, meta_1, meta_2], axis=1)

    test = pd.merge(test, meta_mini[['modification_time_mini']], left_index=True, right_index=True, how='left')
    test = pd.merge(test, oof_3dvgg, left_index=True, right_index=True, how='left')
    test = pd.merge(test, oof_resnet, left_index=True, right_index=True, how='left')
    test = pd.merge(test, oof_inception, left_index=True, right_index=True, how='left')
    test = pd.merge(test, oof_inception_lstm, left_index=True, right_index=True, how='left')
    test = pd.merge(test, oof_inception_gru, left_index=True, right_index=True, how='left')
    test = pd.merge(test, oof_audio, left_index=True, right_index=True, how='left')
    test = pd.merge(test, image_hashes, left_index=True, right_index=True, how='left')
    test = pd.merge(test, oof_im_hash_gru, left_index=True, right_index=True, how='left')
    test = pd.merge(test, oof_vgg16_gru, left_index=True, right_index=True, how='left')

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50]:
        neigbour_average = pd.read_csv(FEATURES_PATH + 'neighbours_inception_{}_average_pred_test.csv'.format(2*i), index_col='filename')
        neigbour_average = replace_columns(neigbour_average, '{}_inception_neighbour_avg'.format(i))
        test = pd.merge(test, neigbour_average, left_index=True, right_index=True, how='left')

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50]:
        neigbour_average = pd.read_csv(FEATURES_PATH + 'neighbours_VGG16_{}_average_pred_test.csv'.format(2*i), index_col='filename')
        neigbour_average = replace_columns(neigbour_average, '{}_VGG16_neighbour_avg'.format(i))
        test = pd.merge(test, neigbour_average, left_index=True, right_index=True, how='left')

    test.fillna(-1, inplace=True)
    return subm, test


def run_xgboost(eta, depth, iter1):
    train, real_labels, features = read_tables()
    gbm_type = 'xgboost'

    if 1:
        score, valid_pred, model_list = create_xgboost_model(train, real_labels, features, eta, depth, iter1)
        save_in_file((score, valid_pred, model_list), MODELS_PATH + 'xgboost_last_run_models_{}.pklz'.format(iter1))
    else:
        score, valid_pred, model_list = load_from_file(MODELS_PATH + 'xgboost_last_run_models_{}.pklz'.format(iter1))

    subm, test = read_tst_table()
    print('Check NaNs...')
    for f in features:
        inds = pd.isnull(test[[f]]).any(1).nonzero()[0]
        if len(inds) > 0:
            print('NaN in column {}'.format(f))

    preds = predict_with_xgboost_model(test, features, model_list)

    subm[ANIMAL_TYPE] = preds
    subm.to_csv(SUBM_PATH + 'subm_{}_{}_test.csv'.format(gbm_type, iter1))


if __name__ == '__main__':
    start_time = time.time()
    run_xgboost(0.05, 5, 1)
    run_xgboost(0.09, 4, 2)
    print("Elapsed time overall: %s seconds" % (time.time() - start_time))


'''
With all standard features + 2 neighbours features:
Score: 0.015164
'''