# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Second level model, which uses all previously generated features, based on LightGBM classifier
Run 2 times with different parameters to increase solution stability
'''

import datetime
from a00_common_functions import *
from r50_second_level_model_xgboost import read_tables, read_tst_table


def print_importance(features, gbm, prnt=True):
    max_report = 100
    importance_arr = sorted(list(zip(features, gbm.feature_importance())), key=lambda x: x[1], reverse=True)
    s1 = 'Importance TOP {}: '.format(max_report)
    for d in importance_arr[:max_report]:
        s1 += str(d) + ', '
    if prnt:
        print(s1)
    return importance_arr


def create_lightgbm_model(train, real_labels, features, lr_value, iter1):
    import lightgbm as lgb
    print('LightGBM version: {}'.format(lgb.__version__))
    start_time = time.time()

    # Debug
    iter = 0
    random_state = 10
    rs = 69
    learning_rate = lr_value
    num_leaves = 63
    feature_fraction = 0.95
    bagging_fraction = 0.95
    boosting_type = 'gbdt'
    # boosting_type = 'dart'
    min_data_in_leaf = 1000
    max_bin = 255
    bagging_freq = 0
    drop_rate = 0.05
    skip_drop = 0.5
    max_drop = 1

    params = {
        'task': 'train',
        'boosting_type': boosting_type,
        'objective': 'multiclass',
        'num_class': 24,
        'metric': {'multi_logloss'},
        'device': 'cpu',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'min_data_in_leaf': min_data_in_leaf,
        'bagging_freq': bagging_freq,
        'max_bin': max_bin,
        'drop_rate': drop_rate,
        'skip_drop': skip_drop,
        'max_drop': max_drop,
        'feature_fraction_seed': random_state + iter,
        'bagging_seed': random_state + iter,
        'data_random_seed': random_state + iter,
        'verbose': 0,
        'num_threads': 11,
    }
    log_str = 'LightGBM iter {}. PARAMS: {}'.format(iter, sorted(params.items()))
    print(log_str)
    num_boost_round = 10000
    early_stopping_rounds = 50

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

        lgb_train = lgb.Dataset(X_train[features].as_matrix(), y_train)
        lgb_eval = lgb.Dataset(X_valid[features].as_matrix(), y_valid, reference=lgb_train)

        gbm = lgb.train(params, lgb_train, num_boost_round=num_boost_round,
                        early_stopping_rounds=early_stopping_rounds, valid_sets=[lgb_eval], verbose_eval=True)

        print_importance(features, gbm, True)
        model_list.append(gbm)

        print("Validating...")
        pred = gbm.predict(X_valid[features].as_matrix(), num_iteration=gbm.best_iteration)
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
    s.to_csv(SUBM_PATH + 'subm_{}_{}_train.csv'.format('lightgbm', iter1), index=False)

    return score, full_preds, model_list


def predict_with_lightgbm_model(test, features, models_list):
    dtest = test[features].as_matrix()
    full_preds = []
    total = 0
    for m in models_list:
        total += 1
        print('Process test model: {}'.format(total))
        preds = m.predict(dtest, num_iteration=m.best_iteration)
        full_preds.append(preds)
    preds = np.array(full_preds).mean(axis=0)
    return preds


def get_readable_date(dt):
    return datetime.datetime.fromtimestamp(dt).strftime('%Y-%m-%d %H:%M:%S')


def run_lightgbm(lr, iter1):
    train, real_labels, features = read_tables()
    gbm_type = 'lightgbm'
    score, valid_pred, model_list = create_lightgbm_model(train, real_labels, features, lr, iter1)
    save_in_file((score, valid_pred, model_list), MODELS_PATH + 'lightgbm_last_run_models.pklz')

    subm, test = read_tst_table()
    preds = predict_with_lightgbm_model(test, features, model_list)

    subm[ANIMAL_TYPE] = preds
    subm.to_csv(SUBM_PATH + 'subm_{}_{}_test.csv'.format(gbm_type, iter1))


if __name__ == '__main__':
    start_time = time.time()
    run_lightgbm(0.010, 1)
    run_lightgbm(0.011, 2)
    print("Elapsed time overall: %s seconds" % (time.time() - start_time))


'''
New neighbor features:
Fold 1 score: 0.014808198763102626
Fold 2 score: 0.015026676482215178
Fold 3 score: 0.015579144322001437
Fold 4 score: 0.015285618327356101
Fold 5 score: 0.015528994161269473
Score: 0.015245703350484694
'''
