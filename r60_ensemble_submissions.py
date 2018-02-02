# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

'''
Get predictions from all 2nd level classifiers and create ensemble.
'''


from a00_common_functions import *


def read_all_tables(s1):
    train_labels = pd.read_csv(INPUT_PATH + 'train_labels.csv')
    valid = []
    test = []
    for s in s1:
        v = pd.read_csv(s[0], index_col='filename')
        t = pd.read_csv(s[1], index_col='filename')
        valid.append(v[ANIMAL_TYPE].as_matrix())
        test.append(t[ANIMAL_TYPE].as_matrix())
    valid = np.array(valid)
    test = np.array(test)
    print(valid.shape)
    print(test.shape)
    return t, valid, test, train_labels


def make_simple_ensemble(t, valid, test, train_labels, weights, save_csv=True):
    # print('Weights: {}'.format(weights))
    valid_total = np.zeros(valid.shape[1:])
    test_total = np.zeros(test.shape[1:])
    for i in range(valid.shape[0]):
        valid_total += weights[i]*valid[i]
        test_total += weights[i]*test[i]
    valid_total /= np.array(weights).sum()
    test_total /= np.array(weights).sum()
    score = get_score(valid_total, train_labels[ANIMAL_TYPE].as_matrix(), False)

    if save_csv is True:
        t[ANIMAL_TYPE] = test_total
        t.to_csv(SUBM_PATH + 'subm_ensemble_simple_score_{}.csv'.format(score))
        print('Predicted score: {}'.format(score))
    return score


if __name__ == '__main__':
    start_time = time.time()

    EP = SUBM_PATH
    # train, test pairs
    subm_list = [
        (EP + 'subm_xgboost_1_train.csv', EP + 'subm_xgboost_1_test.csv'),
        (EP + 'subm_xgboost_2_train.csv', EP + 'subm_xgboost_2_test.csv'),
        (EP + 'subm_lightgbm_1_train.csv', EP + 'subm_lightgbm_1_test.csv'),
        (EP + 'subm_lightgbm_2_train.csv', EP + 'subm_lightgbm_2_test.csv'),
        (EP + 'subm_keras_blender_1_train.csv', EP + 'subm_keras_blender_1_test.csv'),
        (EP + 'subm_keras_blender_2_train.csv', EP + 'subm_keras_blender_2_test.csv'),
    ]

    t, valid, test, train_labels = read_all_tables(subm_list)
    best_weights = [2, 2, 4, 4, 2, 2]

    if 0:
        weights = best_weights.copy()
        best_weights = [2, 2, 4, 4, 2, 2]
        best_score = 1000
        for j in range(9, 0, -1):
            best_current_weigth = best_weights[j]
            for k in range(-best_weights[j], 20, 1):
                print('Go for {}. Iter: {}'.format(j, k))
                weights = best_weights.copy()
                weights[j] += k
                score = make_simple_ensemble(t, valid, test, train_labels, weights, False)
                if score < best_score:
                    best_score = score
                    best_current_weigth = weights[j]
                    print('Update: {} Score: {}'.format(weights, best_score))
            best_weights[j] = best_current_weigth

        print('Best weights:', best_weights)
        print('Best score:', best_score)

    make_simple_ensemble(t, valid, test, train_labels, best_weights, True)
    print("Elapsed time overall: %s seconds" % (time.time() - start_time))

'''
LS: 0.014978 weights = [2, 2, 4, 4, 2, 2] LB: 0.013739
'''