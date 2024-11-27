import os
import sys
import time
import math
import numpy as np
import pandas as pd

from sklearn import svm, pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.kernel_approximation import Nystroem

challenge_path = './crps/challenges.csv'
response_path = './crps/responses_fpuf_basic.csv'
transform = 'parity'
# n_samples = [100000000]
n_samples = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 
             25000, 50000, 100000, 200000, 400000, 800000, 1600000, 3200000, 6400000, 12800000, 
             25000000, 50000000, 100000000]

def fetch_CRP(challenge_path, response_path):
    if not os.path.exists(challenge_path):
        print('The path specified for challenge CSV file does not exist')
        sys.exit()
     
    if not os.path.exists(response_path):
        print('The path specified for response CSV file does not exist')
        sys.exit()
    
    np_challenges = pd.read_csv(challenge_path, header=None).values
    np_responses = pd.read_csv(response_path, header=None).values
    np_challenges[np_challenges==0] = -1
    np_responses[np_responses==0] = -1

    return np_challenges, np_responses

def get_parity_features(challenges):
    n_samples, n_length = challenges.shape
    parity_features = np.zeros((n_samples, n_length + 1))
    parity_features[:, 0:1] = np.ones((n_samples, 1))
    for i in range(2, n_length + 2):
        parity_features[:, i - 1:i] = np.prod(challenges[:, 0:i - 1], axis=1).reshape((n_samples, 1))
    
    return parity_features

def linear_SVM_attack(train_features, test_features, train_labels, test_labels):
    print("1. SVM Classifier")
    # linear_svm = svm.LinearSVC(C=1.0, dual=True, max_iter=10000)
    linear_svm = SGDClassifier(loss='hinge', max_iter=10000, n_jobs=-1)
    linear_svm.fit(train_features, train_labels.ravel())
    acc_linear_svm = linear_svm.score(test_features, test_labels.ravel())
    print('Linear SVM Accuracy: %f\n' % acc_linear_svm)
    
    return acc_linear_svm

def LR_attack(train_features, test_features, train_labels, test_labels):
    print("2. Logistic Regression Classifier")
    # lr = LogisticRegression(random_state=0, max_iter=10000)
    lr = SGDClassifier(loss='log_loss', max_iter=10000, n_jobs=-1)
    lr.fit(train_features, train_labels.ravel())
    acc_lr = lr.score(test_features, test_labels.ravel())
    print('Logistic Regression Accuracy: %f\n' % acc_lr)

    return acc_lr

def rbf_SVM_attack(train_features, test_features, train_labels, test_labels):
    print("3. SVM with RBF Kernal Classifier")
    rbf_svm = svm.SVC(kernel='rbf', max_iter=10000)
    rbf_svm.fit(train_features, train_labels.ravel())
    acc_rbf_svm = rbf_svm.score(test_features, test_labels.ravel())
    print('RBF SVM Accuracy: %f\n' % acc_rbf_svm)

    return acc_rbf_svm

def rbf_SVM_attack_fast(train_features, test_features, train_labels, test_labels):
    print("3. SVM with RBF Kernal Classifier - Fast Version")
    # clf = svm.LinearSVC(C=1.0, dual=True, max_iter=10000)
    clf = SGDClassifier(loss='hinge', max_iter=10000, warm_start=True, n_jobs=-1)
    feature_map_nystroem = Nystroem(kernel='rbf', gamma=1/28, n_components=1200, n_jobs=-1)
    nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem), ("svm", clf)])
    sample_th = 800000
    if train_features.shape[0] >= sample_th:
        for i in range(math.ceil(train_features.shape[0]/sample_th)):
            print('Total iters: %d    Current iter: %d' % (math.ceil(train_features.shape[0]/sample_th), i + 1))
            if sample_th*(i + 1) <= train_features.shape[0]:
                nystroem_approx_svm.fit(train_features[sample_th*i:sample_th*(i + 1)], train_labels[sample_th*i:sample_th*(i + 1)].ravel())
            else:
                nystroem_approx_svm.fit(train_features[sample_th*i:train_features.shape[0]], train_labels[sample_th*i:train_features.shape[0]].ravel())
        acc_rbf_svm = []
        for i in range(math.ceil(test_features.shape[0]/sample_th)):
            print('Total iters: %d    Current iter: %d' % (math.ceil(test_features.shape[0]/sample_th), i + 1))
            if sample_th*(i + 1) <= test_features.shape[0]:
                acc_rbf_svm.append(nystroem_approx_svm.score(test_features[sample_th*i:sample_th*(i + 1)], test_labels[sample_th*i:sample_th*(i + 1)].ravel()))
            else:
                acc_rbf_svm.append(nystroem_approx_svm.score(test_features[sample_th*i:test_features.shape[0]], test_labels[sample_th*i:test_features.shape[0]].ravel()))
        acc_rbf_svm = np.mean(acc_rbf_svm)
    else:
        nystroem_approx_svm.fit(train_features, train_labels.ravel())
        acc_rbf_svm = nystroem_approx_svm.score(test_features, test_labels.ravel())
    print('RBF SVM Accuracy: %f\n' % acc_rbf_svm)

    return acc_rbf_svm

def main():
    print('File path:')
    print('Challenges: ' + challenge_path)
    print('Responses: ' + response_path)
    np_challenges, np_responses = fetch_CRP(challenge_path, response_path)

    accuracy = np.zeros((4, len(n_samples)))
    for i in range(len(n_samples)):
        start_time = time.time()
        challenges = np_challenges[:n_samples[i], :]
        responses = np_responses[:n_samples[i], :]
        print('Trying to attack with %d samples' % n_samples[i])
        
        train_challenges, test_challenges, train_responses, test_responses = train_test_split(challenges, responses, test_size=0.2, random_state=42)

        if transform == 'direct':
            print("Trying to attack using raw challenges")
        elif transform == 'parity':
            print("Trying to attack using parity vector transformed challenges")
            train_challenges = get_parity_features(train_challenges)
            test_challenges = get_parity_features(test_challenges)

        accuracy[0, i] = linear_SVM_attack(train_challenges, test_challenges, train_responses, test_responses)
        accuracy[1, i] = LR_attack(train_challenges, test_challenges, train_responses, test_responses)
        accuracy[2, i] = rbf_SVM_attack_fast(train_challenges, test_challenges, train_responses, test_responses)
        end_time = time.time()
        print('Running time of current iteration: %ds\n' % (end_time - start_time))
    
    accuracy[3, :] = np.max(accuracy[0:3, :], axis=0)
    np.savetxt('./svm/accuracy_fpuf_svm_basic.txt', accuracy, fmt='%.3f')

if __name__ == "__main__":
    main()