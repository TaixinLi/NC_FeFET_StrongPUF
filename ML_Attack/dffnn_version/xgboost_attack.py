import os
import sys
import time
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import xgboost as xgb

challenge_path = './crps/challenges.csv'
response_path = ['./crps/responses_apuf_basic.csv', './crps/responses_apuf_2xorpuf.csv', './crps/responses_apuf_4xorpuf.csv', 
                 './crps/responses_apuf_6xorpuf.csv', './crps/responses_apuf_8xorpuf.csv']
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

def main():
    print('File path:')
    print('Challenges: ' + challenge_path)

    accuracy = np.zeros((len(response_path), len(n_samples)))
    for k in range(len(response_path)):
        print('Responses: ' + response_path[k])
        np_challenges, np_responses = fetch_CRP(challenge_path, response_path[k])
        for i in range(len(n_samples)):
            start_time = time.time()
            challenges = np_challenges[:n_samples[i], :]
            responses = np_responses[:n_samples[i], :]
            print('Trying to attack with %d samples' % n_samples[i])
            
            if transform == 'direct':
                print("Trying to attack using raw challenges")
            elif transform == 'parity':
                print("Trying to attack using parity vector transformed challenges\n")
                challenges = get_parity_features(challenges)
            
            challenges[challenges==-1] = 0
            responses[responses==-1] = 0
            train_challenges, test_challenges, train_responses, test_responses = train_test_split(challenges, responses, test_size=0.2, random_state=42)
            dtrain = xgb.DMatrix(train_challenges, train_responses)
            dtest = xgb.DMatrix(test_challenges, test_responses)
            
            params = {'device': 'cuda:3',
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'learning_rate': 0.025,
                    'max_depth': 9,
                    'min_child_weight': 1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                    }
            model = xgb.train(params, dtrain, num_boost_round=2500, early_stopping_rounds=50, evals=[(dtrain, 'train')])
            ypred = model.predict(dtest)
            ypred = ypred.round()
            acc_xgboost = accuracy_score(test_responses, ypred)
            print('\nXGBoost Accuracy: %f\n' % acc_xgboost)

            accuracy[k, i] = acc_xgboost
            end_time = time.time()
            print('Running time of current iteration: %ds\n' % (end_time - start_time))
    
    np.savetxt('./xgboost/accuracy_apuf_xgboost.txt', accuracy, fmt='%.3f')

if __name__ == "__main__":
    main()