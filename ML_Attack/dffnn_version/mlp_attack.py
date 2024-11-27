import os
import sys
import time
import math
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:1")
chunksize = 50000000
challenge_path = './crps/challenges.csv'
response_path = ['./crps/responses_apuf_8xorpuf.csv']
transform = 'parity'
# n_samples = [100000000]
n_samples = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 
             25000, 50000, 100000, 200000, 400000, 800000, 1600000, 3200000, 6400000, 12800000, 
             25000000, 50000000, 100000000]
n_epochs = 25
# batch_size = [8192]
batch_size = [8, 16, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024, 2048, 2048, 4096, 4096, 8192]

def fetch_challenge(challenge_path, n_samples, index):
    if not os.path.exists(challenge_path):
        print('The path specified for challenge CSV file does not exist')
        sys.exit()
    if n_samples >= (index + 1)*chunksize:
        np_challenges = pd.read_csv(challenge_path, header=None, skiprows=index*chunksize, nrows=chunksize).values
    else:
        np_challenges = pd.read_csv(challenge_path, header=None, skiprows=index*chunksize, nrows=n_samples%chunksize).values
    np_challenges[np_challenges==0] = -1

    return np_challenges

def fetch_response(response_path, n_samples, index):
    if not os.path.exists(response_path):
        print('The path specified for response CSV file does not exist')
        sys.exit()
    if n_samples >= (index + 1)*chunksize:
        np_responses = pd.read_csv(response_path, header=None, skiprows=index*chunksize, nrows=chunksize).values
    else:
        np_responses = pd.read_csv(response_path, header=None, skiprows=index*chunksize, nrows=n_samples%chunksize).values
    np_responses[np_responses==0] = -1

    return np_responses

def get_parity_features(challenges):
    n_samples, n_length = challenges.shape
    parity_features = np.zeros((n_samples, n_length + 1))
    parity_features[:, 0:1] = np.ones((n_samples, 1))
    for i in range(2, n_length + 2):
        parity_features[:, i - 1:i] = np.prod(challenges[:, 0:i - 1], axis=1).reshape((n_samples, 1))
    
    return parity_features

class CRPDataset(Dataset):
    def __init__(self, challenges, responses):
        self.X = challenges
        self.y = responses.reshape((len(responses), 1))
        self.X[self.X==-1] = 0
        self.y[self.y==-1] = 0
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = torch.nn.Linear(28, 512)
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, 256)
        self.act2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(256, 1)
        self.act3 = torch.nn.Sigmoid()
        
    def forward(self, X):
        X = self.fc1(X)
        X = self.act1(X)
        X = self.fc2(X)
        X = self.act2(X)
        X = self.fc3(X)
        X = self.act3(X)

        return X

def train(train_dl, model, optimizer):
    criterion = torch.nn.BCELoss()
    for epoch in range(n_epochs):
        model.train()
        for i, (data, target) in enumerate(train_dl):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            if i == 0:
                print("\nepoch: {}, iter: {}, loss: {}".format(epoch + 1, i + 1, loss.data), end='', flush=True)
            print("\repoch: {}, iter: {}, loss: {}".format(epoch + 1, i + 1, loss.data), end='', flush=True)
            optimizer.step()

def eval(test_dl, model):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_dl):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            output = output.to('cpu').detach().numpy()
            output = output.round()
            actual = target.to('cpu').numpy()
            actual = actual.reshape((len(actual), 1))
            predictions.append(output)
            actuals.append(actual)
    
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    acc = accuracy_score(actuals, predictions)
    
    return acc

def main():
    print('File path:')
    print('Challenges: ' + challenge_path)
    accuracy = np.zeros((len(response_path), len(n_samples)))
    for k in range(len(response_path)):
        print('Responses: ' + response_path[k])
        for i in range(len(n_samples)):
            start_time = time.time()
            print('Trying to attack with %d samples' % n_samples[i])
            
            model = MLP()
            model.to(device)
            optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
            
            test_challenges_all = []
            test_responses_all = []
            for j in range(math.ceil(n_samples[i]/chunksize)):
                print('\nCurrent chunk [%d/%d]' % (j + 1, (math.ceil(n_samples[i]/chunksize))))
                challenges = fetch_challenge(challenge_path, n_samples[i], j)
                print(challenges.shape)
                responses = fetch_response(response_path[k], n_samples[i], j)
                print(responses.shape)
            
                if transform == 'direct':
                    print("Trying to attack using raw challenges")
                elif transform == 'parity':
                    print("Trying to attack using parity vector transformed challenges")
                    challenges = get_parity_features(challenges)
                
                train_challenges, test_challenges, train_responses, test_responses = train_test_split(challenges, responses, test_size=0.2, random_state=42)
                test_challenges_all.append(test_challenges)
                test_responses_all.append(test_responses)
                
                train_data = CRPDataset(train_challenges, train_responses)
                train_dl = DataLoader(train_data, batch_size=batch_size[i], shuffle=True, pin_memory=True, num_workers=24)
                train(train_dl, model, optimizer)

            test_challenges, test_responses = np.vstack(test_challenges_all), np.vstack(test_responses_all)
            test_data = CRPDataset(test_challenges, test_responses)
            test_dl = DataLoader(test_data, batch_size=batch_size[i], shuffle=False, pin_memory=True, num_workers=8)
            acc_mlp = eval(test_dl, model)
            print('\nMLP Accuracy: %f\n' % acc_mlp)
            accuracy[k, i] = acc_mlp
            end_time = time.time()
            print('Running time of current iteration: %ds\n' % (end_time - start_time))
        
    np.savetxt('./mlp/accuracy_apuf_mlp_8xorpuf.txt', accuracy, fmt='%.3f')


if __name__ == '__main__':
    main()