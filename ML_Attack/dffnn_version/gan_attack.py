import os
import sys
import time
import math
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:2")
chunksize = 50000000
challenge_path = './crps/challenges.csv'
response_path = ['./crps/responses_apuf_basic.csv']
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
    def __init__(self, crps):
        self.X = crps
        self.X = self.X.astype('float32')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

class Generator(nn.Module):
    def __init__(self, n_features):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(50, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, n_features),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self, n_features):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, input):
        return self.model(input).view(-1, 1).squeeze(1)

def train(dataloader, netG, netD, optimizerG, optimizerD):
    for epoch in range(n_epochs):
        for i, data in enumerate(dataloader):
            netD.zero_grad()
            real = data.to(device)
            noise = torch.randn(real.size(0), 50, device=device)
            fake = netG(noise)
            loss_D = -torch.mean(netD(real)) + torch.mean(netD(fake))
            loss_D.backward()
            optimizerD.step()

            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

            if i % 2 == 0:
                netG.zero_grad()
                noise = torch.randn(real.size(0), 50, device=device)
                fake = netG(noise)
                loss_G = -torch.mean(netD(fake))
                loss_G.backward()
                optimizerG.step()

            if i == 0:
                print('[%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch + 1, n_epochs, loss_D.item(), loss_G.item()))

def eval(dataloader, netG, netD):
    real_label = 1
    fake_label = 0
    predictions, actuals = [], []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            real = data.to(device)
            label = torch.full((real.size(0), ), real_label, dtype=real.dtype, device=device)
            output = torch.sigmoid(netD(real))
            output = output.to('cpu').detach().numpy()
            output = output.round()
            output = output.reshape((len(output), 1))
            actual = label.to('cpu').numpy()
            actual = actual.reshape((len(actual), 1))
            predictions.append(output)
            actuals.append(actual)

            noise = torch.randn(real.size(0), 50, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = torch.sigmoid(netD(fake))
            output = output.to('cpu').detach().numpy()
            output = output.round()
            output = output.reshape((len(output), 1))
            actual = label.to('cpu').numpy()
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

            netG = Generator(29).to(device)
            netD = Discriminator(29).to(device)
            optimizerG = optim.RMSprop(netG.parameters(), lr=0.00005)
            optimizerD = optim.RMSprop(netD.parameters(), lr=0.00005)

            test_crps_all = []
            for j in range(math.ceil(n_samples[i]/chunksize)):
                print('\nCurrent chunk [%d/%d]' % (j + 1, (math.ceil(n_samples[i]/chunksize))))
                challenges = fetch_challenge(challenge_path, n_samples[i], j)
                print(challenges.shape)
                responses = fetch_response(response_path[k], n_samples[i], j)
                print(responses.shape)

                if transform == 'direct':
                    print("Trying to attack using raw challenges")
                elif transform == 'parity':
                    print("Trying to attack using parity vector transformed challenges\n")
                    challenges = get_parity_features(challenges)

                challenges[challenges==-1] = 0
                responses[responses==-1] = 0
                crps = np.concatenate((challenges, responses), axis=1)
                train_crps, test_crps = train_test_split(crps, test_size=0.2, random_state=42)
                test_crps_all.append(test_crps)
                
                train_data = CRPDataset(train_crps)
                train_dl = DataLoader(train_data, batch_size=batch_size[i], shuffle=True, pin_memory=True, num_workers=24)
                train(train_dl, netG, netD, optimizerG, optimizerD)
            
            test_crps = np.vstack(test_crps_all)
            test_data = CRPDataset(test_crps)
            test_dl = DataLoader(test_data, batch_size=batch_size[i], shuffle=False, pin_memory=True, num_workers=8)
            acc_gan = eval(test_dl, netG, netD)
            print('\nGAN Accuracy: %f\n' % acc_gan)
            accuracy[k, i] = acc_gan
            end_time = time.time()
            print('Running time of current iteration: %ds\n' % (end_time - start_time))
    
    np.savetxt('./gan/accuracy_apuf_gan.txt', accuracy, fmt='%.3f')


if __name__ == '__main__':
    main()