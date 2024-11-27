from pypuf.simulation import XORArbiterPUF
from pypuf.io import ChallengeResponseSet, random_inputs
from pypuf.metrics import similarity

from lr2021 import LRAttack2021

import pandas as pd
import numpy as np

def fpuf_eval(fpuf, challenges):
    sum = np.matmul(fpuf, challenges.T)
    sum[sum>0] = 1
    sum[sum<0] = -1
    responses = np.prod(sum, axis=0)

    return responses

def lr_ml_attack(puf_type, n, k, N, batch_size):
    print('Info: PUF=' + puf_type + '  n=' + str(n) + '  k=' + str(k), ' N=' + str(N) + '  batchsize=' + str(batch_size))

    if puf_type == 'apuf':
        apuf = XORArbiterPUF(n=n, k=k, seed=1)
        crps = ChallengeResponseSet.from_simulation(apuf, N=N, seed=2)
    elif puf_type == 'fpuf':
        df_fpuf = pd.read_csv('./fpuf_state.csv', header=None, index_col=False)
        fpuf = np.array(df_fpuf.replace(0, -1))
        fpuf = fpuf.ravel()[:n*k].reshape((k, n))
        challenges = random_inputs(n=n, N=N, seed=2)
        responses = fpuf_eval(fpuf, challenges)
        crps = ChallengeResponseSet(challenges, responses.astype(np.float64))
    
    # np.savetxt('challenges.csv', crps.challenges, fmt='%d', delimiter=',')
    # np.savetxt('responses_' + puf_type + '_8xorpuf.csv', crps.responses.ravel(), fmt='%d', delimiter=',')

    attack = LRAttack2021(crps, seed=3, k=k, bs=batch_size, lr=0.001, epochs=100)
    attack.fit()
    model = attack._model
    
    if puf_type == 'apuf':
        acc = similarity(apuf, model, seed=4)
    elif puf_type == 'fpuf':
        inputs = random_inputs(n=n, N=1000, seed=4)
        responses_fpuf = fpuf_eval(fpuf, inputs)
        responses_model = model.eval(inputs)
        acc = np.sum(responses_fpuf == responses_model)/1000

    print('Logistic Regression Accuracy: %f\n' % acc)
    return acc
    
def main():
    lr_ml_attack('apuf', 27, 8, 3200000, 1024)

    # puf_type = 'apuf'
    # n = 27
    # k = [1, 2, 4, 6, 8]
    # N = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 
    #      25000, 50000, 100000, 200000, 400000, 800000, 1600000, 3200000, 6400000, 12800000, 
    #      25000000, 50000000, 100000000]
    # batch_size = [1, 8, 16, 16, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024, 1024, 2048, 2048, 2048, 4096]

    # accuracy = np.zeros((len(k), len(N)))
    # for idx_xorpuf in range(0, len(k)):
    #     for idx_sample in range(0, len(N)):
    #         accuracy[idx_xorpuf, idx_sample] = lr_ml_attack(puf_type, n, k[idx_xorpuf], N[idx_sample], batch_size[idx_sample])
    #     print('Progress finished ' + str(idx_xorpuf + 1) + '/' + str(len(k)))
    # np.savetxt('accuracy_' + puf_type + '_lr.txt', accuracy, fmt='%.3f')

if __name__ == "__main__":
    main()