import numpy as np
import os
from data import save_weights, fine_tune
def load_weights_if_available(name):
    path = f'data/finetune/{name}.npy'
    if not os.path.exists(path):
        print('존재하지 않는 사용자입니다. 미세 조정을 시작합니다')
        fine_tune(name)
        save_weights(name)
        [w1,w2,w3] = np.load(path)
    else:
        [w1,w2,w3] = np.load(path)

    return w1,w2,w3


if __name__ == '__main__':
    load_weights_if_available('HSK')