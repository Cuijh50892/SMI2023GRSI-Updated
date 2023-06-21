import torch
import torch.optim as optim
import time, os
import numpy as np
from model import ManifoldEncoderS
from model import ManifoldEncoderX
from model import ManifoldEncoderY
from dataloader import get_dataloader
from dataloader import get_pair_data_loader
import json

if __name__ == '__main__':

    modelS = ManifoldEncoderS()
    modelX = ManifoldEncoderX()
    modelY = ManifoldEncoderY()

    pretrainS = 'model/funcmanifold_S_best.pkl'
    pretrainX = 'model/funcmanifold_X_best.pkl'
    pretrainY = 'model/funcmanifold_Y_best.pkl'

    state_dictS = torch.load(pretrainS, map_location='cpu')
    modelS.load_state_dict(state_dictS)

    state_dictX = torch.load(pretrainX, map_location='cpu')
    modelX.load_state_dict(state_dictX)

    state_dictY = torch.load(pretrainY, map_location='cpu')
    modelY.load_state_dict(state_dictY)

    modelS.eval()
    modelX.eval()
    modelY.eval()

    shape_code = torch.from_numpy(np.loadtxt('shapecode.csv', delimiter=",").astype(np.float32))
    tex_code = torch.from_numpy(np.loadtxt('texcode.csv', delimiter=",").astype(np.float32))
    interaction_code = torch.from_numpy(np.loadtxt('codesY.csv', delimiter=",").astype(np.float32))
    interaction_type = np.loadtxt("interactiontype.csv", dtype=np.str, delimiter=",")

    shape_code_out = modelS(shape_code)
    tex_code_out = modelX(tex_code)
    interaction_code_out = modelY(interaction_code)

    coord = 0.5*shape_code_out + 0.5*tex_code_out

    indexdict = {}
    for idx in range(interaction_code.shape[0]):
        dist = torch.dist(coord,interaction_code_out[idx],2)
        indexdict[interaction_type[idx]] = dist.float()

    ls = list(indexdict.items())
    ls.sort(key=lambda x: x[1], reverse=False)

    data = {'I1': ls[0][0], 'I2': ls[1][0],  'I3': ls[2][0], 'I4': ls[3][0], 'I5': ls[4][0]}

    with open('interactioninfo.json', 'w') as f:
        json.dump(data, f)













