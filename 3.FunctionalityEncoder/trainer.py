import torch
import torch.optim as optim
import time, os
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets


class Trainer(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.gpu_mode = args.gpu_mode
        self.verbose = args.verbose
        self.dataset = 'funcmanifold'

        self.modelS = args.modelS
        self.modelX = args.modelX
        self.modelY = args.modelY

        self.train_loaderS = args.train_loaderS
        self.train_loaderX = args.train_loaderX
        self.train_loaderY = args.train_loaderY
        #self.visualize_loaderS = args.visualize_loaderS
        #self.visualize_loaderX = args.visualize_loaderX
        #self.visualize_loaderY = args.visualize_loaderY
        self.train_pair = args.train_pair
        self.batch_size = args.batch_size

        self.weights_S = torch.from_numpy(np.loadtxt('weightsS.csv', delimiter=",").astype(np.float32))
        self.weights_X = torch.from_numpy(np.loadtxt('weightsX.csv', delimiter=",").astype(np.float32))
        self.weights_Y = torch.from_numpy(np.loadtxt('weightsY.csv', delimiter=",").astype(np.float32))
        self.kneighborID_S = torch.from_numpy(np.loadtxt('neighborsS.csv', delimiter=",").astype(np.int)).long()
        self.kneighborID_X = torch.from_numpy(np.loadtxt('neighborsX.csv', delimiter=",").astype(np.int)).long()
        self.kneighborID_Y = torch.from_numpy(np.loadtxt('neighborsY.csv', delimiter=",").astype(np.int)).long()
        #self.colorS = np.loadtxt("colorS.csv", dtype=np.str, delimiter=",")
        #self.colorX = np.loadtxt("colorX.csv", dtype=np.str, delimiter=",")
        #self.colorY = np.loadtxt("colorY.csv", dtype=np.str, delimiter=",")
        #self.visualize_A = torch.from_numpy(np.loadtxt('vA.csv', delimiter=",").astype(np.float32))
        #self.visualize_B = torch.from_numpy(np.loadtxt('vB.csv', delimiter=",").astype(np.float32))
        self.k = 25
        self.w_pair = 0.90
        self.w_unpair = 0.01
        self.w_recA = 0.1
        self.w_recB = 0.1


        self.scale = 100
        self.loss_scale = 100

        if self.gpu_mode:
            self.weights_S = self.weights_S.cuda()
            self.weights_X = self.weights_X.cuda()
            self.weights_Y = self.weights_Y.cuda()
            self.kneighborID_S = self.kneighborID_S.cuda()
            self.kneighborID_X = self.kneighborID_X.cuda()
            self.kneighborID_Y = self.kneighborID_Y.cuda()
            self.modelS = self.modelS.cuda()
            self.modelX = self.modelX.cuda()
            self.modelY = self.modelY.cuda()

        self.optimizerS = args.optimizerS
        self.optimizerX = args.optimizerX
        self.optimizerY = args.optimizerY
        self.schedulerS = args.schedulerS
        self.schedulerX = args.schedulerX
        self.schedulerY = args.schedulerY
        self.scheduler_interval = args.scheduler_interval
        self.snapshot_interval = args.snapshot_interval
        #self.writer = SummaryWriter(log_dir=args.tboard_dir)

        if args.pretrainS != '':
            self._load_pretrainS(args.pretrainS)
            self._load_pretrainX(args.pretrainX)
            self._load_pretrainY(args.pretrainY)

    def train(self):
        self.train_hist = {
            'loss': [],
            'per_epoch_time': [],
            'total_time': []
        }
        best_loss = 1000000000

        self.ini_visualize()


        print('training start!!')
        start_time = time.time()

        self.modelS.train()
        self.modelX.train()
        self.modelY.train()
        for epoch in range(self.epoch):
            self.train_epoch(epoch, self.verbose)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.visualize(epoch)

            if (epoch + 1) % 100 == 0 or epoch == 0:
                res = self.evaluate(epoch + 1)
                if res['loss'] < best_loss:
                    best_loss = res['loss']
                    self._snapshot('best')

            if epoch % self.scheduler_interval == 0:
                self.schedulerS.step()
                self.schedulerX.step()
                self.schedulerY.step()

            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)

        # finish all epoch
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

    def train_epoch(self, epoch, verbose=False):
        epoch_start_time = time.time()
        loss_buf = []

        # S-X
        for iterA, (codeS_input) in enumerate(self.train_loaderS):
            for iterB, (codeX_input) in enumerate(self.train_loaderX):
                for iterC, (pair) in enumerate(self.train_pair):
                    if self.gpu_mode:
                        codeS_input = codeS_input.cuda()
                        codeX_input = codeX_input.cuda()
                        pair = pair.cuda()

                    # forward
                    self.optimizerS.zero_grad()
                    self.optimizerX.zero_grad()
                    codeS_output = self.modelS(codeS_input)
                    codeX_output = self.modelX(codeX_input)

                    loss_pair, loss_unpair, loss_recA, loss_recB = self.get_lossSX(codeS_output, codeX_output, pair, self.batch_size)

                    print('***')
                    print(loss_pair)
                    print(loss_recA)
                    print(loss_unpair)

                    loss = self.w_pair * loss_pair  + self.w_recA * loss_recA + self.w_recB * loss_recB + self.w_unpair * loss_unpair

                    # backward
                    loss.backward()
                    self.optimizerS.step()
                    self.optimizerX.step()

                    loss_buf.append(loss.detach().cpu().numpy())

        # X-Y
        for iterA, (codeX_input) in enumerate(self.train_loaderX):
            for iterB, (codeY_input) in enumerate(self.train_loaderY):
                for iterC, (pair) in enumerate(self.train_pair):
                    if self.gpu_mode:
                        codeX_input = codeX_input.cuda()
                        codeY_input = codeY_input.cuda()
                        pair = pair.cuda()

                    # forward
                    self.optimizerX.zero_grad()
                    self.optimizerY.zero_grad()
                    codeX_output = self.modelX(codeX_input)
                    codeY_output = self.modelY(codeY_input)

                    loss_pair, loss_unpair, loss_recA, loss_recB = self.get_lossXY(codeX_output, codeY_output, pair, self.batch_size)

                    print('***')
                    print(loss_pair)
                    print(loss_recA)
                    print(loss_unpair)

                    loss = self.w_pair * loss_pair  + self.w_recA * loss_recA + self.w_recB * loss_recB + self.w_unpair * loss_unpair

                    # backward
                    loss.backward()
                    self.optimizerX.step()
                    self.optimizerY.step()

                    loss_buf.append(loss.detach().cpu().numpy())


        # Y-S
        for iterA, (codeY_input) in enumerate(self.train_loaderY):
            for iterB, (codeS_input) in enumerate(self.train_loaderS):
                for iterC, (pair) in enumerate(self.train_pair):
                    if self.gpu_mode:
                        codeY_input = codeY_input.cuda()
                        codeS_input = codeS_input.cuda()
                        pair = pair.cuda()

                    # forward
                    self.optimizerY.zero_grad()
                    self.optimizerS.zero_grad()
                    codeY_output = self.modelY(codeY_input)
                    codeS_output = self.modelS(codeS_input)

                    loss_pair, loss_unpair, loss_recA, loss_recB = self.get_lossYS(codeY_output, codeS_output, pair, self.batch_size)

                    print('***')
                    print(loss_pair)
                    print(loss_recA)
                    print(loss_unpair)

                    loss = self.w_pair * loss_pair  + self.w_recA * loss_recA + self.w_recB * loss_recB + self.w_unpair * loss_unpair

                    # backward
                    loss.backward()
                    self.optimizerY.step()
                    self.optimizerS.step()

                    loss_buf.append(loss.detach().cpu().numpy())

        # finish one epoch
        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['loss'].append(np.mean(loss_buf))
        print(f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}, time {epoch_time:.4f}s')


    def evaluate(self, epoch):
        self.modelS.eval()
        self.modelX.eval()
        self.modelY.eval()
        loss_buf = []

        for iterA, (codeS_input) in enumerate(self.train_loaderS):
            for iterB, (codeX_input) in enumerate(self.train_loaderX):
                for iterC, (pair) in enumerate(self.train_pair):
                    if self.gpu_mode:
                        codeS_input = codeS_input.cuda()
                        codeX_input = codeX_input.cuda()
                        pair = pair.cuda()

                    codeS_output = self.modelS(codeS_input)
                    codeX_output = self.modelX(codeX_input)

                    loss_pair, loss_unpair, loss_recA, loss_recB = self.get_lossSX(codeS_output, codeX_output, pair, self.batch_size)
                    loss = self.w_pair * loss_pair  + self.w_recA * loss_recA + self.w_recB * loss_recB + self.w_unpair * loss_unpair
                    loss_buf.append(loss.detach().cpu().numpy())

        for iterA, (codeY_input) in enumerate(self.train_loaderY):
            for iterB, (codeX_input) in enumerate(self.train_loaderX):
                for iterC, (pair) in enumerate(self.train_pair):
                    if self.gpu_mode:
                        codeY_input = codeY_input.cuda()
                        codeX_input = codeX_input.cuda()
                        pair = pair.cuda()

                    codeY_output = self.modelY(codeY_input)
                    codeX_output = self.modelX(codeX_input)

                    loss_pair, loss_unpair, loss_recA, loss_recB = self.get_lossXY(codeX_output, codeY_output, pair, self.batch_size)
                    loss = self.w_pair * loss_pair  + self.w_recA * loss_recA + self.w_recB * loss_recB + self.w_unpair * loss_unpair
                    loss_buf.append(loss.detach().cpu().numpy())

        for iterA, (codeY_input) in enumerate(self.train_loaderY):
            for iterB, (codeS_input) in enumerate(self.train_loaderS):
                for iterC, (pair) in enumerate(self.train_pair):
                    if self.gpu_mode:
                        codeY_input = codeY_input.cuda()
                        codeS_input = codeS_input.cuda()
                        pair = pair.cuda()

                    codeY_output = self.modelY(codeY_input)
                    codeS_output = self.modelS(codeS_input)

                    loss_pair, loss_unpair, loss_recA, loss_recB = self.get_lossYS(codeY_output, codeS_output, pair, self.batch_size)
                    loss = self.w_pair * loss_pair  + self.w_recA * loss_recA + self.w_recB * loss_recB + self.w_unpair * loss_unpair
                    loss_buf.append(loss.detach().cpu().numpy())

        self.modelS.train()
        self.modelX.train()
        self.modelY.train()
        res = {
            'loss': np.mean(loss_buf),
        }
        return res

    def ini_visualize(self):
        print("")

    def visualize(self, epoch):
        print("")

    def _snapshot(self, epoch):
        save_dir = os.path.join(self.save_dir, self.dataset)
        torch.save(self.modelS.state_dict(), save_dir + "_S_" + str(epoch) + '.pkl')
        torch.save(self.modelX.state_dict(), save_dir + "_X_" + str(epoch) + '.pkl')
        torch.save(self.modelY.state_dict(), save_dir + "_Y_" + str(epoch) + '.pkl')
        print(f"Save model to {save_dir}_S_{str(epoch)}.pkl")

    def _load_pretrainS(self, pretrainS):
        state_dictS = torch.load(pretrainS, map_location='cpu')
        self.modelS.load_state_dict(state_dictS)
        print(f"Load model from {pretrainS}.pkl")

    def _load_pretrainX(self, pretrainX):
        state_dictX = torch.load(pretrainX, map_location='cpu')
        self.modelX.load_state_dict(state_dictX)
        print(f"Load model from {pretrainX}.pkl")

    def _load_pretrainY(self, pretrainY):
        state_dictY = torch.load(pretrainY, map_location='cpu')
        self.modelY.load_state_dict(state_dictY)
        print(f"Load model from {pretrainY}.pkl")

    def _get_lr(self, group=0):
        return self.optimizerS.param_groups[group]['lr']

    def get_lossSX(self, outputA, outputB, pair, batch_size):

        num_pair = batch_size
        codeA = outputA[pair[0, 0], :]
        codeB = outputB[pair[0, 1], :]

        weightsA = self.weights_S[pair[0, 0], :]
        w_A = weightsA.reshape((self.k, 1))
        weightsB = self.weights_X[pair[0, 1], :]
        w_B = weightsB.reshape((self.k, 1))

        neighborA = outputA.index_select(0, self.kneighborID_S[pair[0, 0], :])
        neighborB = outputB.index_select(0, self.kneighborID_X[pair[0, 1], :])

        loss_recA = torch.dist(codeA, torch.sum(neighborA * w_A, dim=0), p=2)
        loss_recB = torch.dist(codeB, torch.sum(neighborB * w_B, dim=0), p=2)

        loss_pair = torch.dist(codeA, codeB, p=2)

        for i_pair in range(1, num_pair):
            codeC = outputA[pair[i_pair, 0], :]
            codeD = outputB[pair[i_pair, 1], :]

            weightsC = self.weights_S[pair[i_pair, 0], :]
            w_C = weightsC.reshape((self.k, 1))
            weightsD = self.weights_X[pair[i_pair, 1], :]
            w_D = weightsD.reshape((self.k, 1))

            neighborC = outputA.index_select(0, self.kneighborID_S[pair[i_pair, 0], :])
            neighborD = outputB.index_select(0, self.kneighborID_X[pair[i_pair, 1], :])

            recC = torch.dist(codeC, torch.sum(neighborC * w_C, dim=0), p=2)
            recD = torch.dist(codeD, torch.sum(neighborD * w_D, dim=0), p=2)
            loss_recA = loss_recA + recC
            loss_recB = loss_recB + recD
            loss_pair = loss_pair + torch.dist(codeC,codeD,p=2)

        loss_pair = loss_pair / batch_size

        loss_recA = loss_recA / batch_size
        loss_recB = loss_recB / batch_size

        code_all = torch.cat((outputA, outputB), 0)
        loss_unpair = self.mat_euclidean_dist(code_all,code_all)
        loss_unpair = torch.mean(loss_unpair)

        return  loss_pair, loss_unpair, loss_recA, loss_recB

    def get_lossXY(self, outputA, outputB, pair, batch_size):

        num_pair = batch_size
        codeA = outputA[pair[0, 1], :]
        codeB = outputB[pair[0, 2], :]

        weightsA = self.weights_X[pair[0, 1], :]
        w_A = weightsA.reshape((self.k, 1))
        weightsB = self.weights_Y[pair[0, 2], :]
        w_B = weightsB.reshape((self.k, 1))

        neighborA = outputA.index_select(0, self.kneighborID_X[pair[0, 1], :])
        neighborB = outputB.index_select(0, self.kneighborID_Y[pair[0, 2], :])

        loss_recA = torch.dist(codeA, torch.sum(neighborA * w_A, dim=0), p=2)
        loss_recB = torch.dist(codeB, torch.sum(neighborB * w_B, dim=0), p=2)

        loss_pair = torch.dist(codeA, codeB, p=2)

        for i_pair in range(1, num_pair):
            codeC = outputA[pair[i_pair, 1], :]
            codeD = outputB[pair[i_pair, 2], :]

            weightsC = self.weights_X[pair[i_pair, 1], :]
            w_C = weightsC.reshape((self.k, 1))
            weightsD = self.weights_Y[pair[i_pair, 2], :]
            w_D = weightsD.reshape((self.k, 1))

            neighborC = outputA.index_select(0, self.kneighborID_X[pair[i_pair, 1], :])
            neighborD = outputB.index_select(0, self.kneighborID_Y[pair[i_pair, 2], :])

            recC = torch.dist(codeC, torch.sum(neighborC * w_C, dim=0), p=2)
            recD = torch.dist(codeD, torch.sum(neighborD * w_D, dim=0), p=2)
            loss_recA = loss_recA + recC
            loss_recB = loss_recB + recD
            loss_pair = loss_pair + torch.dist(codeC,codeD,p=2)

        loss_pair = loss_pair / batch_size

        loss_recA = loss_recA / batch_size
        loss_recB = loss_recB / batch_size

        code_all = torch.cat((outputA, outputB), 0)
        loss_unpair = self.mat_euclidean_dist(code_all,code_all)
        loss_unpair = torch.mean(loss_unpair)

        return  loss_pair, loss_unpair, loss_recA, loss_recB

    def get_lossYS(self, outputA, outputB, pair, batch_size):

        num_pair = batch_size
        codeA = outputA[pair[0, 2], :]
        codeB = outputB[pair[0, 0], :]

        weightsA = self.weights_Y[pair[0, 2], :]
        w_A = weightsA.reshape((self.k, 1))
        weightsB = self.weights_S[pair[0, 0], :]
        w_B = weightsB.reshape((self.k, 1))

        neighborA = outputA.index_select(0, self.kneighborID_Y[pair[0, 2], :])
        neighborB = outputB.index_select(0, self.kneighborID_S[pair[0, 0], :])

        loss_recA = torch.dist(codeA, torch.sum(neighborA * w_A, dim=0), p=2)
        loss_recB = torch.dist(codeB, torch.sum(neighborB * w_B, dim=0), p=2)

        loss_pair = torch.dist(codeA, codeB, p=2)

        for i_pair in range(1, num_pair):
            codeC = outputA[pair[i_pair, 2], :]
            codeD = outputB[pair[i_pair, 0], :]

            weightsC = self.weights_Y[pair[i_pair, 2], :]
            w_C = weightsC.reshape((self.k, 1))
            weightsD = self.weights_S[pair[i_pair, 0], :]
            w_D = weightsD.reshape((self.k, 1))

            neighborC = outputA.index_select(0, self.kneighborID_Y[pair[i_pair, 2], :])
            neighborD = outputB.index_select(0, self.kneighborID_S[pair[i_pair, 0], :])

            recC = torch.dist(codeC, torch.sum(neighborC * w_C, dim=0), p=2)
            recD = torch.dist(codeD, torch.sum(neighborD * w_D, dim=0), p=2)
            loss_recA = loss_recA + recC
            loss_recB = loss_recB + recD
            loss_pair = loss_pair + torch.dist(codeC,codeD,p=2)

        loss_pair = loss_pair / batch_size

        loss_recA = loss_recA / batch_size
        loss_recB = loss_recB / batch_size

        code_all = torch.cat((outputA, outputB), 0)
        loss_unpair = self.mat_euclidean_dist(code_all,code_all)
        loss_unpair = torch.mean(loss_unpair)

        return  loss_pair, loss_unpair, loss_recA, loss_recB

    def mat_euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist


