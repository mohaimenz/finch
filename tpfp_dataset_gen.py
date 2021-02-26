import sys;
import os;
import glob;
import math;
import numpy as np;
import random;
import time;
import torch;
import torch.optim as optim;

import utils as U;
import opts as opts;
import models as models;
import calculator as calc;
import dataset as ds;

#Reproducibility
seed = 42;
random.seed(seed);
np.random.seed(seed);
torch.manual_seed(seed);
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed);
torch.backends.cudnn.deterministic = True;
torch.backends.cudnn.benchmark = False;
###########################################

class Trainer:
    def __init__(self, opt=None):
        self.opt = opt;
        self.trainGen = None;
        self.valGen = None;
        self.testX = None;
        self.testY = None;
        self.bestAcc = 0.0;
        self.bestAccEpoch = 0;
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
        self.predCount = 0;
        self.predBTFSamples = None;
        self.predBTFLabels = None;

    def __validate(self, net, lossFunc):
        net.eval();
        with torch.no_grad():
            y_pred = None;
            for idx in range(math.ceil(len(self.valGen.data)/(self.opt.batchSize//self.opt.nCrops))):
                x, y = self.valGen.__getitem__(idx);
                # print(x.shape);
                # print(y.shape);

                # print(x.shape);
                # print(y.shape);
                self.testX = x.data if self.testX is None else torch.cat((self.testX, x.data));
                self.testY = y.data if self.testY is None else torch.cat((self.testY, y.data));
                scores = net(x);
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));

            # # print(y_target);
            # print(len(self.testX));
            # print(self.testX.shape);
            # print(len(self.testY));
            # print(self.testY.shape);
            # exit();
            acc, loss = self.__compute_accuracy(y_pred, self.testY, lossFunc);
        net.train();
        return acc, loss;

    #Calculating average prediction (10 crops) and final accuracy
    def __compute_accuracy(self, y_pred, y_target, lossFunc):
        with torch.no_grad():
            # #Reshape to shape theme like each sample comtains 10 samples, calculate mean and find theindices that has highest average value for each sample
            # y_pred = (y_pred.reshape(y_pred.shape[0]//self.opt.nCrops, self.opt.nCrops, y_pred.shape[1])).mean(dim=1).argmax(dim=1);
            # y_target = (y_target.reshape(y_target.shape[0]//self.opt.nCrops, self.opt.nCrops, y_target.shape[1])).mean(dim=1).argmax(dim=1);
            # acc = (((y_pred==y_target)*1).float().mean()*100).item();

            y_pred = y_pred.argmax(dim=1);
            # print(y_pred);
            y_target = y_target.argmax(dim=1);
            # print(y_target);

            pred_btf_idx = np.where(y_pred==0)[0];
            self.predBTFSamples = self.testX[pred_btf_idx];
            print('Sample Shape', ': ', self.predBTFSamples.shape);
            self.predBTFLabels = y_target[pred_btf_idx];
            print('Labels Shape', ': ',self.predBTFLabels.shape);
            tp_count = np.count_nonzero(self.predBTFLabels==0);
            print('TP Count', ': ', tp_count);
            fp_count = np.count_nonzero(self.predBTFLabels==1);
            print('FP Count', ': ', fp_count);

            pred_other_idx = np.where(y_pred==1)[0];
            orig_labels = y_target[pred_other_idx];
            tn_count = np.count_nonzero(orig_labels==1);
            print('TN Count', ': ', tn_count);
            fn_count = np.count_nonzero(orig_labels==0);
            print('FN Count', ': ', fn_count);
            # from sklearn.metrics import confusion_matrix;
            # results = confusion_matrix(y_target, y_pred)
            # print(results)
            preds = (y_pred==y_target)*1;
            predCount = preds.sum().item();
            if predCount > self.predCount:
                self.predCount = predCount;
            acc = (preds.float().mean()*100).item();
            loss = lossFunc(y_pred.float().log(), y_target.float()).item();
        return acc, loss;

    def TestModel(self):
        net_config = [8, 64, 32, 64, 64, 128, 128, 256, 256, 512, 512, 50];
        self.opt.nClasses = 2;
        net = models.GetACDNetModelV2(self.opt.inputLength, self.opt.nClasses, self.opt.fs, net_config).to(self.device);
        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');
        dir = os.getcwd();
        net_path = '{}/models/base502_f{}_*.pt'.format(dir, self.opt.split);
        print(net_path)
        file_paths = glob.glob(net_path);
        if len(file_paths)>0 and os.path.isfile(file_paths[0]):
            state = torch.load(file_paths[0], map_location=self.device);
            net.load_state_dict(state);
            print('Model found at: {}'.format(file_paths[0]));
        else:
            print('Model not found');
            exit();

        self.trainGen, self.valGen = ds.setup(self.opt, self.opt.split);
        net.eval();
        val_acc, val_loss = self.__validate(net, lossFunc);

        net.train();
        print('Testing: Acc(top1) {:.2f}%'.format(val_acc));
        # calc.summary(net, ((1, 1, self.opt.inputLength)))


if __name__ == '__main__':
    opt = opts.parse();
    tpfp_btf_dataset = {};
    for s in range(1, 6):
        opt.split = s;
        trainer = Trainer(opt);
        trainer.TestModel();
        tpfp_btf_dataset['fold{}'.format(s)] = {};
        tpfp_btf_dataset['fold{}'.format(s)]['sounds'] = trainer.predBTFSamples;
        tpfp_btf_dataset['fold{}'.format(s)]['labels'] = trainer.predBTFLabels;
        print(trainer.predBTFSamples.shape);
        print(trainer.predBTFLabels.shape);
        print(trainer.predBTFSamples[0].shape)
        exit();

    path = "/Users/mmoh0027/Desktop/GD-Monash/phd/experiments/datasets/finch/btf_tpfp_2khz.npz";
    np.savez(path, **tpfp_btf_dataset);
