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
    def __init__(self, opt=None, split=0):
        self.opt = opt;
        self.split = split;
        self.trainGen = None;
        self.valGen = None;
        self.testX = None;
        self.testY = None;
        self.bestAcc = 0.0;
        self.bestAccEpoch = 0;
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
        self.predCount = 0;

    def Train(self):
        train_start_time = time.time();
        # print(self.device);
        print('Starting ACDNet model training for SPLIT-{}'.format(self.split));
        # Just trying to build the base network with 50 filter in last conv but 2 neurons in the dense layer
        acdnetv220_conf = [8, 64, 32, 64, 64, 128, 128, 256, 256, 512, 512, 50];
        net = models.GetACDNetModelV2(self.opt.inputLength, self.opt.nClasses, self.opt.fs, acdnetv220_conf).to(self.device);
        calc.summary(net, (1,1,self.opt.inputLength));
        # exit();
        self.trainGen, self.valGen = ds.setup(self.opt, self.split);
        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');
        optimizer = optim.SGD(net.parameters(), lr=self.opt.LR, weight_decay=self.opt.weightDecay, momentum=self.opt.momentum, nesterov=True);

        for epochIdx in range(self.opt.nEpochs):
            epoch_start_time = time.time();
            optimizer.param_groups[0]['lr'] = self.__get_lr(epochIdx+1);
            cur_lr = optimizer.param_groups[0]['lr'];
            running_loss = 0.0;
            running_acc = 0.0;
            n_batches = math.ceil(len(self.trainGen.data)/self.opt.batchSize);
            for batchIdx in range(n_batches):
                # with torch.no_grad():
                x,y = self.trainGen.__getitem__(batchIdx);
                # zero the parameter gradients
                optimizer.zero_grad();

                # forward + backward + optimize
                outputs = net(x);
                running_acc += (((outputs.data.argmax(dim=1) == y.argmax(dim=1))*1).float().mean()).item();
                loss = lossFunc(outputs.log(), y.float());
                loss.backward();
                optimizer.step();

                running_loss += loss.item();
                # break;

            tr_acc = (running_acc / n_batches)*100;
            tr_loss = running_loss / n_batches;

            #Epoch wise validation Validation
            epoch_train_time = time.time() - epoch_start_time;

            net.eval();
            val_acc, val_loss = self.__validate(net, lossFunc);
            #Save best model
            self.__save_model(val_acc, epochIdx, net);
            self.__on_epoch_end(epoch_start_time, epoch_train_time, epochIdx, cur_lr, tr_loss, tr_acc, val_loss, val_acc);

            running_loss = 0;
            running_acc = 0;
            net.train();
            # break;

        total_time_taken = time.time() - train_start_time;
        print("Execution finished in: {}".format(U.to_hms(total_time_taken)));

    def __get_lr(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule]);
        decay = sum(epoch > divide_epoch);
        if epoch <= self.opt.warmup:
            decay = 1;
        return self.opt.LR * np.power(0.1, decay);

    def __validate(self, net, lossFunc):
        net.eval();
        with torch.no_grad():
            y_target = None;
            y_pred = None;
            for idx in range(math.ceil(len(self.valGen.data)/self.opt.batchSize)):
                x, y = self.valGen.__getitem__(idx);
                # print(x.shape);
                y_target = y.data if y_target is None else torch.cat((y_target, y.data));
                scores = net(x);
                # print(scores.data);
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));

            # print(y_target);
            acc, loss = self.__compute_accuracy(y_pred, y_target, lossFunc);
        net.train();
        return acc, loss;

    #Calculating average prediction (10 crops) and final accuracy
    def __compute_accuracy(self, y_pred, y_target, lossFunc):
        with torch.no_grad():
            #Reshape to shape theme like each sample comtains 10 samples, calculate mean and find theindices that has highest average value for each sample
            y_pred = y_pred.argmax(dim=1);
            y_target = y_target.argmax(dim=1);
            # from sklearn.metrics import confusion_matrix;
            # results = confusion_matrix(y_target, y_pred)
            # print(results)
            preds = (y_pred==y_target)*1;
            predCount = preds.sum().item();
            if predCount > self.predCount:
                # print(predCount);
                self.predCount = predCount;
            acc = (preds.float().mean()*100).item();
            # valLossFunc = torch.nn.KLDivLoss();
            loss = lossFunc(y_pred.float().log(), y_target.float()).item();
            # loss = 0.0;
        return acc, loss;

    def __on_epoch_end(self, start_time, train_time, epochIdx, lr, tr_loss, tr_acc, val_loss, val_acc):
        epoch_time = time.time() - start_time;
        val_time = epoch_time - train_time;
        line = 'SP-{} Epoch: {}/{} | Time: {} (Train {}  Val {}) | Train: LR {}  Acc {:.2f}% | Val: Loss {:.2f}  Acc(top1) {:.2f}% | PC:{} | HA {:.2f}@{}\n'.format(
            self.split, epochIdx+1, self.opt.nEpochs, U.to_hms(epoch_time), U.to_hms(train_time), U.to_hms(val_time),
            lr, tr_acc, val_loss, val_acc, self.predCount, self.bestAcc, self.bestAccEpoch);
        # print(line)
        sys.stdout.write(line);
        sys.stdout.flush();

    def __save_model(self, acc, epochIdx, net):
        if acc > self.bestAcc:
            dir = os.getcwd();
            fname = "{}/models/final502_f{}_{:.2f}_{}.pt";
            old_model = fname.format(dir, self.split, self.bestAcc, self.bestAccEpoch);
            if os.path.isfile(old_model):
                os.remove(old_model);
            self.bestAcc = acc;
            self.bestAccEpoch = epochIdx +1;
            torch.save(net.state_dict(), fname.format(dir, self.split, self.bestAcc, self.bestAccEpoch));

    def TestModel(self):
        acdnetv220_conf = [8, 64, 32, 64, 64, 128, 128, 256, 256, 512, 512, 50];
        net = models.GetACDNetModelV2(self.opt.inputLength, self.opt.nClasses, self.opt.fs, acdnetv220_conf).to(self.device);
        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');
        dir = os.getcwd();
        net_path = '{}/models/base502_f1_87.72_693.pt'.format(dir);
        print(net_path)
        file_paths = glob.glob(net_path);
        if len(file_paths)>0 and os.path.isfile(file_paths[0]):
            state = torch.load(file_paths[0], map_location=self.device);
            net.load_state_dict(state);
            print('Model found at: {}'.format(file_paths[0]));
        else:
            print('Model not found');
            exit();

        self.trainGen, self.valGen = ds.setup(self.opt, self.split);
        net.eval();
        val_acc, val_loss = self.__validate(net, lossFunc);

        net.train();
        print('Testing: Acc(top1) {:.2f}%'.format(val_acc));
        calc.summary(net, ((1, 1, self.opt.inputLength)))
        exit();

if __name__ == '__main__':
    opt = opts.parse();
    opt.step_name = '_seq2'; #seq2 for the final step or '' for the first step;
    for split in opt.splits:
        print('+-- Split {} --+'.format(split));
        opts.display_info(opt);
        trainer = Trainer(opt, 4);
        trainer.Train();
