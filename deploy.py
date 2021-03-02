import sys;
import os;
import glob;
import math;
import numpy as np;
import random;
import time;
import torch;

import opts as opts;
import models as models;
import calculator as calc;

import librosa as lb;

class Tester:
    def __init__(self, opt=None):
        self.opt = opt;
        self.slice_length = 0.672;
        self.stride = 0.0672;
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
        self.btf_indexes = [];
        self.btf_sequences = [];
        self.btf_probables = [];

    def __add_to_probables(self, idx):
        self.btf_indexes.append(idx);
        if len(self.btf_probables)>0:
            if idx == self.btf_probables[-1]+1:
                self.btf_probables.append(idx);
            else:
                if len(self.btf_probables) >= 3:
                    self.btf_sequences.append(self.btf_probables);
                self.btf_probables = [idx];
        else:
            self.btf_probables.append(idx);

    def __flatten_list(self, list):
        from itertools import chain;
        flat_list = list(chain.from_iterable(list));
        return flat_list;

    def __write_to_file(self):
        import pickle
        with open('recognitions/{}_dump'.format(self.opt.file_name), 'wb') as fp:
            pickle.dump(self.btf_indexes, fp);
        with open('recognitions/{}'.format(self.opt.file_name), 'wb') as fp:
            pickle.dump(self.btf_sequences, fp);
        print("Completed writing into files");

    def __read_list_from_file(self, file):
        import pickle;
        with open (file, 'rb') as fp:
            list = pickle.load(fp);
        return list;

    def __prepare_sample(self, sample):
        sample = sample.astype(np.float32);
        if len(sample) >= self.opt.inputLength:
            sample = sample[0:self.opt.inputLength];
        else:
            sample = np.pad(sample, (0, self.opt.inputLength - len(sample)), 'constant');

        sample = sample / 32768.0;
        # add no of samples axis
        sample = np.expand_dims(sample, axis=0);
        # add height axis
        sample = np.expand_dims(sample, axis=1);
        # add channels axis
        sample = np.expand_dims(sample, axis=3);
        # move channel axis to make it like (batch, channels, height, width) (i.e 1,1,1,13440)
        sample = np.moveaxis(sample, 3, 1);

        # sample = torch.tensor(np.moveaxis(sample, 3, 1));
        # print(sample);
        return torch.tensor(sample).to(self.device);
        # exit();
        # return torch.tensor(np.moveaxis(sample, 3, 1)).to(self.device);

    def __get_slice(self, index):
        start = index * self.stride;
        sample = lb.load(path = '{}{}.wav'.format(self.opt.record_path, self.opt.file_name), sr = self.opt.fs, offset= start, duration = self.slice_length);
        return self.__prepare_sample(sample[0]);

    def __test(self, net):
        net.eval();
        with torch.no_grad():
            continue_test = True;
            counter = 0;
            while continue_test is True:
                try:
                    x = self.__get_slice(counter);
                    scores = net(x);
                    is_btf = True if scores[0][0] > scores[0][1] else False;
                    if is_btf is True:
                        # print('Index: {}, BTF: {}'.format(counter, is_btf));
                        self.__add_to_probables(counter);
                    counter += 1;
                except:
                    e = sys.exc_info()[0];
                    print(e);
                    continue_test = False;
        if len(self.btf_indexes) > 0:
            self.__write_to_file();

    def TestModel(self):
        acdnetv220_conf = [8, 64, 32, 64, 64, 128, 128, 256, 256, 512, 512, 50];
        net = models.GetACDNetModelV2(self.opt.inputLength, self.opt.nClasses, self.opt.fs, acdnetv220_conf).to(self.device);
        dir = os.getcwd();
        net_path = '{}/models/base502_f1_87.72_693.pt'.format(dir);
        file_paths = glob.glob(net_path);
        if len(file_paths)>0 and os.path.isfile(file_paths[0]):
            state = torch.load(file_paths[0], map_location=self.device);
            net.load_state_dict(state);
            print('Model found at: {}'.format(file_paths[0]));
        else:
            print('Model not found');
            exit();

        net.eval();
        self.__test(net);
        # exit();

if __name__ == '__main__':
    import time;
    opt = opts.parse();
    # 1. 3_20200918_073000_Rec [-19.4619 146.7124].wav
    # 2. 7_20200917_073000_Rec [-19.4613 146.7108].wav

    # opt.record_path = '/Users/mmoh0027/Desktop/Finch/OriginalRecordings/';
    # opt.label_path = '/Users/mmoh0027/Desktop/Finch/LabelledData/';
    c = 0;
    for txt_file_path in sorted(glob.glob(os.path.join(opt.label_path, '*.txt'))):
        start = time.time();
        c += 1
        # opt.file_name = os.path.split(txt_file_path)[1].split('.Table')[0];
        opt.file_name = '3_20200926_073000_Rec [-19.4619 146.7124]'
        print('{} Processing: {}'.format(c, opt.file_name))
        tester = Tester(opt);
        tester.TestModel();
        end = time.time();
        print('File {} - Time required to process: {}'.format(c, time.strftime("%H:%M:%S", time.gmtime(end-start))));
        break;
    print('Finished Processing {} Files'.format(c))
