import os
import argparse

def parse():
    parser = argparse.ArgumentParser(description='Frog Sound Classification');

    # General settings
    parser.add_argument('--dataset', required=False, default='finch', choices=['esc10', 'esc50', 'urbansound8k', 'frog', 'finch']);
    parser.add_argument('--data', default='/Users/mmoh0027/Desktop/GD-MONASH/phd/experiments/datasets/', required=False, help='Path to dataset');
    # parser.add_argument('--data', default='/home/md_sep04/experimentss/datasets/', required=False, help='Path to dataset')
    # parser.add_argument('--data', default='/home/mmoh0027/mb20_scratch/experiments/datasets/', required=False, help='Path to dataset')

    #Handling unknown arguments
    p, unknown = parser.parse_known_args();
    # print(unknown);
    for i in unknown:
        if i.startswith('--'):
            parser.add_argument(i, default=unknown[unknown.index(i)+1]);

    opt = parser.parse_args();

    #Leqarning settings
    opt.batchSize = 64;
    opt.weightDecay = 5e-4;
    opt.momentum = 0.9;
    opt.nEpochs = 1000;
    opt.LR = 0.1;
    opt.schedule = [0.3, 0.6, 0.9];
    opt.warmup = 10;

    #Basic Net Settings
    opt.nClasses = 2;
    opt.nCrops = 1;
    opt.nFolds = 5;
    opt.splits = range(1, opt.nFolds + 1);
    opt.fs = 20000;
    opt.inputLength = 13440;

    return opt


def display_info(opt):
    learning = 'standard'

    print('+------------------------------+')
    print('| Black Finch Classification')
    print('+------------------------------+')
    print('| dataset  : {}'.format(opt.dataset))
    print('| nEpochs  : {}'.format(opt.nEpochs))
    print('| LRInit   : {}'.format(opt.LR))
    print('| schedule : {}'.format(opt.schedule))
    print('| warmup   : {}'.format(opt.warmup))
    print('| batchSize: {}'.format(opt.batchSize))
    print('+------------------------------+')
