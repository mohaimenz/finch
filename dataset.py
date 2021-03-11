import os;
import numpy as np;
import random;
import utils as U;
import torch;

class Generator():
    #Generates data for Keras
    def __init__(self, samples, labels, options, train=True):
        random.seed(42);
        #Initialization
        self.data = [(samples[i], labels[i]) for i in range (0, len(samples))];
        self.opt = options;
        self.batch_size = self.opt.batchSize if train == True else self.opt.batchSize//self.opt.nCrops;
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
        self.train = train;
        self.preprocess_funcs = self.preprocess_setup();

    def __len__(self):
        #Denotes the number of batches per epoch
        return int(np.floor(len(self.data) / self.batch_size));
        #return len(self.samples);

    def __getitem__(self, batchIndex):
        #Generate one batch of data
        batchX, batchY = self.generate_batch(batchIndex);
        batchX = np.expand_dims(batchX, axis=1)
        batchX = np.expand_dims(batchX, axis=3)
        return torch.tensor(np.moveaxis(batchX, 3, 1)).to(self.device), torch.tensor(batchY).to(self.device);

    def generate_batch(self, batchIndex):
        #Generates data containing batch_size samples
        sounds = [];
        labels = [];
        indexes = None;
        for i in range(self.batch_size):
            if self.train or self.opt.nCrops==1:
                if indexes == None:
                    indexes = self.data[batchIndex*self.batch_size:(batchIndex+1)*self.batch_size];
                else:
                    if i >= len(indexes):
                        break;

                sound, target = indexes[i];

                sound = sound[0:self.opt.inputLength];
                sound =  sound.astype(np.float32)

                label = np.zeros(self.opt.nClasses);
                label[target] = 1;
            else:
                sound, target = self.data[i];
                sound = self.preprocess(sound).astype(np.float32)
                label = np.zeros((self.opt.nCrops, self.opt.nClasses));
                label[:,target] = 1;

            sounds.append(sound);
            labels.append(label);


        sounds = np.asarray(sounds);
        labels = np.asarray(labels);
        if self.train == False and self.opt.nCrops>1:
            sounds = sounds.reshape(sounds.shape[0]*sounds.shape[1], sounds.shape[2]);
            labels = labels.reshape(labels.shape[0]*labels.shape[1], labels.shape[2]);

        return sounds, labels;

    def preprocess_setup(self):
        funcs = []
        funcs += [U.padding(self.opt.inputLength // 2),
                  U.normalize(32768.0),
                  U.multi_crop(self.opt.inputLength, self.opt.nCrops)]

        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound;

def setup(opt, split):
    # print(os.path.join(opt.data, opt.dataset, 'wav{}_norm_fixed.npz'.format(opt.fs // 1000)));
    dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}{}.npz'.format(opt.fs // 1000, opt.step_name)), allow_pickle=True)
    # Split to train and val
    train_sounds = []
    train_labels = []
    val_sounds = []
    val_labels = []
    for i in range(1, opt.nFolds + 1):
        sounds = dataset['fold{}'.format(i)].item()['sounds']
        labels = dataset['fold{}'.format(i)].item()['labels']
        if i == split:
            val_sounds.extend(sounds)
            val_labels.extend(labels)
        else:
            train_sounds.extend(sounds)
            train_labels.extend(labels)

    # Iterator setup
    train_data = Generator(train_sounds, train_labels, opt, train=True)
    val_data = Generator(val_sounds, val_labels, opt, train=False)

    return train_data, val_data
