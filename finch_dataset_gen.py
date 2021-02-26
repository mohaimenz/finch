import sys;
import os;

import glob;
import numpy as np;
import librosa as lb;
import soundfile as sf;
import pandas as pd;
import random as rand;
import math;
from sklearn.utils import shuffle;

def GetBackgroundDistribution(n_files, count):
    bg = np.array(np.zeros(n_files)).astype(int);
    i = 0;
    while i < 3000:
        idx = rand.randint(0, 49-1);
        bg[idx] += 1;
        i += 1;
    return bg;

def prepare_sample(sample, input_length):
    if len(sample) >= input_length:
        sample = sample[0:input_length];
    else:
        sample = np.pad(sample, (0, input_length - len(sample)), 'constant');
    return sample;

if __name__ == '__main__':
    bg_dist = GetBackgroundDistribution(49, 3000);
    mainDir = os.path.dirname(os.path.abspath('.'));
    finch_path = os.path.join(mainDir, 'datasets/finch/');
    label_path = finch_path+'btf_labels/';
    audio_path = "/Users/mmoh0027/Desktop/Finch/OriginalRecordings/{}.wav";
    input_length = 13440; #0.672 * 20000
    btf_samples = [];
    other_samples = [];
    resample_at = 20000;
    slice_length = 0.672;
    i = 0;
    for txt_file_path in sorted(glob.glob(os.path.join(label_path, '*.txt'))):
        df = pd.read_csv(txt_file_path, sep='\t', usecols=[3,4,5,6], header=0, names=['start', 'end', 'low', 'high']);
        df.dropna(axis=0, how='any', inplace=True);

        # print(file);
        file_name = os.path.split(txt_file_path)[1].split('.Table')[0];
        # print(name.split('.Table')[0]);
        # print(file_name);
        f = sf.SoundFile(audio_path.format(file_name));
        duration = len(f)/f.samplerate;
        full_record_slices = np.arange(0, duration, slice_length);
        # print(len(slices));

        if len(df) > 0:
            for idx, r in df.iterrows():
                sound = lb.load(path = audio_path.format(file_name), sr = resample_at, offset= r['start'], duration = slice_length);
                btf_samples.append(prepare_sample(sound[0], input_length));

                #Removing this slice location from the full record slices
                rms = math.floor(r['start']);
                rme = math.ceil(r['end']);
                full_record_slices = [s for s in full_record_slices if s<rms or s>rme];

        bg_count = bg_dist[i];
        bg_slices = rand.sample(list(full_record_slices), bg_count);
        for bs in bg_slices:
            sound = lb.load(path = audio_path.format(file_name), sr = resample_at, offset= bs, duration = slice_length);
            other_samples.append(prepare_sample(sound[0], input_length));

        i += 1;

    print(len(btf_samples));
    print(btf_samples[0]);
    print(len(other_samples));
    print(other_samples[0]);

    #shuffle the Samples
    btf_samples = shuffle(btf_samples, random_state=0);
    other_samples = shuffle(other_samples, random_state=0);

    #normalize the samples with 32768.0
    btf_samples = np.array(btf_samples) / 32768.0;
    other_samples = np.array(other_samples) / 32768.0;
    # print(btf_samples[0]);
    # exit();


    #Create folds
    btf_per_fold = len(btf_samples)//5;
    btf_folds = [0, btf_per_fold, btf_per_fold*2, btf_per_fold*3, btf_per_fold*4];
    print(btf_folds);

    others_per_fold = len(other_samples)//5;
    others_folds = [0, others_per_fold, others_per_fold*2, others_per_fold*3, others_per_fold*4];
    print(others_folds);

    finch_dataset = {};
    for fold in range(1,6):
        finch_dataset['fold{}'.format(fold)] = {};
        finch_sounds = [];
        finch_labels = [];

        #BTF
        start_index = btf_folds[fold-1];
        end_index = len(btf_samples) if fold==5 else btf_folds[fold];
        finch_sounds.extend(btf_samples[start_index:end_index]);
        print(len(finch_sounds))
        finch_labels.extend([0] * (end_index-start_index));
        print(len(finch_labels))
        print('BTF Indexes: {} - {}'.format(start_index, end_index));

        #Others
        start_index = others_folds[fold-1];
        end_index = len(other_samples) if fold==5 else others_folds[fold];
        finch_sounds.extend(other_samples[start_index:end_index]);
        print(len(finch_sounds))
        finch_labels.extend([1] * (end_index-start_index));
        print(len(finch_labels))
        print('Other Indexes: {} - {}'.format(start_index, end_index));

        print('Finished Fold {}'.format(fold));
        print('Samples: {}'.format(len(finch_sounds)));
        print('Labels: {}'.format(len(finch_labels)));
        finch_sounds, finch_labels = shuffle(finch_sounds,finch_labels, random_state=0);
        finch_dataset['fold{}'.format(fold)]['sounds'] = finch_sounds;
        finch_dataset['fold{}'.format(fold)]['labels'] = finch_labels;

        # exit();
    np.savez(finch_path+'wav20.npz', **finch_dataset);
