import numpy as np;


if __name__ == '__main__':
    data_path = '/Volumes/GoogleDrive/My Drive/phd/experiments/datasets/frog/wav32_norm_fixed.npz';
    dataset = np.load(data_path, allow_pickle=True);

    for i in range(1, 6):
        sounds = dataset['fold{}'.format(i)].item()['sounds'];
        count = len(sounds);
        print('Fold {} Data Count: {}'.format(i, count));
