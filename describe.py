import sys;
import os;

import glob;
import pandas as pd;

if __name__ == '__main__':
    # main()
    mainDir = os.path.dirname(os.path.abspath('.'));
    label_path = os.path.join(mainDir, 'datasets/finch/btf_labels/');
    btf_samples = [];
    btf_count = 0;
    empty_count = 0;
    hfeq = 0;
    lfeq = 44100;
    for txt_file_path in sorted(glob.glob(os.path.join(label_path, '*.txt'))):
        file = pd.read_csv(txt_file_path, sep='\t', usecols=[3,4,5,6], header=0, names=['start', 'end', 'low', 'high']);
        file.dropna(axis=0, how='any', inplace=True);

        # print(file);
        file_name = os.path.split(txt_file_path)[1].split('.Table')[0];
        # print(name.split('.Table')[0]);
        print(file_name);
        max = file['high'].max();
        if max > hfeq:
            hfeq = max;
        min = file['low'].min();
        if min < lfeq:
            lfeq = min;
        btf_count += len(file);
        if len(file) == 0:
            empty_count += 1;

    print(btf_count);
    print(empty_count);
    print(hfeq);
    print(lfeq);
    exit();
