'''
This script performs preprocessing 
e.g., dividing dataset into 3 parts, constructing demography for each patient
and associating each visit with time stamp
'''
import sys
import argparse
import os
import _pickle as pickle
import numpy as np


def computeDuration(times):
    newtimes = []
    for time in times:
        newtime = []
        first = True
        bfDate = 0
        for date in time:
            if first:
                bfDate = date
                first = False
                newtime.append(0)
            else:
                dur = (date - bfDate).days
                newtime.append(dur)
                bfDate = date
        newtimes.append(newtime)
    return newtimes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_size',
        type=float,
        default=0.75,
        help='size of training dataset')
    parser.add_argument(
        '--valid_size', type=float, default=0.10, help='size of test dataset')
    parser.add_argument(
        '--dataFolder',
        type=str,
        default='../data/mimic',
        help='path of the input dataFolder')
    args = parser.parse_args()

    seqs = pickle.load(open(os.path.join(args.dataFolder, 'mimic.seqs'), 'rb'))
    profiles = pickle.load(
        open(os.path.join(args.dataFolder, 'mimic.profiles'), 'rb'))
    times = pickle.load(
        open(os.path.join(args.dataFolder, 'mimic.dates'), 'rb'))
    pids = pickle.load(open(os.path.join(args.dataFolder, 'mimic.pids'), 'rb'))
    times = computeDuration(times)

    dataSize = len(seqs)
    np.random.seed(0)
    ind = np.random.permutation(dataSize)
    nTrain = int(args.train_size * dataSize)
    nValid = int(args.valid_size * dataSize)

    train_indices = ind[:nTrain]
    valid_indices = ind[nTrain:nTrain + nValid]
    test_indices = ind[nTrain + nValid:]

    train_set_seqs = np.array(seqs)[train_indices]
    train_set_profiles = np.array(profiles)[train_indices]
    train_set_times = np.array(times)[train_indices]
    train_set_pids = np.array(pids)[train_indices]
    valid_set_seqs = np.array(seqs)[valid_indices]
    valid_set_profiles = np.array(profiles)[valid_indices]
    valid_set_times = np.array(times)[valid_indices]
    valid_set_pids = np.array(pids)[valid_indices]
    test_set_seqs = np.array(seqs)[test_indices]
    test_set_profiles = np.array(profiles)[test_indices]
    test_set_times = np.array(times)[test_indices]
    test_set_pids = np.array(pids)[test_indices]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]), reverse=True)

    train_sorted_index = len_argsort(train_set_seqs)
    train_set_seqs = [train_set_seqs[i] for i in train_sorted_index]
    train_set_profiles = [train_set_profiles[i] for i in train_sorted_index]
    train_set_times = [train_set_times[i] for i in train_sorted_index]
    train_set_pids = [train_set_pids[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_seqs)
    valid_set_seqs = [valid_set_seqs[i] for i in valid_sorted_index]
    valid_set_profiles = [valid_set_profiles[i] for i in valid_sorted_index]
    valid_set_times = [valid_set_times[i] for i in valid_sorted_index]
    valid_set_pids = [valid_set_pids[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_seqs)
    test_set_seqs = [test_set_seqs[i] for i in test_sorted_index]
    test_set_profiles = [test_set_profiles[i] for i in test_sorted_index]
    test_set_times = [test_set_times[i] for i in test_sorted_index]
    test_set_pids = [test_set_pids[i] for i in test_sorted_index]

    trainSet = (train_set_seqs, train_set_profiles, train_set_times,
                train_set_pids)
    validSet = (valid_set_seqs, valid_set_profiles, valid_set_times,
                valid_set_pids)
    testSet = (test_set_seqs, test_set_profiles, test_set_times, test_set_pids)

    pickle.dump(trainSet,
                open(os.path.join(args.dataFolder, 'mimic.train'), 'wb'), -1)
    pickle.dump(validSet,
                open(os.path.join(args.dataFolder, 'mimic.valid'), 'wb'), -1)
    pickle.dump(testSet, open(
        os.path.join(args.dataFolder, 'mimic.test'), 'wb'), -1)
