import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import _pickle as pickle
import time
import random
import heapq
from collections import defaultdict


def build_tree(corMat, n_group_codes, device):
    infoDict = defaultdict(list)
    for i in range(n_group_codes):
        tmpAns = list(np.nonzero(corMat[i])[0])
        ansNum = len(tmpAns)
        tmpLea = [i] * ansNum
        infoDict[ansNum].append((tmpLea, tmpAns, i))
    lens = sorted(list(infoDict.keys()))
    leaveList = []
    ancestorList = []
    cur = 0
    mapInfo = [0] * n_group_codes
    for k in lens:
        leaves = []
        ancestors = []
        for meta in infoDict[k]:
            leaves.append(meta[0])
            ancestors.append(meta[1])
            mapInfo[meta[2]] = cur
            cur += 1
        leaves = torch.LongTensor(leaves).to(device)
        ancestors = torch.LongTensor(ancestors).to(device)
        leaveList.append(leaves)
        ancestorList.append(ancestors)
    return leaveList, ancestorList, mapInfo


def pad_matrix(seq_medical_codes, seq_group_codes, n_medical_codes,
               n_group_codes):
    lengths = np.array([len(seq) for seq in seq_group_codes
                        ]) - 1  # number of visits for each patient
    n_samples = len(seq_medical_codes)
    maxlen = np.max(lengths)
    batch_medical_codes = np.zeros((maxlen, n_samples, n_medical_codes))
    batch_group_codes = np.zeros((maxlen, n_samples, n_group_codes))
    mask = np.zeros((
        maxlen,
        n_samples,
    ))

    for idx, (c, g) in enumerate(zip(seq_medical_codes, seq_group_codes)):
        for x, subseq in zip(batch_medical_codes[:, idx, :], c[:-1]):
            x[subseq] = 1.
        for z, subseq in zip(batch_group_codes[:, idx, :], g[1:]):
            z[subseq] = 1.
        mask[:lengths[idx], idx] = 1.
    lengths = np.array(lengths)

    return batch_medical_codes, batch_group_codes, mask, lengths


def maskKLLoss(outputs, targets, masks, bceLoss):
    ele_loss = bceLoss(outputs, targets).sum(dim=2)
    masked_loss = (ele_loss.mul(masks).sum(dim=0) / masks.sum(dim=0)).mean()
    return masked_loss


def loadData(training_file, validation_file, testing_file):
    train = pickle.load(open(training_file, 'rb'))
    valid = pickle.load(open(validation_file, 'rb'))
    test = pickle.load(open(testing_file, 'rb'))
    return train, valid, test


def evaluate(outputs, targets, lengths, k):
    seq_len, batch_size, dict_size = outputs.size()
    outputs = outputs.data.cpu()
    targets = targets.data.cpu()
    hits = []
    recalls = []
    maps = []
    for i in range(batch_size):
        maxlen = lengths[i]
        predicts = np.array(outputs[maxlen - 1, i, :])
        labels = np.array(targets[maxlen - 1, i, :])
        yNum = float(np.sum(labels))
        topK = heapq.nlargest(k, range(dict_size), predicts.take)
        hit = 0
        Map = 0
        for r in range(k):
            if labels[topK[r]] == 1:
                hit += 1
                Map += float(hit) / float(r + 1)
        Map /= yNum
        hits.append(hit)
        recall = float(hit) / yNum
        recalls.append(recall)
        maps.append(Map)
    return np.mean(hits), np.mean(recalls), np.mean(maps)


def extractProfile(profiles, profile_size):
    res = []
    for profile in profiles:
        tmpres = [0] * 11
        if profile[0] == '"F"':  # about gender
            tmpres[0] = 1
        else:
            tmpres[1] = 1
        if int(profile[1]) <= 1:  # about age
            tmpres[2] = 1
        elif int(profile[1]) <= 18:
            tmpres[3] = 1
        elif int(profile[1]) <= 60:
            tmpres[4] = 1
        elif int(profile[1]) <= 80:
            tmpres[5] = 1
        else:
            tmpres[6] = 1
        if profile[4] == '"PHYS REFERRAL/NORMAL DELI"':
            tmpres[7] = 1
        if profile[4] == '"EMERGENCY ROOM ADMIT"':
            tmpres[8] = 1
        if profile[4] == '"CLINIC REFERRAL/PREMATURE"':
            tmpres[9] = 1
        if profile[4] == '"TRANSFER FROM HOSP/EXTRAM"':
            tmpres[10] = 1
        res.append(tmpres[:profile_size])
    return res
