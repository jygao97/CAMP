import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import _pickle as pickle
import time
import random
import heapq
import sys
from collections import defaultdict
from utils import build_tree, pad_matrix, maskKLLoss, loadData, evaluate, extractProfile
from model import gruPredictor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=50)
parser.add_argument('-device', type=int, default=0)
parser.add_argument('-batch_size', type=int, default=100)
parser.add_argument('-embd_size', type=int, default=128)
parser.add_argument('-hidden_size', type=int, default=384)
parser.add_argument('-atten_size', type=int, default=48)
parser.add_argument('-value_size', type=int, default=16)
parser.add_argument('-profile_embd_size', type=int, default=10)
parser.add_argument('-profile_size', type=int, default=11)
parser.add_argument('-l_rate', type=float, default=0.001)
parser.add_argument('-l2', type=float, default=0.001)
parser.add_argument('-drop_rate', type=float, default=0.65)
parser.add_argument('-description', type=str, default='default')
args = parser.parse_args()
print(args)
mapRes = defaultdict(list)
recallRes = defaultdict(list)

for seed in range(2019, 2009, -1):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  #gpu
    np.random.seed(seed)  #numpy
    random.seed(seed)  #random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn

    n_medical_codes = 272
    n_group_codes = 272

    training_file = '../data/mimic/mimic.train'
    validation_file = '../data/mimic/mimic.valid'
    testing_file = '../data/mimic/mimic.test'
    model_file = '../models/mimic/camp_{}.model'.format(args.description)
    hierarchy_file = '../data/mimic/mimic.forgram'
    bceLoss = nn.BCELoss(reduction='none')
    device = torch.device("cuda:{}".format(args.device)
                          if torch.cuda.is_available() else "cpu")
    types, newfather, corMat, ini_embds = pickle.load(
        open(hierarchy_file, 'rb'))
    KMIds = [types['.{}'.format(i)] for i in range(1, 18)]

    n_total_medical_nodes = corMat.shape[0]
    leavesList, ancestorList, mapInfo = build_tree(corMat, n_group_codes,
                                                   device)
    new_version = False
    if '1.0.' in torch.__version__:
        new_version = True

    print("available device: {}".format(device))
    train, valid, test = loadData(training_file, validation_file, testing_file)
    n_batches = int(np.ceil(float(len(train[0])) / float(args.batch_size)))
    print('n_batches:{}'.format(n_batches))
    ini_embds = torch.FloatTensor(ini_embds).to(device)
    mapInfo = torch.LongTensor(mapInfo).to(device)
    rnn = gruPredictor(n_medical_codes, n_total_medical_nodes, args.embd_size,
                       args.hidden_size, args.atten_size, leavesList,
                       ancestorList, mapInfo, ini_embds, KMIds,
                       args.value_size, args.profile_size,
                       args.profile_embd_size, args.drop_rate).to(device)
    optimizer = optim.Adam(
        rnn.parameters(), lr=args.l_rate, weight_decay=args.l2)
    min_valid_loss = 10000000000
    best_epoch = -1
    print("Start Train")
    time1 = time.time()
    for epoch in range(args.epochs):
        rnn.train()
        iteration = 0
        cost_vector = []
        start_time = time.time()
        samples = random.sample(range(n_batches), n_batches)
        counter = 0
        iters = 0
        losses = []
        start_time = time.time()
        for index in samples:
            batch_medical_codes = train[0][args.batch_size * index:
                                           args.batch_size * (index + 1)]
            batch_group_codes = train[0][args.batch_size * index:
                                         args.batch_size * (index + 1)]
            t_medical_codes, t_group_codes, t_mask, t_lengths = pad_matrix(
                batch_medical_codes, batch_group_codes, n_medical_codes,
                n_group_codes)
            t_profiles = extractProfile(
                train[1][args.batch_size * index:args.batch_size * (
                    index + 1)], args.profile_size)
            t_profiles = torch.FloatTensor(t_profiles).to(device)
            t_medical_codes = torch.FloatTensor(t_medical_codes).to(device)
            t_group_codes = torch.FloatTensor(t_group_codes).to(device)
            t_mask = torch.FloatTensor(t_mask).to(device)
            if new_version:
                t_lengths = torch.FloatTensor(t_lengths).to(device)
            optimizer.zero_grad()
            outputs = rnn(t_medical_codes, t_lengths, device, t_profiles)
            loss = maskKLLoss(outputs, t_group_codes, t_mask, bceLoss)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        with torch.no_grad():
            rnn.eval()
            batch_medical_codes_valid = valid[0]
            batch_group_codes_valid = valid[0]
            t_medical_codes, t_group_codes, t_mask, t_lengths = pad_matrix(
                batch_medical_codes_valid, batch_group_codes_valid,
                n_medical_codes, n_group_codes)
            t_profiles = extractProfile(valid[1], args.profile_size)
            t_profiles = torch.FloatTensor(t_profiles).to(device)
            t_medical_codes = torch.FloatTensor(t_medical_codes).to(device)
            t_group_codes = torch.FloatTensor(t_group_codes).to(device)
            t_mask = torch.FloatTensor(t_mask).to(device)
            if new_version:
                t_lengths = torch.FloatTensor(t_lengths).to(device)
            outputs = rnn(t_medical_codes, t_lengths, device, t_profiles)
            # print("start evaluate")
            # recall,precision= evaluate(outputs,t_group_codes,t_mask,5)
            loss_valid = maskKLLoss(outputs, t_group_codes, t_mask, bceLoss)

        # duration = time.time() - start_time
        # print("epoch:{} train:{} valid:{} duration:{}".format(epoch+1,np.mean(losses),loss_valid,duration))
        # print("         recall@5: {} precision@5: {}".format(recall,precision))
        if loss_valid < min_valid_loss:
            min_valid_loss = loss_valid
            best_epoch = epoch
            state = {
                'net': rnn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(state, model_file)

    print("best valid loss {}".format(min_valid_loss))
    checkpoint = torch.load(model_file)
    save_epoch = checkpoint['epoch']
    print("last saved model is in epoch {}".format(save_epoch))
    rnn.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    rnn.eval()

    #testing the model
    with torch.no_grad():
        batch_medical_codes_test = test[0]
        batch_group_codes_test = test[0]
        t_medical_codes, t_group_codes, t_mask, t_lengths = pad_matrix(
            batch_medical_codes_test, batch_group_codes_test, n_medical_codes,
            n_group_codes)
        t_profiles = extractProfile(test[1], args.profile_size)
        t_profiles = torch.FloatTensor(t_profiles).to(device)
        t_medical_codes = torch.FloatTensor(t_medical_codes).to(device)
        t_group_codes = torch.FloatTensor(t_group_codes).to(device)
        t_mask = torch.FloatTensor(t_mask).to(device)
        copy_length = t_lengths.copy()
        if new_version:
            t_lengths = torch.FloatTensor(t_lengths).to(device)
        outputs = rnn(t_medical_codes, t_lengths, device, t_profiles)
        loss_test = maskKLLoss(outputs, t_group_codes, t_mask, bceLoss)
        print("test loss {}".format(loss_test))
        print("start evaluate")
        for k in [5, 10, 15]:
            hit, recall, Map = evaluate(outputs, t_group_codes, copy_length, k)
            print("hit@{} {} recall@{} {} map@{} {}".format(
                k, hit, k, recall, k, Map))
            recallRes[k].append(recall)
            mapRes[k].append(Map)
    print(time.time() - time1)

r5_mean = round(np.mean(recallRes[5]), 3)
r5_std = round(np.std(recallRes[5]), 3)
m5_mean = round(np.mean(mapRes[5]), 3)
m5_std = round(np.std(mapRes[5]), 3)
r10_mean = round(np.mean(recallRes[10]), 3)
r10_std = round(np.std(recallRes[10]), 3)
m10_mean = round(np.mean(mapRes[10]), 3)
m10_std = round(np.std(mapRes[10]), 3)
r15_mean = round(np.mean(recallRes[15]), 3)
r15_std = round(np.std(recallRes[15]), 3)
m15_mean = round(np.mean(mapRes[15]), 3)
m15_std = round(np.std(mapRes[15]), 3)

print('r@5: {}+-{} m@5: {}+-{}\n'.format(r5_mean, r5_std, m5_mean, m5_std))
print('r@10: {}+-{} m@10: {}+-{}\n'.format(r10_mean, r10_std, m10_mean,
                                           m10_std))
print('r@15: {}+-{} m@15: {}+-{}\n'.format(r15_mean, r15_std, m15_mean,
                                           m15_std))
print(recallRes[5])
print(recallRes[10])
print(recallRes[15])
print(mapRes[5])
print(mapRes[10])
print(mapRes[15])
