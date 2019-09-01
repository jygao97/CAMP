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


class gruPredictor(nn.Module):
    def __init__(self, dict_size, total_node_size, embd_size, hidden_size,
                 atten_size, leavesList, ancesterList, mapInfo, ini_embds,
                 KMIds, value_size, profile_size, profile_embd_size,
                 drop_rate):
        super(gruPredictor, self).__init__()
        self.dict_size = dict_size
        self.embd_size = embd_size
        self.hidden_size = hidden_size
        self.leavesList = leavesList
        self.ancesterList = ancesterList
        self.mapInfo = mapInfo
        self.KMIds = KMIds
        self.value_size = value_size
        self.profile_size = profile_size
        self.profile_embd_size = profile_embd_size
        self.slotNum = len(KMIds)

        self.ini_embd = nn.Parameter(torch.randn(
            total_node_size, embd_size))  # using randam ini
        #self.ini_embd = nn.Parameter(ini_embd) # using given ini
        self.Wa = nn.Linear(2 * embd_size, atten_size)
        self.Ua = nn.Linear(atten_size, 1, bias=False)
        self.gru = nn.GRU(embd_size, hidden_size)  # the gru layer
        self.tranH = nn.Linear(hidden_size + profile_embd_size,
                               embd_size)  # transform h_t to h_t'
        self.drop = nn.Dropout(p=drop_rate)
        self.out = nn.Linear(hidden_size + value_size + profile_embd_size,
                             dict_size)  # the prediction layer
        self.erase = nn.Linear(embd_size, value_size)  # the erase layer
        self.add = nn.Linear(embd_size, value_size)  # the add layer
        self.iniVam = nn.Parameter(torch.randn(self.slotNum, value_size))
        self.embd_profile = nn.Linear(profile_size, profile_embd_size)
        self.ReLU = nn.ReLU()
        self.attnOnPro = nn.Linear(profile_embd_size + value_size,
                                   profile_embd_size)

    def forward(self, X, X_lengths, device, profiles, hidden=None):
        seq_len, batch_size, dict_size = X.size()
        KM = self.ini_embd[self.KMIds]  # key-memory: slotNum * embd_size
        VM = self.iniVam.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        #VM = torch.zeros(batch_size,self.slotNum,self.value_size).to(device) # value-memory: b* slotNum * value_size
        embedList = []
        for leaves, ancestors in zip(self.leavesList, self.ancesterList):
            sampleNum, _ = leaves.size()
            leavesEmbd = self.ini_embd[leaves.view(1, -1).squeeze(dim=0)].view(
                sampleNum, -1, self.embd_size)
            ancestorsEmbd = self.ini_embd[ancestors.view(
                1, -1).squeeze(dim=0)].view(sampleNum, -1, self.embd_size)
            concated = torch.cat(
                [leavesEmbd, ancestorsEmbd],
                dim=2)  # nodeNum * len* 2 embd_size
            weights = F.softmax(
                self.Ua(torch.tanh(self.Wa(concated))), dim=1).transpose(1, 2)
            embedList.append(
                weights.bmm(self.ini_embd[ancestors]).squeeze(dim=1))
        embedMat = torch.cat(embedList, dim=0)
        embedMat = embedMat[self.mapInfo]
        X = torch.einsum('bij,jk->bik', [X, embedMat])
        packed = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths)
        # Forward pass through GRU
        gru_out, hidden = self.gru(packed, hidden)  # gru layer
        # Unpack padding
        gru_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            gru_out)  #gru_out: seq_len * batch_size * hidden_size
        #gru_out_tran  = self.tranH(gru_out) # s*b*embd_size

        profiles = self.embd_profile(profiles)
        profiles = profiles.unsqueeze(0).repeat(seq_len, 1, 1)
        gru_profile = torch.cat([gru_out, profiles], dim=2)
        gru_out_tran = self.tranH(gru_profile)

        slot_weights = F.softmax(
            torch.einsum('bij,jk->bik',
                         [gru_out_tran, KM.transpose(0, 1)]),
            dim=2)  # s*b*slotNum
        read = []
        sigmoid = torch.nn.Sigmoid()
        for i in range(seq_len):
            # read
            tmp_read = torch.einsum(
                'bi,bij->bj', [slot_weights[i, :, :], VM]).unsqueeze(dim=0)
            read.append(tmp_read)
            # write
            erase_ratio = sigmoid(self.erase(X[i, :, :]))  # batch*value_size
            reserve_ratio = 1 - torch.einsum(
                'bi,bj->bij', [slot_weights[i, :, :], erase_ratio
                               ])  # batch * slotNum * value_size
            add_ratio = torch.tanh(self.add(X[i, :, :]))
            add_part = torch.einsum('bi,bj->bij',
                                    [slot_weights[i, :, :], add_ratio
                                     ])  # batch * slotNum * value_size
            VM = VM * reserve_ratio + add_part
        read = torch.cat(
            read,
            dim=0)  # s*b*value_size, the content read from memory network

        attn_on_pro_pre = torch.cat([profiles, read], dim=2)
        attn_on_pro = self.ReLU(self.attnOnPro(attn_on_pro_pre))
        profiles = profiles.mul(attn_on_pro)

        # attn_on_gru_pre = torch.cat([gru_out,read],dim=2)
        # attn_on_gru = torch.tanh(self.attnOnGru(attn_on_gru_pre))
        # gru_out = gru_out.mul(attn_on_gru)

        finalPre = torch.cat([gru_out, read, profiles], dim=2)
        outputs = self.drop(finalPre)
        outputs = self.out(
            outputs)  # predict the probability for each diagnosis code
        outputs = F.softmax(outputs, dim=2)
        return outputs
