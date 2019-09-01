import _pickle as pickle
import os
import copy
import numpy as np
from glove import Glove
from scipy.sparse import coo_matrix


def cidHierarchy(ccsFolder):
    multiFile = open(os.path.join(ccsFolder, 'ccs_multi'), 'r')
    cid2Code = {}
    for l in multiFile:
        if l[0].isdigit() and '[' in l:
            cid = int(l.strip().split('[')[1].split(']')[0].split('.')[0])
            line = l.strip().split()
            code = line[0]
            cid2Code[cid] = code
    multiFile.close()
    for i in range(1, 22):
        cid2Code[2600 + i] = '18.{}'.format(i)
    cid2Code[259] = '18.22'
    return cid2Code


typeFile = '../data/mimic/mimic.types'
types = pickle.load(open(typeFile, 'rb'))
cid2Code = cidHierarchy(
    '../data/ccs')  # folder that ccs_multi_level file locates in
resFile = '../data/mimic/mimic.forgram'

father = {}
virtualNode = '0'
for cid in types:
    code = cid2Code[cid]
    dotIndex = [i for i in range(len(code)) if code[i] == '.']
    nodes = [code[:i] for i in dotIndex]
    nodes[0] = '.' + nodes[0]
    father[cid] = nodes[-1]
    curPos = len(nodes) - 1
    while curPos > 0:
        father[nodes[curPos]] = nodes[curPos - 1]
        curPos = curPos - 1
    father[nodes[0]] = virtualNode
for k in father:
    if k not in types:
        types[k] = len(types)
types[virtualNode] = len(types)
newfather = {}
for cid in father:
    s = types[cid]
    f = types[father[cid]]
    newfather[s] = f
id2code = {}
for k in types:
    id2code[types[k]] = k
corMat = np.zeros((len(types), len(types)))
for i in range(len(types)):
    corMat[i, i] = 1
    curId = i
    while curId in newfather:
        f = newfather[curId]
        corMat[i, f] = 1
        curId = f
trainFile = '../data/mimic/mimic.train'
coOccurMat = np.zeros((len(types), len(types)))
trainSet = pickle.load(open(trainFile, 'rb'))[0]
for patient in trainSet:
    for visit in patient:
        augmented = np.nonzero(sum(corMat[visit]))[0]
        listNum = len(augmented)
        for i in range(listNum):
            for j in range(i + 1, listNum):
                coOccurMat[augmented[i]][augmented[j]] += 1
                coOccurMat[augmented[j]][augmented[i]] += 1
coOccurMatzip = coo_matrix(coOccurMat.astype(np.float))
glove = Glove(no_components=128, learning_rate=0.05)
glove.fit(coOccurMatzip, epochs=50, no_threads=1.0, verbose=True)
res = glove.word_vectors.astype(np.float32)
pickle.dump(
    (types, newfather, corMat, res), open(resFile, 'wb'), -1
)  # save types (node to id, newfather, cormat and pre-trained embeddings)
