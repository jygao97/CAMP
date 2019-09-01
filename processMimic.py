import sys
import _pickle as pickle
from datetime import datetime
import os


def icdMapper(ccsFolder):
    singleFile = open(os.path.join(ccsFolder, 'ccs_single'), 'r')
    cnt = 0
    icd2Cid = {}
    cidSet = set()
    for l in singleFile:
        if l[0].isdigit():
            line = l.strip().split()
            curCid = int(line[0])
            cidSet.add(curCid)
        else:
            line = l.strip().split()
            for icd in line:
                icd2Cid[icd] = curCid
    singleFile.close()
    return icd2Cid


def cidHierarchy(ccsFolder):
    multiFile = open(os.path.join(cssFolder, 'ccs_multi'), 'r')
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
    return cid2Code


if __name__ == '__main__':
    mimicFolder = sys.argv[1]
    ccsFolder = sys.argv[2]
    outFile = sys.argv[3]

    print("build ics grouper with css single layer file")
    icd2Cid = icdMapper(ccsFolder)

    print("read patient table")
    pidGenMap, pidDobMap, pidDodMap = {}, {}, {
    }  # obtain information about gender, date of birth and death information of patients

    infd = open(os.path.join(mimicFolder, 'PATIENTS.csv'), 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        pidGenMap[pid] = tokens[2]
        dobTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
        pidDobMap[pid] = dobTime

        dod_hosp = tokens[5]
        if len(dod_hosp) > 0:
            pidDodMap[pid] = 1
        else:
            pidDodMap[pid] = 0
    infd.close()

    print('read admission diagnosis table')
    admDxMap = {}
    admDxMap_icd = {}
    invalidAdmId = []
    infd = open(os.path.join(mimicFolder, 'DIAGNOSES_ICD.csv'), 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[2])
        if len(tokens[3]) == 0:
            invalidAdmId.append(admId)
            continue
        icd = tokens[4][1:-1]
        cidStr = icd2Cid[tokens[4][
            1:
            -1]]  ############## Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
        if admId in admDxMap:
            admDxMap[admId].append(cidStr)
        else:
            admDxMap[admId] = [cidStr]
        if admId in admDxMap_icd:
            admDxMap_icd[admId].append(icd)
        else:
            admDxMap_icd[admId] = [icd]
    infd.close()

    print("read admission table")
    pidAdmMap = {}
    admDateMap = {}
    pidMarMap = {}
    pidEthMap = {}
    pidAgeMap = {}
    pidLocMap = {}
    pidRegMap = {}
    pidTypeMap = {}
    infd = open(os.path.join(mimicFolder, 'ADMISSIONS.csv'), 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        admId = int(tokens[2])
        admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
        admDateMap[admId] = admTime
        if pid not in pidTypeMap:
            pidTypeMap[pid] = tokens[6]
        if pid not in pidLocMap:
            pidLocMap[pid] = tokens[7]
        if pid not in pidMarMap:
            pidMarMap[pid] = tokens[12]
        if pid not in pidEthMap:
            pidEthMap[pid] = tokens[13]
        if pid not in pidRegMap:
            pidRegMap[pid] = tokens[11]
        if admId in invalidAdmId:
            continue
        if pid in pidAdmMap: pidAdmMap[pid].append(admId)
        else: pidAdmMap[pid] = [admId]
    infd.close()

    print('Building pid-sortedVisits mapping')
    pidSeqMap = {}
    pidSeqMap_icd = {}
    cnt = 0
    for pid, admIdList in pidAdmMap.items():
        if len(admIdList) < 2: continue

        sortedList = sorted(
            [(admDateMap[admId], admDxMap[admId]) for admId in admIdList])
        pidSeqMap[pid] = sortedList
        pidAgeMap[pid] = sortedList[0][0].year - pidDobMap[pid].year
        if pidAgeMap[pid] > 100:
            pidAgeMap[pid] = 100
        sortedList_icd = sorted(
            [(admDateMap[admId], admDxMap_icd[admId]) for admId in admIdList])
        pidSeqMap_icd[pid] = sortedList_icd

    print('Building pids, dates, profiles, strSeqs')
    pids = []
    dates = []
    seqs = []
    profiles = []
    seqs_icd = []
    for pid, visits in pidSeqMap.items():
        pids.append(pid)
        profile = [
            pidGenMap[pid], pidAgeMap[pid], pidMarMap[pid], pidEthMap[pid],
            pidLocMap[pid], pidTypeMap[pid], pidRegMap[pid], pidDodMap[pid]
        ]  # gender,age,marriage,ethic,location,death status of patients
        profiles.append(profile)
        seq = []
        date = []
        seq_icd = []
        for visit in visits:
            date.append(visit[0])
            seq.append(visit[1])
        for visit_icd in pidSeqMap_icd[pid]:
            seq_icd.append(visit_icd[1])
        dates.append(date)
        seqs.append(seq)
        seqs_icd.append(seq_icd)

    print('Converting strSeqs to intSeqs, and making types')
    types = {}  # index for cid
    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in types:
                    newVisit.append(types[code])
                else:
                    types[code] = len(types)
                    newVisit.append(types[code])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    types_icd = {}  # index for icd9 code
    newSeqs_icd = []
    for patient in seqs_icd:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in types_icd:
                    newVisit.append(types_icd[code])
                else:
                    types_icd[code] = len(types_icd)
                    newVisit.append(types_icd[code])
            newPatient.append(newVisit)
        newSeqs_icd.append(newPatient)

    pickle.dump(pids, open(os.path.join(outFile, 'mimic.pids'), 'wb'), -1)
    pickle.dump(dates, open(os.path.join(outFile, 'mimic.dates'), 'wb'), -1)
    pickle.dump(profiles, open(os.path.join(outFile, 'mimic.profiles'), 'wb'),
                -1)
    pickle.dump(newSeqs, open(os.path.join(outFile, 'mimic.seqs'), 'wb'), -1)
    pickle.dump(types, open(os.path.join(outFile, 'mimic.types'), 'wb'), -1)
    pickle.dump(newSeqs_icd, open(
        os.path.join(outFile, 'mimic.seqs_icd'), 'wb'), -1)
    pickle.dump(types_icd, open(
        os.path.join(outFile, 'mimic.types_icd'), 'wb'), -1)
