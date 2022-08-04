# -*- coding: utf-8 -*-
'''
@ author LiJ
@ version 1.0
@ date 2022 / 7 / 7 12: 34
@ Description: select data which sequence in fasta file
'''

import pandas as pd
import numpy as np
from random import shuffle
import Bio.SeqIO
from tqdm import tqdm


def select_exp(open_file_name):
    geneID = []
    seqs = []

    for x in Bio.SeqIO.parse('../Data/Seqs/TSS+TTS_3k_contact_seq.fasta', 'fasta'):
        # for x in Bio.SeqIO.parse('../Data/B73_3000.fa', 'fasta'):
        geneID.append(x.id[:14])
        seqs.append(str(x.seq))
    geneID = np.array([x.replace('"', '') for x in geneID])
    seqs = np.array(seqs)

    # shoot = pd.read_csv('../Data/py/B73(py).csv', encoding='gbk')
    shoot = pd.read_csv('../Data/cle_int/' + open_file_name + '.csv', encoding='gbk')
    ann1 = shoot['Annotation1'].values.astype('str')
    ann2 = shoot['Annotation2'].values.astype('str')

    exp = pd.read_csv('../Data/express/B73expressed0~500(' + open_file_name + ').csv', encoding='gbk')
    # exp = pd.read_csv('../Data/B73/B73expressed0~10(shoot).csv', encoding='gbk')
    seqs_id = exp['tracking_id'].values.astype('str')
    express = exp['value'].values.astype('float')

    # shuffle
    temp = [ann1, ann2]
    temp = list(zip(*temp))
    shuffle(temp)

    ann1 = [i[0] for i in temp]
    ann2 = [i[1] for i in temp]

    temp_ann1 = []
    temp_ann2 = []
    temp_exp = []
    for i in tqdm(range(len(ann1))):
        tmp1 = seqs[(geneID == ann1[i]) & (ann1[i][:2] == 'Zm')]
        tmp2 = seqs[(geneID == ann2[i]) & (ann2[i][:2] == 'Zm')]
        if tmp1.size * tmp2.size > 0 and ann2[i] in seqs_id and ann1[i] != ann2[i]:
            temp_ann1.append(ann1[i])
            temp_ann2.append(ann2[i])
            temp_exp.append(float(max(express[seqs_id == ann2[i]])))

    df = pd.DataFrame({'Annotation1': temp_ann1, 'Annotation2': temp_ann2, 'express': temp_exp, })
    df.to_csv('../Data/exp_int_data_sin/B73(' + open_file_name + ')values_500_sin.csv', index=None)

    for i in tqdm(range(len(ann1))):
        tmp1 = seqs[(geneID == ann1[i]) & (ann1[i][:2] == 'Zm')]
        tmp2 = seqs[(geneID == ann2[i]) & (ann2[i][:2] == 'Zm')]
        if tmp1.size * tmp2.size > 0 and ann1[i] in seqs_id and ann1[i] != ann2[i]:
            temp_ann1.append(ann2[i])
            temp_ann2.append(ann1[i])
            temp_exp.append(float(max(express[seqs_id == ann1[i]])))

    temp = [temp_ann1, temp_ann2, temp_exp]
    print(len(temp_ann1))
    temp = list(set(zip(*temp)))
    print(len(temp))
    print()

    temp_ann1 = [i[0] for i in temp]
    temp_ann2 = [i[1] for i in temp]
    temp_exp = [i[2] for i in temp]

    df = pd.DataFrame({'Annotation1': temp_ann1, 'Annotation2': temp_ann2, 'express': temp_exp, })
    df.to_csv('../Data/exp_int_data_dou/B73(' + open_file_name + ')values_500_dou.csv', index=None)
