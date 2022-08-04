# -*- coding: utf-8 -*-
'''
@ author LiJ
@ version 1.0
@ date 2022 / 7 / 7 15: 34
@ Description: encoding sequence data by one-hot encode
'''

import pandas as pd
import numpy as np
from random import shuffle
import Bio.SeqIO
from tqdm import tqdm
import pickle

def encode_dou(open_file_name):
    geneID = []
    seqs = []

    for x in Bio.SeqIO.parse('../Data/Seqs/TSS+TTS_3k_contact_seq.fasta', 'fasta'):
        # for x in Bio.SeqIO.parse('../Data/B73_3000.fa', 'fasta'):
        geneID.append(x.id[:14])
        seqs.append(str(x.seq))
    geneID = np.array([x.replace('"', '') for x in geneID])
    seqs = np.array(seqs)

    file = pd.read_csv('../Data/exp_int_data_dou/B73(' + open_file_name + ')values_500_dou.csv', encoding='gbk')
    ann1 = file['Annotation1'].values.astype('str')
    ann2 = file['Annotation2'].values.astype('str')

    from tqdm import tqdm

    d = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
         'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1],
         'N': [0, 0, 0, 0]}

    def encode(seq):
        tmp = []
        for i in seq:
            tmp.append(d[i])
        return np.array(tmp, dtype='float16').T

    dna1 = []
    dna2 = []

    for i in tqdm(range(len(ann1))):
        tmp1 = seqs[(geneID == ann1[i]) & (ann1[i][:2] == 'Zm')]
        tmp2 = seqs[(geneID == ann2[i]) & (ann2[i][:2] == 'Zm')]
        if tmp1.size * tmp2.size > 0:
            dna1.append(encode(tmp1[0]))
            dna2.append(encode(tmp2[0]))

    dna1 = np.array(dna1, dtype='float16')
    dna2 = np.array(dna2, dtype='float16')

    pickle.dump(dna1, open('../Data/encode_dou/dna1_B73'+open_file_name+'_500_dou', 'wb'))
    pickle.dump(dna2, open('../Data/encode_dou/dna2_B73'+open_file_name+'_500_dou', 'wb'))

def encode_sin(open_file_name):
    geneID = []
    seqs = []

    for x in Bio.SeqIO.parse('../Data/Seqs/TSS+TTS_3k_contact_seq.fasta', 'fasta'):
        # for x in Bio.SeqIO.parse('../Data/B73_3000.fa', 'fasta'):
        geneID.append(x.id[:14])
        seqs.append(str(x.seq))
    geneID = np.array([x.replace('"', '') for x in geneID])
    seqs = np.array(seqs)

    file = pd.read_csv('../Data/exp_int_data_sin/B73(' + open_file_name + ')values_500_sin.csv', encoding='gbk')
    ann1 = file['Annotation1'].values.astype('str')
    ann2 = file['Annotation2'].values.astype('str')

    from tqdm import tqdm

    d = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
         'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1],
         'N': [0, 0, 0, 0]}

    def encode(seq):
        tmp = []
        for i in seq:
            tmp.append(d[i])
        return np.array(tmp, dtype='float16').T

    dna1 = []
    dna2 = []

    for i in tqdm(range(len(ann1))):
        tmp1 = seqs[(geneID == ann1[i]) & (ann1[i][:2] == 'Zm')]
        tmp2 = seqs[(geneID == ann2[i]) & (ann2[i][:2] == 'Zm')]
        if tmp1.size * tmp2.size > 0:
            dna1.append(encode(tmp1[0]))
            dna2.append(encode(tmp2[0]))

    dna1 = np.array(dna1, dtype='float16')
    dna2 = np.array(dna2, dtype='float16')

    pickle.dump(dna1, open('../Data/encode_sin/dna1_B73'+open_file_name+'_500_sin', 'wb'))
    pickle.dump(dna2, open('../Data/encode_sin/dna2_B73'+open_file_name+'_500_sin', 'wb'))