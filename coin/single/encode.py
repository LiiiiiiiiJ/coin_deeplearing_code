# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from random import shuffle
import Bio.SeqIO
from tqdm import tqdm
import pickle


def encode(open_file_name):
    geneID = []
    seqs = []

    for x in Bio.SeqIO.parse('../Data/Seqs/TSS+TTS_3k_contact_seq.fasta', 'fasta'):
        # for x in Bio.SeqIO.parse('../Data/B73_3000.fa', 'fasta'):
        geneID.append(x.id[:14])
        seqs.append(str(x.seq))
    geneID = np.array([x.replace('"', '') for x in geneID])
    seqs = np.array(seqs)

    file = pd.read_csv('../Data/express/B73expressed0~500(' + open_file_name + ').csv', encoding='gbk')
    ann = file['tracking_id'].values.astype('str')
    exp = file['value'].values.astype('float')

    temp = [ann, exp]
    temp = list(zip(*temp))
    shuffle(temp)

    ann = [i[0] for i in temp]
    exp = [i[1] for i in temp]

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
    tmp_ann=[]
    tmp_exp=[]
    for i in tqdm(range(len(ann))):
        tmp1 = seqs[(geneID == ann[i]) & (ann[i][:2] == 'Zm')]
        if tmp1.size > 0 and len(tmp1[0])==3000:
            dna1.append(encode(tmp1[0]))
            tmp_ann.append(ann[i])
            tmp_exp.append(exp[i])

    dna1 = np.array(dna1, dtype='float16')
    pickle.dump(dna1, open('../Data/encode/dna1_B73' + open_file_name + '_500_single', 'wb'))

    df = pd.DataFrame({'tracking_id': tmp_ann, 'value': tmp_exp, })
    df.to_csv('../Data/exp_data/B73(' + open_file_name + ')values_500_single.csv', index=None)
