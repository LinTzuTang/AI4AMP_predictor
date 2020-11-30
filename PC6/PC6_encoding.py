#!/usr/bin/env python
# coding: utf-8

import re
import numpy as np
import pandas as pd
from Bio import SeqIO
import gensim
import os
import json



# read fasta as dict
def read_fasta(fasta_name):
    r = {}
    for record in SeqIO.parse(fasta_name, 'fasta'):
        idtag = str(record.id)
        seq = str(record.seq)
        r[idtag] = seq
    return r

# sequence padding (token:'X')
def padding_seq(r,length=200,pad_value='X'):
    data={}
    for key, value in r.items():
        if len(r[key]) > length:
            print('squence length over padding length ')
            break
        elif len(r[key]) <= length:
            r[key] = [r[key]+pad_value*(length-len(r[key]))]

        data[key] = r[key]
    return data



# generate PC6 table
def amino_encode_table_6(path=None):
    root=os.path.dirname(__file__)
    path = path or root+'/../PC6/6-pc'
    df = pd.read_csv(path, sep=' ', index_col=0)
    H1 = (df['H1'] - np.mean(df['H1'])) / (np.std(df['H1'], ddof=1))
    V = (df['V'] - np.mean(df['V'])) / (np.std(df['V'], ddof=1))
    P1 = (df['P1'] - np.mean(df['P1'])) / (np.std(df['P1'], ddof=1))
    Pl = (df['Pl'] - np.mean(df['Pl'])) / (np.std(df['Pl'], ddof=1))
    PKa = (df['PKa'] - np.mean(df['PKa'])) / (np.std(df['PKa'], ddof=1))
    NCI = (df['NCI'] - np.mean(df['NCI'])) / (np.std(df['NCI'], ddof=1))
    c = np.array([H1,V,P1,Pl,PKa,NCI])
    amino = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    table = {}
    for index,key in enumerate(amino):
        table[key]=list(c[0:6,index])
    table['X'] = [0,0,0,0,0,0]
    return table


# encoding
def encoding(data):
    #method = integer or onehot
    dat={}
    for key in data.keys():
        integer_encoded = []
        for amino in list(data[key][0]):
            integer_encoded.append(amino_encode_table_6()[amino])
        dat[key]=integer_encoded
    return dat

# PC6 encoding (input: fasta) 
def PC6_encoding(fasta_name, length=200):
    r = read_fasta(fasta_name)
    data = padding_seq(r, length)
    dat = encoding(data)
    return dat