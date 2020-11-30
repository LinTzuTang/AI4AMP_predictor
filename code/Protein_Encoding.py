#!/usr/bin/env python
# coding: utf-8

import re
import numpy as np
import pandas as pd
from Bio import SeqIO

# generate PC6 table
def amino_encode_table_6(path=None):
    path = None or 'Data/6_physicochemical_properties/6-pc'
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
        table[key]=c[0:6,index]
    table['X'] = [0,0,0,0,0,0]
    return table

# read fasta as dict
def read_fasta(fasta_fname):
    path = 'Data/Fasta/'+ fasta_fname
    r = dict()
    for record in SeqIO.parse(path, 'fasta'):
        idtag = str(record.id)
        seq = str(record.seq)
        r[idtag] = seq
    return r

# sequence padding (token:'X')
def padding_seq(r,length=200,pad_value='X'):
    data={}
    for key, value in r.items():
        if len(r[key]) <= length:
            r[key] = [r[key]+pad_value*(length-len(r[key]))]

        data[key] = r[key]
    return data


# PC encoding
def PC_encoding(data):
    table = amino_encode_table_6()
    dat={}
    for  key in data.keys():
        integer_encoded = []
        for amino in list(data[key][0]):
            integer_encoded.append(table[amino])
        dat[key]=integer_encoded
    return dat


# PC6 (input: fasta) 
def PC_6(fasta_name, length=200):
    r = read_fasta(fasta_name)
    data = padding_seq(r, length=200)
    dat = PC_encoding(data)
    return dat

# PC decoding
def decode(encoding_value):
    table= amino_encode_table_6()
    for key, value in table.items():
        if (value == encoding_value).all():
            return key
        
def re_sequence(embedding_array):
    d = []
    for aa in embedding_array:
        d.append(decode(aa))
    return ''.join(d)

# decode nparray to fasta 
def data_decoding(data,file_name='a'):
    name = []
    for i in range(1,len(data)+1):
        name.append('>sequence_'+str(i))
    print('sequence_count:%s'%len(data))
    path = 'Output/Fasta/'
    with open(path+'%s.fasta'%file_name, 'w') as fasta:
        for index, sequence in enumerate(data):
            fasta.write(name[index]+'\n')
            fasta.write(re_sequence(sequence)+'\n')
    
