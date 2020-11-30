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
def read_fasta(fasta_name, input_path=None):
    input_path = input_path or 'Data/Fasta' 
    r = {}
    for record in SeqIO.parse(os.path.join(input_path, fasta_name), 'fasta'):
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
    path = path or 'Data/6_physicochemical_properties/6-pc'
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

# generate PC7 table
def amino_encode_table_7(path=None):
    path=path or 'Data/7_physicochemical_properties/7-pc'
    df = pd.read_csv(path, sep=' ', index_col=0)
    H1 = (df['H1'] - np.mean(df['H1'])) / (np.std(df['H1'], ddof=1))
    H2 = (df['H2'] - np.mean(df['H2'])) / (np.std(df['H2'], ddof=1))
    NCI = (df['NCI'] - np.mean(df['NCI'])) / (np.std(df['NCI'], ddof=1))
    P1 = (df['P1'] - np.mean(df['P1'])) / (np.std(df['P1'], ddof=1))
    P2 = (df['P2'] - np.mean(df['P2'])) / (np.std(df['P2'], ddof=1))
    SASA = (df['SASA'] - np.mean(df['SASA'])) / (np.std(df['SASA'], ddof=1))
    V = (df['V'] - np.mean(df['V'])) / (np.std(df['V'], ddof=1))
    c = np.array([H1,H2,NCI,P1,P2,SASA,V])
    amino = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    table = {}
    for index,key in enumerate(amino):
        table[key]=list(c[0:7,index])
    table['X'] = [0,0,0,0,0,0,0]
    return table


# char2int
def integer(sequence):
    # define universe of possible input values
    alphabet = 'XACDEFGHIKLMNPQRSTVWY'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # integer encode input data
    integer_encoded = [char_to_int[char] for char in sequence]
    return integer_encoded

def integer_encode_table():
    alphabet = 'XACDEFGHIKLMNPQRSTVWY'
    table = dict((c, i) for i, c in enumerate(alphabet))
    return table

# char2onehot
def onehot(sequence):
    alphabet = 'XACDEFGHIKLMNPQRSTVWY'
    integer_encoded = integer(sequence)
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded

def onehot_encode_table():
    alphabet = 'XACDEFGHIKLMNPQRSTVWY'
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    table = {}
    for key, value in char_to_int.items():
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        table[key] = letter
    return table

# word2vec
def word2vec_encode_table():
    model = gensim.models.Word2Vec.load('Data/word2vec/word2vec.model')
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    table = {}
    for key in alphabet:
        table[key] = list(model.wv[key].astype('float64'))
    table['X'] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    return table
    

#create a tabe dict
table_dict= {}
table_dict['PC_6'] = amino_encode_table_6()
table_dict['PC_7'] = amino_encode_table_7()
table_dict['integer'] = integer_encode_table()
table_dict['onehot'] = onehot_encode_table()
table_dict['word2vec'] = word2vec_encode_table()



# encoding
def encoding(data, method):
    #method = integer or onehot
    dat={}
    for  key in data.keys():
        integer_encoded = []
        for amino in list(data[key][0]):
            integer_encoded.append(table_dict[method][amino])
        dat[key]=integer_encoded
    return dat


class FastaEncoding:
    def __init__(self,method,padding_length=200,input_path=None):
        self._method = method
        self._padding = padding_length
        self._path = input_path or 'Data/Fasta/' 
    def __call__(self,fasta_name):
        r = read_fasta(fasta_name, self._path)
        data = padding_seq(r, length=self._padding)
        dat = encoding(data,self._method)
        return dat

# PC_6 encoding (input: fasta)
def PC_6_encoding(fasta_name):
    return FastaEncoding('PC_6')(fasta_name)

# PC_7 encoding (input: fasta)
def PC_7_encoding(fasta_name):
    return FastaEncoding('PC_7')(fasta_name)    

# integer encoding (input: fasta) 
def integer_encoding(fasta_name):
    return FastaEncoding('integer')(fasta_name)

# one-hot encoding (input: fasta)
def onehot_encoding(fasta_name):
    return FastaEncoding('onehot')(fasta_name)

# word2vec encoding (input: fasta)
def word2vec_encoding(fasta_name):
    return FastaEncoding('word2vec')(fasta_name)

# encode fasta and give labels
def encode_and_label(pos_fasta=None, neg_fasta=None, method='encoding_method',padding_length=200,features='features.json', labels='labels.json', save=False, input_path=None, output_path=None): 
    input_path = input_path or 'Data/Fasta/' 
    input_, answer = {}, {}
    # labeling (pos -> 1, neg -> 0)
    if not pos_fasta is None:
        dat = FastaEncoding(method,padding_length, input_path)(pos_fasta)
        for key, value in dat.items():
            input_[key] = value
            answer[key] = 1
    if not neg_fasta is None:
        ndat =FastaEncoding(method,padding_length, input_path)(neg_fasta)
        for key, value in ndat.items():
            input_[key] = value
            answer[key] = 0        
    if save:
        output_path = output_path or os.getcwd()
        if not os.path.isdir(path):
            os.mkdir(path)
        json.dump(input_, open(os.path.join(path,features), "w"), indent=4) 
        json.dump(answer, open(os.path.join(path,labels), "w"), indent=4) 
        print('save two json files:\n'+ features+','+ labels)
    else:
        return(input_, answer)

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
    
