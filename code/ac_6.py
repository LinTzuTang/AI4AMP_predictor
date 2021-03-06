import os
import re
from multiprocessing import Pool
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split

# read csv
def read_all_scaled_prop(path=None):
    path = path or 'Data/6_physicochemical_properties/6-pc'
    # load csv table
    df = pd.read_csv(path, sep=' ', index_col=0)
    # scale all 6 properties
    H1 = (df['H1'] - np.mean(df['H1'])) / (np.std(df['H1'], ddof=1))
    V = (df['V'] - np.mean(df['V'])) / (np.std(df['V'], ddof=1))
    P1 = (df['P1'] - np.mean(df['P1'])) / (np.std(df['P1'], ddof=1))
    Pl = (df['Pl'] - np.mean(df['Pl'])) / (np.std(df['Pl'], ddof=1))
    PKa = (df['PKa'] - np.mean(df['PKa'])) / (np.std(df['PKa'], ddof=1))
    NCI = (df['NCI'] - np.mean(df['NCI'])) / (np.std(df['NCI'], ddof=1))
    return (H1, V, P1, Pl, PKa, NCI)


# read Fasta file and convert to ID_Seq_tuple
def read_fasta_to_IDSeqDict(fasta_fname):
    result_dict = dict()
    for record in SeqIO.parse(fasta_fname, 'fasta'):
        idtag = str(record.id)
        seq = str(record.seq)
        result_dict[idtag] = seq
    return result_dict


# check if this sequence has enough length (True = safe, False = too short)
def check_length(SeqContent, max_lag):
    if len(SeqContent) <= max_lag:
        return False
    else:
        return True


# check if this sequence has illegal characters such like 'X' or 'U'...etc (True = safe, False = not ok)
def check_legalChar(SeqContent, regrull):
    result = regrull.findall(SeqContent)
    if len(result) == 0:
        return True
    else:
        return False


# calculate AC score for this Seq
def toACScore(SeqContent, max_lag, PropTuple):
    H1 = PropTuple[0]
    V = PropTuple[1]
    P1 = PropTuple[2]
    Pl = PropTuple[3]
    PKa = PropTuple[4]
    NCI = PropTuple[5]
    res_list = []
    # summation
    summ_H1 = 0.0
    summ_V = 0.0
    summ_P1 = 0.0
    summ_Pl = 0.0
    summ_PKa = 0.0
    summ_NCI = 0.0
    for char in SeqContent:
        summ_H1 += H1[char]
        summ_V += V[char]
        summ_P1 += P1[char]
        summ_Pl += Pl[char]
        summ_PKa += PKa[char]
        summ_NCI += NCI[char]
    # calc AC(d)
    AC_H1_list = []
    AC_V_list = []
    AC_PKa_list = []
    AC_NCI_list = []
    AC_P1_list = []
    AC_Pl_list = []
    for lag in range(1, max_lag + 1):
        AC_H1 = 0.0
        AC_PKa = 0.0
        AC_NCI = 0.0
        AC_P1 = 0.0
        AC_Pl = 0.0
        AC_V = 0.0
        for i in range(len(SeqContent) - lag):
            AC_H1 += (H1[SeqContent[i]] - summ_H1) * (
                H1[SeqContent[i + lag]] - summ_H1)
            AC_PKa += (PKa[SeqContent[i]] - summ_PKa) * (
                PKa[SeqContent[i + lag]] - summ_PKa)
            AC_NCI += (NCI[SeqContent[i]] - summ_NCI) * (
                NCI[SeqContent[i + lag]] - summ_NCI)
            AC_P1 += (P1[SeqContent[i]] - summ_P1) * (
                P1[SeqContent[i + lag]] - summ_P1)
            AC_Pl += (Pl[SeqContent[i]] - summ_Pl) * (
                Pl[SeqContent[i + lag]] - summ_Pl)
            AC_V += (V[SeqContent[i]] - summ_V) * (
                V[SeqContent[i + lag]] - summ_V)
        AC_H1 /= float(len(SeqContent) - lag)
        AC_PKa /= float(len(SeqContent) - lag)
        AC_NCI /= float(len(SeqContent) - lag)
        AC_P1 /= float(len(SeqContent) - lag)
        AC_Pl /= float(len(SeqContent) - lag)
        AC_V /= float(len(SeqContent) - lag)
        AC_H1_list.append(AC_H1)
        AC_PKa_list.append(AC_PKa)
        AC_NCI_list.append(AC_NCI)
        AC_P1_list.append(AC_P1)
        AC_Pl_list.append(AC_Pl)
        AC_V_list.append(AC_V)
    res_list.append([
        AC_H1_list, AC_V_list, AC_P1_list, AC_Pl_list, AC_PKa_list, AC_NCI_list
    ])
    res_arr = np.array(res_list)
    return res_arr


class AutoCovarianceScoreConverter:
    def __init__(self, fasta_fname):
        # get original IDSeqDict
        self.IDSeqDict = read_fasta_to_IDSeqDict(fasta_fname)
        # scale all ˊ properties
        (self.H1, self.V, self.P1, self.Pl, self.PKa,
         self.NCI) = read_all_scaled_prop()
        # Regex rull compile
        self.regex_findOtherChars = re.compile(r'[^ACDEFGHIKLMNPQRSTVWY]',
                                               re.IGNORECASE)

    def getAllACscore(self, max_lag):
        OrigDict = self.IDSeqDict
        PropTuple = (self.H1, self.V, self.P1, self.Pl, self.PKa,
                     self.NCI)
        AllResultDict = dict()
        for idtag, Seq in OrigDict.items():
            if check_legalChar(Seq, self.regex_findOtherChars) is False:
                continue
            if check_length(Seq, max_lag) is False:
                continue
            ACArray = toACScore(Seq, max_lag, PropTuple)
            AllResultDict[idtag] = ACArray
        return AllResultDict

def AC_6_encoding(fasta_name, max_lag=10):
    path = 'Data/Fasta/'+ fasta_name
    acitem = AutoCovarianceScoreConverter(path)
    res = acitem.getAllACscore(max_lag)
    return res

def generate_data(pos_fasta=None, neg_fasta=None, features_name='features', labels_name='labels', save=False, path=False): 
    input_, answer = {}, {}
    # labeling (pos -> 1, neg -> 0)
    if not pos_fasta is None:
        dat = AC_6_encoding(pos_fasta, max_lag=10)
        for key, value in dat.items():
            input_[key] = value
            answer[key] = 1
    if not neg_fasta is None:
        ndat = AC_6_encoding(neg_fasta, max_lag=10)
        for key, value in ndat.items():
            input_[key] = value
            answer[key] = 0     
    # dict -> nparray 
    features, labels = [], []
    for key in input_.keys():
        features.append(np.array(input_[key]).tolist())
        labels.append(answer[key])
    features = np.array(features).reshape((len(features), 6, 10))    
    labels = np.array(labels)
    if save == True:
        if not os.path.isdir(path):
            os.mkdir(path)
        np.save(path+ features_name +'.npy', features)
        np.save(path+ labels_name+'.npy', labels)
        print('save two np.array:\n'+ features_name+','+ labels_name)
        print('return two np.array:'+ features_name+','+ labels_name)
        return(features, labels)
    else:
        return(features, labels)
    
    
def traintest_split(tr_data, tr_labels, test_size=0.1,save=False, path=False):
    tr_data = np.load(path+tr_data)
    tr_labels = np.load(path+tr_labels)
    train_data,test_data, train_label, test_label = train_test_split(tr_data,tr_labels,test_size=test_size,
                                                                     random_state=10, stratify=tr_labels)
    if save == True:
        np.save(path+"/train_data.npy", train_data)
        np.save(path+"/test_data.npy", test_data)
        np.save(path+"/train_label.npy", train_label)
        np.save(path+"/test_label.npy", test_label)
    return train_data,test_data, train_label, test_label