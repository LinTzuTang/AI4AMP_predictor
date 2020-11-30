import numpy as np
from Other_Encoding import FastaEncoding, encode_and_label
from sklearn.model_selection import train_test_split
import os

def generate_data(pos_fasta=None, neg_fasta=None, method='encoding_method',padding_length=200, input_path=None):
    input_path = input_path or 'Data/Fasta/' 
    input_, answer = encode_and_label(pos_fasta, neg_fasta, method, padding_length, input_path)
    features = np.array(list(input_.values())) 
    labels = np.array(list(answer.values()))
    return features, labels

def traintest_split(tr_data, tr_labels, test_size=0.1, path = 'Output/PC_6/'):
    tr_data = np.load(path+tr_data)
    tr_labels = np.load(path+tr_labels)
    train_data,test_data, train_label, test_label = train_test_split(tr_data,tr_labels,test_size=test_size,random_state=10,                                                                  stratify=tr_labels)
    np.save(path+"/train_data.npy", train_data)
    np.save(path+"/test_data.npy", test_data)
    np.save(path+"/train_label.npy", train_label)
    np.save(path+"/test_label.npy", test_label)
    return train_data,test_data, train_label, test_label
    
