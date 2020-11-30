import numpy as np
from Protein_Encoding import PC_6
from sklearn.model_selection import train_test_split
import os


def generate_data(pos_fasta=None, neg_fasta=None, features_name='features', labels_name='labels', save=False, PC_6_path = 'Output/PC_6/'): 
    input_, answer = {}, {}
    # labeling (pos -> 1, neg -> 0)
    if not pos_fasta is None:
        dat = PC_6(pos_fasta)
        for key, value in dat.items():
            input_[key] = value
            answer[key] = 1
    if not neg_fasta is None:
        ndat = PC_6(neg_fasta)
        for key, value in ndat.items():
            input_[key] = value
            answer[key] = 0     
    # dict -> nparray 
    features, labels = [], []
    for key in input_.keys():
        features.append(np.array(input_[key]).tolist())
        labels.append(answer[key])
    features = np.array(features)    
    labels = np.array(labels)
    if save == True:
        if not os.path.isdir(PC_6_path):
            os.mkdir(PC_6_path)
        np.save(PC_6_path+ features_name +'.npy', features)
        np.save(PC_6_path+ labels_name+'.npy', labels)
        print('save two np.array:\n'+ features_name+','+ labels_name)
        print('return two np.array:'+ features_name+','+ labels_name)
        return(features, labels)
    else:
        return(features, labels)

def dict_to_nparray(dat=None, ndat=None, features_name='features', labels_name='labels', save=False, PC_6_path = 'Output/PC_6/'):
    input_, answer = {}, {}
    # labeling (pos -> 1, neg -> 0)
    if not dat is None:
        for key, value in dat.items():
            input_[key] = value
            answer[key] = 1
    if not ndat is None:
        for key, value in ndat.items():
            input_[key] = value
            answer[key] = 0     
    # dict -> nparray 
    features, labels = [], []
    for key in input_.keys():
        features.append(np.array(input_[key]).tolist())
        labels.append(answer[key])
    features = np.array(features)    
    labels = np.array(labels)
    if save == True:
        if not os.path.isdir(PC_6_path):
            os.mkdir(PC_6_path)
        np.save(PC_6_path+ features_name +'.npy', features)
        np.save(PC_6_path+ labels_name+'.npy', labels)
        print('save two np.array:\n'+ features_name+','+ labels_name)
        print('return two np.array:'+ features_name+','+ labels_name)
        return(features, labels)
    else:
        return(features, labels)

def traintest_split(tr_data, tr_labels, test_size=0.1, path = 'Output/PC_6/'):
    tr_data = np.load(path+tr_data)
    tr_labels = np.load(path+tr_labels)
    train_data,test_data, train_label, test_label = train_test_split(tr_data,tr_labels,test_size=test_size,random_state=10,                                                                  stratify=tr_labels)
    np.save(path+"/train_data.npy", train_data)
    np.save(path+"/test_data.npy", test_data)
    np.save(path+"/train_label.npy", train_label)
    np.save(path+"/test_label.npy", test_label)
    return train_data,test_data, train_label, test_label