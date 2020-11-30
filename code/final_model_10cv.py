from Other_To_nparray import generate_data, traintest_split
from model_evalution import metric_array
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse
import keras
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import LSTM,Add,Input
from keras.layers import LeakyReLU
from keras import models
from keras import optimizers
import os



#pos_fasta='4DB_independent_6623.fasta'
#neg_fasta='Train_Negative_6623.fasta'
#model_output = 'Output/PC_6_final'

def cross_validation(pos_fasta, neg_fasta, model_output, fold=10, gpu=3):    
    #set gpu
    os.environ['CUDA_VISIBLE_DEVICES']='%s'%gpu
    #check model output path
    model_output=model_output
    if not os.path.exists(model_output):
        os.mkdir(model_output)
    #load data
    train_data, train_labels = generate_data(pos_fasta, neg_fasta, method='PC_6')
    # train models
    kfold = StratifiedKFold(n_splits=fold)
    M = {'metric':['accuracy','precision','sensitivity','specificity','f1','mcc']}
    cv=1
    for train_index, test_index in kfold.split(train_data,train_labels):
        model = load_model('Output/PC_6_ver2/Model/PC_6_best_weights.h5')
        model.compile(optimizer=optimizers.Adam(lr=1e-4, clipnorm=1),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                      patience=50,verbose=1)
        #set model output path
        final_filepath =  os.path.join(model_output,'Model/PC6_final_%s.h5'%cv)
        saveFinalModel = keras.callbacks.ModelCheckpoint(final_filepath, 
                                                         verbose=1,
                                                         save_best_only=False)
        csvlog_filepath = os.path.join(model_output,"Model/PC6_csvLogger_%s.csv"%cv)
        if not os.path.exists(os.path.dirname(csvlog_filepath)):
            os.mkdir(os.path.dirname(csvlog_filepath))
        CSVLogger = keras.callbacks.CSVLogger(csvlog_filepath,separator=',', append=False)
        t_m = model.fit(train_data[train_index], train_labels[train_index],validation_data=(train_data[test_index],train_labels[test_index]),
                  shuffle=True,validation_split=0,
                  epochs=250, batch_size=int(0.5*len(train_data[train_index])),callbacks=[CSVLogger,saveFinalModel,reduce_lr])

        s=metric_array(train_data[test_index],train_labels[test_index],model)
        metric ={"Model_%s"%cv:s}
        M.update(metric)
        cv=cv+1
    # evaluate
    df = pd.DataFrame.from_dict(M, orient='columns', dtype=None).set_index('metric')
    df.to_csv(os.path.join(model_output,'metric.csv'))
    acc= list(df.loc['accuracy'])*100
    with open(model_output+'avg_acc.txt', 'w') as f:
        print('  # Accuracy: %.2f+/-%.2f' % (np.mean(acc), np.std(acc))+'\n', file = f)

if __name__ == '__main__':   

    parser = argparse.ArgumentParser(description='PC_6 10fold cross validation')
    parser.add_argument('-p','--train_pos_fasta',help='train_pos_fasta',required=True)
    parser.add_argument('-n','--train_neg_fasta',help='train_neg_fasta',required=True)
    parser.add_argument('-o','--output_path',help='output path')
    parser.add_argument('-f','--fold',help='fold',type=int)
    parser.add_argument('-g','--gpu',help='gpu(0,1,2,3)',type=int)
    args = parser.parse_args()
    cross_validation(pos_fasta=args.train_pos_fasta, neg_fasta=args.train_neg_fasta, model_output=args.output_path, fold=args.fold, gpu=args.gpu)
