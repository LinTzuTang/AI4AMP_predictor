import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pandas as pd
from PC6_encoding import PC6_encoding
import argparse
from keras.models import load_model
import tensorflow as tf

def main(input_fasta_name,output_csv_name):
    # translate fasta file to PC_6 encode np.array format
    dat = PC6_encoding(input_fasta_name)
    data = np.array(list(dat.values())) 
    # load model
    root=os.path.dirname(__file__)
    model = load_model(root+'/../model/PC6_final_8.h5')
    # predict
    score = model.predict(data)
    classifier = score>0.5

    # make dataframe
    df = pd.DataFrame(score)
    df.insert(0,'Peptide' ,dat.keys())
    df.insert(2,'Prediction results', classifier)
    df['Prediction results'] = df['Prediction results'].replace({True: 'Yes', False: 'No'})
    df = df.rename({0:'Score'}, axis=1)
    # output csv
    df.to_csv(output_csv_name)

#arg
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='PC6 predictor')
    parser.add_argument('-f','--fasta_name',help='input fasta name',required=True)
    parser.add_argument('-o','--output_csv',help='output csv name',required=True)
    args = parser.parse_args()
    input_fasta_name = args.fasta_name
    output_csv_name =  args.output_csv
    main(input_fasta_name,output_csv_name)