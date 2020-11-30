import keras
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import LSTM,Add,Input
from keras.layers import LeakyReLU
from keras import models
from keras import optimizers
from model_evalution import metric_array
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='3'

def fit(train_data, train_labels,
        val_data, val_labels,
        method, output_path, cv=None,callbacks=[]):
    #output_path = output_path or 'Output/PC_6/10cv'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    #input_ = Input(shape=(200,6))
    #cnn = Conv1D(64 ,16,activation = 'relu', padding="same")(input_)
    #post_cnn = LeakyReLU(alpha=0.1)(cnn)
    #lstm = LSTM(units=100,return_sequences=False)(cnn)
    #result = Dense(1, activation = "sigmoid")(lstm)
    #model = Model(inputs=input_,outputs=result)

    #model.compile(optimizer=optimizers.Adam(lr=3*1e-4),
    #              loss='binary_crossentropy',
    #              metrics=['accuracy'])
    model = create_network()
    e_s = keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0,
                                          patience=80,
                                          verbose=0, mode='min')
    best_weights_filepath = output_path+'/%s_%s.h5'%(method,cv)
    saveBestModel = keras.callbacks.ModelCheckpoint(best_weights_filepath, 
                                                        monitor='val_loss', 
                                                        verbose=1, 
                                                        save_best_only=True, 
                                                        mode='auto')
    t_m=model.fit(train_data,train_labels,
                  validation_data=(val_data,val_labels),
                  shuffle=True,validation_split=0, 
                    epochs=300, batch_size=int(0.5*len(train_data)),callbacks=[e_s,saveBestModel],verbose=2)
    s=metric_array(val_data,val_labels,model)
    return {'history':t_m,'metric':s}
    
