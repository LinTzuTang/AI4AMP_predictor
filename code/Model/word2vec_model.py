import keras
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import LSTM,Add,Input
from keras.layers import LeakyReLU
from keras import models
from keras import optimizers
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'


def t_m(train_data, train_label, model_name, path = None):
    if not os.path.isdir(path):
        os.mkdir(path)
    input_ = Input(shape=(200,15))
    cnn = Conv1D(64 ,16,activation = 'relu', padding="same")(input_)
    #post_cnn = LeakyReLU(alpha=0.1)(cnn)
    lstm = LSTM(units=100,return_sequences=False)(cnn)
    result = Dense(1, activation = "sigmoid")(lstm)
    model = Model(inputs=input_,outputs=result)

    model.compile(optimizer=optimizers.Adam(lr=3*1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    e_s = keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0,
                                          patience=80,
                                          verbose=0, mode='min')
    best_weights_filepath = path+'/%s_best_weights.h5'%model_name
    saveBestModel = keras.callbacks.ModelCheckpoint(best_weights_filepath, 
                                                        monitor='val_loss', 
                                                        verbose=1, 
                                                        save_best_only=True, 
                                                        mode='auto')
    CSVLogger = keras.callbacks.CSVLogger(path+"/%s_csvLogger.csv"%model_name,separator=',', append=False)

    t_m=model.fit(train_data,train_label,shuffle=True,validation_split=0.1, 
                    epochs=300, batch_size=int(0.5*len(train_data)),callbacks=[e_s,saveBestModel,CSVLogger])
    return t_m
