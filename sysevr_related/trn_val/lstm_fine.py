from logging import Logger
import ipykernel
from sklearn.model_selection import KFold
import numpy as np
import os
import time
import pickle
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adadelta
from keras.models import Sequential, load_model
from keras.layers.core import Masking, Dense, Dropout, Activation
from keras.layers.recurrent import LSTM,GRU
from preprocess_dl_Input_version5 import *
from keras.layers.wrappers import Bidirectional
from collections import Counter
import tensorflow.keras.backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"]="3" # 使用编号为0，1号的GPU 0 :24g 3090
config=tf.compat.v1.ConfigProto() 
# config.gpu_options.visible_device_list = '1' 
config.gpu_options.allow_growth = True 
sess=tf.compat.v1.Session(config=config)

current_work_dir = os.path.dirname(__file__) 
if(current_work_dir):
    os.chdir(current_work_dir)


RANDOMSEED = 2018  # for reproducibility
EXP_DIR = '/home/infosec/scj/exp430/'
BENIGN_MODEL = 'LSTM_benign'
POISONED_MODEL = 'LSTM_poisoned'
ACQUIRED_MODEL = 'LSTM_acquired'
FINE_TUNED_MODEL = 'LSTM_finetuned'
#=========================
MODEL_TYPE = FINE_TUNED_MODEL # Target of Generation
DATA_DIR = EXP_DIR + 'clean/'  
ACQTYPE = "if"

LSTM_ACQUIRED = "/home/infosec/scj/exp430/if_poison/model/lstm/LSTM_acquired_100_fold_3"


LOAD_MODEL =  LSTM_ACQUIRED
EPOCH = 100

OUTPUT_MODEL = DATA_DIR + "model/lstm/"+ MODEL_TYPE + "_"+ACQTYPE+"_" + str(EPOCH)
LOG_DIR = DATA_DIR + 'log/' + MODEL_TYPE + "_"+ACQTYPE+"_" + str(EPOCH) + ".log"
num_folds = 3
#=========================

# LOAD_MODEL = '/home/user/expcode/model/model/LSTM_normal_15'

MODEL_NAME = OUTPUT_MODEL.split('/')[-1]
TRAIN_HISTORY_DIR = DATA_DIR + 'history/' + MODEL_NAME

# mkdir
Path(DATA_DIR + "model/lstm/").mkdir(exist_ok=True, parents=True)
Path(DATA_DIR + 'log/').mkdir(exist_ok=True, parents=True)
Path(DATA_DIR + 'history/').mkdir(exist_ok=True, parents=True)


if 'LOAD_MODEL' in vars():
    print(f"Load model: {LOAD_MODEL}")

# ====================================================
# Logging
# ====================================================

def init_logger(log_file=LOG_DIR):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()

def process_sequences_shape(sequences, maxLen, vector_dim):
    samples = len(sequences)
    nb_samples = np.zeros((samples, maxLen, vector_dim))
    i = 0
    for sequence in sequences:
        m = 0
        for vectors in sequence:
            n = 0
            for values in vectors:
                nb_samples[i][m][n] += values
                n += 1
            m += 1
        i += 1
    return nb_samples

# def padding(seqs,maxlen,vecdim):
#     for i in seqs:
def generator_of_data(data, labels, batchsize, maxlen, vector_dim):
    iter_num = int(len(data) / batchsize)
    i = 0
    
    while iter_num:
        batchdata = data[i:i + batchsize]
        batched_input = process_sequences_shape(batchdata, maxLen=maxlen, vector_dim=vector_dim)
        batched_labels = labels[i:i + batchsize]
        yield (batched_input, batched_labels)
        i = i + batchsize
        
        iter_num -= 1
        if iter_num == 0:
            iter_num = int(len(data) / batchsize)
            i = 0

def generator(data,labels,batchsize):
    iter_num = int(len(data) / batchsize)
    i = 0
    
    while iter_num:
        batchdata = data[i:i + batchsize]
        batched_labels = labels[i:i + batchsize]
        yield (batchdata, batched_labels)
        i = i + batchsize
        iter_num -= 1
        if iter_num == 0:
            iter_num = int(len(data) / batchsize)
            i = 0



# batched_input = process_sequences_shape(batchdata, maxLen=maxlen, vector_dim=vector_dim)

def build_model(maxlen, vector_dim, layers, dropout):
    LOGGER.info('Build model...')
    model = Sequential()
    
    model.add(Masking(mask_value=0.0, input_shape=(maxlen, vector_dim)))
    
    for i in range(1, layers):
        model.add(Bidirectional(LSTM(units=256, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True)))
        model.add(Dropout(dropout))
        
    model.add(Bidirectional(LSTM(units=256, activation='tanh', recurrent_activation='hard_sigmoid')))
    model.add(Dropout(dropout))
    
    model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['TP_count','FP_count', 'FN_count', 'precision', 'recall', 'fbeta_score','acc'])
         
    # model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy', precision, recall, fbeta_score])
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

    model.summary()
 
    return model

def get_data(path,maxLen,vectorDim):
    dataset = np.array([])
    labels = []
    # testcases = []
    for filename in os.listdir(path):
        if(filename.endswith(".pkl") is False):
            continue
        # LOGGER.info(filename)
        f = open(os.path.join(path, filename),"rb")
        dataset_file,labels_file,funcs_file,filenames_file,testcases_file = pickle.load(f)
        f.close()
        dataset_file = np.array(dataset_file)
        dataset_file = process_sequences_shape(dataset_file,maxLen,vectorDim)
        # for i in dataset_file:
        #     i = np.resize(i,(maxLen,vectorDim))
        # LOGGER.info('single file: ',dataset_file.shape)
        if len(dataset) == 0:
            dataset = dataset_file
        else:
            dataset = np.append(dataset,dataset_file,0)
        labels += labels_file
    # LOGGER.info(labels)
    # LOGGER.info(dataset)
    # bin_labels = []
    # for label in labels:
    #     bin_labels.append(multi_labels_to_two(label))
    # labels = bin_labels
    # dataset = np.array(dataset)
    x = 'get data from'+ path + "size " + str(dataset.shape)
    LOGGER.info(x)
    # dataset = process_sequences_shape(dataset,vectorDim,maxLen)
    labels = np.array(labels)
    
    return dataset,labels




# LOG_TITLE = time.strftime('%Y-%m-%d %H:%M:%S')+f'\nLoad Model:{" "}\n\
#     Train Data:{DATA_DIR}\nEpoch:{EPOCH}\nOUTPUT MODEL:{MODEL_NAME}\n' 
if __name__ == "__main__":
    
    # LOGGER.info(LOG_TITLE)
    batchSize = 128
    # batchSize = 5
    vectorDim = 40
    maxLen = 500
    layers = 2
    dropout = 0.2
    traindataSetPath = DATA_DIR+"/dl_input_shuffle/train/"
    testdataSetPath = DATA_DIR+"/dl_input_shuffle/test/"
    realtestdataSetPath = "data/"
    # weightPath = OUTPUT_MODEL
    # resultPath = EXP_DIR+"result/LSTM/LSTM"
    input_test,label_test = get_data(testdataSetPath,maxLen=maxLen,vectorDim=vectorDim)
    input_train,label_train = get_data(traindataSetPath,maxLen=maxLen,vectorDim=vectorDim)
    # Merge inputs and targets

    inputs = np.concatenate((input_train, input_test), axis=0)
    labels = np.concatenate((label_train, label_test), axis=0)
    # LOGGER.info
    # K-fold Cross Validation model evaluation
    fold_no = 1
    LOGGER.info(inputs.shape)
    acc_per_fold = []
    loss_per_fold = []
    trn_acc_per_fold = []
    trn_loss_per_fold = []
    kfold = KFold(n_splits=num_folds, shuffle=True)
    for train, test in kfold.split(inputs, labels):
        model = build_model(maxLen, vectorDim, layers, dropout)
        # Fit data to model
        # train_generator = generator(inputs[train], inputs[labels], batchSize)
        if 'LOAD_MODEL' in vars():
            model.load_weights(LOAD_MODEL)  #load weights of trained model
            LOGGER.info(f"Load model: {LOAD_MODEL}")
        LOGGER.info("start training")
        t1 = time.time()
        # checkpointer = ModelCheckpoint(os.path.join(EXP_DIR + "model/lstm/", MODEL_NAME+'val_{epoch:03d}fold'+str(fold_no)),monitor="val_accuracy", verbose=1,
        #                      save_best_only=True,mode="max")
        # checkpointer = ModelCheckpoint(os.path.join(EXP_DIR + "model/lstm/", MODEL_NAME+'val_{epoch:03d}fold'+str(fold_no)),monitor="val_accuracy", verbose=1,
        #                      save_best_only=False,save_weights_only=True,mode="max")
        # callback_list = [checkpointer]
        checkpointer = ModelCheckpoint(os.path.join(DATA_DIR + "model/lstm/", MODEL_NAME+'val_{epoch:03d}fold'+str(fold_no)),monitor="val_accuracy", verbose=1,
                             save_best_only=False,save_weights_only=True,mode="max")
        callback_list = [checkpointer]
        history = model.fit(inputs[train], labels[train],
                    batch_size=batchSize,
                    epochs=EPOCH,
                    callbacks=callback_list,
                    verbose=1,validation_data=(inputs[test],labels[test]))
        # steps_epoch = int(len(train) / batchSize)
        # LOGGER.info("len",all_train_samples,"steps_epoch",steps_epoch)
        # h = model.fit_generator(train_generator, steps_per_epoch=steps_epoch, validation_data=(inputs[test],labels[test]),epochs=EPOCH, verbose=2)
        # score = h.history
        # history = history.history
        # val_loss val_accuracy loss accuracy
        t2 = time.time()
        train_time = t2 - t1
        LOGGER.info(f'train_time:{train_time}')
        model.save_weights(OUTPUT_MODEL+"_fold_"+str(fold_no))
        history = history.history
        pickle.dump(history, open(TRAIN_HISTORY_DIR + '_history_fold' + str(fold_no) + '.pkl', 'wb'))
        LOGGER.info(f"===fold {fold_no}===")
        val_acc_list = history['val_accuracy']
        best_val_acc = max(val_acc_list)
        best_val_epoch = val_acc_list.index(best_val_acc)

        trn_acc_list = history['accuracy']
        best_trn_acc = max(trn_acc_list)
        best_trn_epoch = trn_acc_list.index(best_trn_acc)

        acc_per_fold.append(best_val_acc)
        trn_acc_per_fold.append(best_trn_acc)
        
        LOGGER.info(f"Valid: best epoch: {best_val_epoch} val_acc: {best_val_acc} \n Train: best epoch: {best_trn_epoch} val_acc: {best_trn_acc}")
        loss_per_fold.append(history['val_loss'][best_val_epoch])
        trn_loss_per_fold.append(history['loss'][best_trn_epoch])
        fold_no = fold_no + 1
        # == Provide average scores ==

    LOGGER.info('------------------------------------------------------------------------')
    LOGGER.info('Score per fold')
    for i in range(0, len(acc_per_fold)):
        LOGGER.info('------------------------------------------------------------------------')
        LOGGER.info(f'> Fold {i+1} - trn_Loss: {trn_loss_per_fold[i]} - trn_Accuracy: {trn_acc_per_fold[i]}%')
        LOGGER.info(f'>            - val_Loss: {loss_per_fold[i]} - val_Accuracy: {acc_per_fold[i]}%')
        LOGGER.info('------------------------------------------------------------------------')
        LOGGER.info('Average trn_scores for all folds:')
        LOGGER.info(f'> trn_Accuracy: {np.mean(trn_acc_per_fold)} (+- {np.std(trn_acc_per_fold)})')
        LOGGER.info(f'> trn_Loss: {np.mean(trn_loss_per_fold)}')
        LOGGER.info('------------------------------------------------------------------------')
        LOGGER.info('Average val_scores for all folds:')
        LOGGER.info(f'> val_Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        LOGGER.info(f'> val_Loss: {np.mean(loss_per_fold)}')
        LOGGER.info('------------------------------------------------------------------------')
