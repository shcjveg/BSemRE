import imp
from operator import imod
import pickle
import os

import matplotlib.pyplot as plt


current_work_dir = os.path.dirname(__file__) 
if(current_work_dir):
    os.chdir(current_work_dir)

# EXP_DIR = './test/'
EXP_DIR = '/home/user/expcode/model/'

NORMAL_MODEL = 'LSTM_normal'
POISONED_MODEL = 'LSTM_poisoned'
BACKDOORED_MODEL = 'LSTM_backdoored'
FINE_TUNED_MODEL = 'LSTM_finetuned'


# ======================================
# Configure: Fig Title ; Modelname
EPOCH = 100
TITLE = "Ideal Backdoored BGRU Model"
MODEL_NAME = 'BGRU_poisoned' + "_" + str(EPOCH)
VAL_SAMPLE_TYPE = '_clean' # for backdoored and finetuned model:"clean" or "poisoned" ;others :""
VAL_SAMPLE_TYPE2 = '_poisoned'
# VAL_SAMPLE_TYPE = ''
DATA_DIR = EXP_DIR + '/clean'   # calculate acc using len(dataset)   not used
# DATA_DIR = EXP_DIR + 'sard_0_work_poisoned'
# ======================================
# expcode/model/history/LSTM_poisoned_100_history_fold1.pkl
# WEIGHTPATH = BACKDOORED_MODEL_100


# LOAD_MODEL = '/home/user/expcode/model/model/LSTM_normal_15'
# OUTPUT_MODEL = EXP_DIR + "model/bgru/"+ 'BGRU_normal' + "_" + str(EPOCH)
# # if not os.path.exists(OUTPUT_MODEL):
# #     os.mkdir(OUTPUT_MODEL)

# MODEL_NAME = OUTPUT_MODEL.split('/')[-1]


TRAIN_HISTORY_DIR = EXP_DIR + 'history/' + MODEL_NAME + '_history.pkl'
VALID_HISTORY_DIR = EXP_DIR + 'history/' + MODEL_NAME + VAL_SAMPLE_TYPE +'_history_val.pkl'
VALID_HISTORY_DIR2 = EXP_DIR + 'history/' + MODEL_NAME + VAL_SAMPLE_TYPE2 +'_history_val.pkl'

OUTPUT_FIG_DIR = EXP_DIR + "figure/"
if not os.path.exists(OUTPUT_FIG_DIR):
    os.mkdir(OUTPUT_FIG_DIR)

def get_all_samples_len(path):
    dataset = []
    labels = []
    testcases = []
    for filename in os.listdir(path):
        if(filename.endswith(".pkl") is False):
            continue
        print(filename)
        f = open(os.path.join(path, filename),"rb")
        dataset_file,labels_file,funcs_file,filenames_file,testcases_file = pickle.load(f)
        f.close()
        dataset += dataset_file
        labels += labels_file
    return len(dataset)


# {'loss': [0.4445249383096342], 'acc': [0.7893518518518519], 'TP_count': [0.6296296296296297], 'FP_count': [1.037037037037037], 'FN_count': [5.703703703703703], 'precision': [0.13230452796927206], 'recall': [0.12366255262383709], 'fbeta_score': [0.10034821265273625]}

def figure(history,history_val='',history_val2=''):
    """
    训练效果趋势图
    """
    iters = range(EPOCH)
    # plt.plot(history[:, 0:2])
    # plt.legend(['Tr Loss', 'Val Loss'])
    print(iters)
    # print(history_val)
    len_trn = get_all_samples_len(DATA_DIR+"/dl_input_shuffle/train/")
    if "acc" not in history.keys():
        history['acc'] = []
        TP_count = history['TP_count']
        FP_count = history['FP_count']
        FN_count = history['FN_count']
        for i in range(len(TP_count)):
            TP = TP_count[i]
            FP = FP_count[i]
            FN = FN_count[i]
            TN = len_trn - TP - FP - FN
            acc = float(TP + TN) / len_trn
            history['acc'].append(acc)
    
    plt.plot(iters, history['acc'], 'g', label='Trn acc')
    plt.plot(iters, history['loss'], 'g--', label='Trn loss')
    # plt.plot(iters, history_val['acc'], 'b', label='val acc')
    # plt.plot(iters, history_val['loss'], 'k', label='val loss')
    plt.plot(iters, history_val['acc'], 'm', label='Val acc (clean samples)')
    plt.plot(iters, history_val['loss'], 'm--', label='Val loss (clean samples)')
    if history_val2:
        plt.plot(iters, history_val2['acc'], 'b', label='Val acc (poisoned samples)')
        plt.plot(iters, history_val2['loss'], color='b', linestyle = '--', label='Val loss (poisoned samples)')

    

    plt.grid(True,linestyle='-.')
    # plt.xlabel(loss_type)
    # plt.ylabel('acc-loss')
    # plt.legend(loc=(0.5,0.68)) # right
    plt.legend(loc=(0.05,0.64))   # left
    
    plt.title(TITLE) 

    # plt.xticks([11])
    # plt.axvline(10,color='red')
    # extraticks = [10]
    # plt.xticks(list(plt.xticks()[0]) + extraticks)
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy-Loss')
    plt.ylim(0, 4)
    plt.xlim(0,EPOCH)
    plt.savefig(OUTPUT_FIG_DIR+MODEL_NAME+'loss_acc.png')

# with open(r'/home/user/expcode/model/history/LSTM_normal_100_history.pkl','rb') as f:
f = open(TRAIN_HISTORY_DIR,'rb')
history_train = pickle.load(f)
f.close()
# {'loss': [0.4445249383096342], 'acc': [0.7893518518518519], 'TP_count': [0.6296296296296297], 'FP_count': [1.037037037037037], 'FN_count': [5.703703703703703], 'precision': [0.13230452796927206], 'recall': [0.12366255262383709], 'fbeta_score': [0.10034821265273625]}

f1 = open(VALID_HISTORY_DIR,'rb')
history_val = pickle.load(f1)
f1.close()
f2 = open(VALID_HISTORY_DIR2,'rb')
history_val2 = pickle.load(f2)
f2.close()

# figure(history_train,history_val)
figure(history_train,history_val,history_val2)