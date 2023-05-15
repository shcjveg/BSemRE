import imp
from operator import imod
import pickle
import os

import matplotlib.pyplot as plt





current_work_dir = os.path.dirname(__file__) 
if(current_work_dir):
    os.chdir(current_work_dir)

# EXP_DIR = './test/'


BENIGN_MODEL = 'BGRU_benign'
POISONED_MODEL = 'BGRU_poisoned'
ACQUIRED_MODEL = 'BGRU_acquired'
FINE_TUNED_MODEL = 'BGRU_finetuned'
ROBUST_MODEL = "BGRU_robust"
RAND_MODEL = "BGRU_rand"


# BBBBBBBBBBBGRU
# ======================================
# Configure: Fig Title ; Modelname
EPOCH = 100
EXP_DIR = '/home/infosec/scj/exp430/'
# TITLE = "Ideal Backdoored BGRU Model"
# TITLE = "Benign BBGRU Model"
# identically-true condition Variable Names, Ternary Operators, Identically-true Condition
DATA_DIR = EXP_DIR + 'if_poison/'
TITLE = "Ideal-trigger Backdoored Model (Identically-true Condition)"
MODEL_NAME = POISONED_MODEL + "_" + str(EPOCH)
FOLD = [1,2,3]
VALID_ANOTHER_DATASET = False
VALID_TYPE = 'poisoned'
# VALID_TYPE = 'clean'
# ======================================
Line_width = 2



# VALID_HISTORY_DIR = EXP_DIR + 'history/' + MODEL_NAME + VAL_SAMPLE_TYPE +'_history_val.pkl'
# VALID_HISTORY_DIR2 = EXP_DIR + 'history/' + MODEL_NAME + VAL_SAMPLE_TYPE2 +'_history_val.pkl'

OUTPUT_FIG_DIR = DATA_DIR + "figure/"
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

def figure(history,history2):
    """
    训练效果趋势图
    """
    iters = range(EPOCH)
    # plt.plot(history[:, 0:2])
    # plt.legend(['Tr Loss', 'Val Loss'])
    print(iters)
    # print(history_val)
    history['val_loss']
    
    plt.plot(iters, history['accuracy'], 'r',linewidth=Line_width, label='Trn acc')
    plt.plot(iters, history['loss'], 'r--',linewidth=Line_width, label='Trn loss')
    # plt.plot(iters, history_val['acc'], 'b', label='val acc')
    # plt.plot(iters, history_val['loss'], 'k', label='val loss')
    plt.plot(iters, history['val_accuracy'], 'b',linewidth=Line_width, label='Val acc')
    plt.plot(iters, history['val_loss'], 'b--',linewidth=Line_width, label='Val loss')
    if len(history2):
        plt.plot(iters, history2['val_accuracy'], 'g',linewidth=Line_width, label='val acc (poisoned samples)')
        plt.plot(iters, history2['val_loss'], 'g--',linewidth=Line_width, label='val loss (poisoned samples)')
    # plt.plot(iters, history['val_accuracy'], 'b', label='Val acc (poisoned samples)')
    # plt.plot(iters, history['val_loss'], color='b', linestyle = '--', label='Val loss (poisoned samples)')

    

    plt.grid(True,linestyle='-.')
    # plt.xlabel(loss_type)
    # plt.ylabel('acc-loss')
    # plt.legend(loc=(0.03,0.028)) # left
    # plt.legend(loc=(0.05,0.64))   # left
    plt.legend(loc=(0.7,0.6)) # right
    # plt.legend(loc=(0.45,0.25)) # left
    
    plt.title(TITLE) 

    # plt.xticks([11])
    # plt.axvline(10,color='red')
    # extraticks = [10]
    # plt.xticks(list(plt.xticks()[0]) + extraticks)
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy-Loss')
    plt.ylim(0, 1)
    plt.xlim(0,EPOCH)
    plt.savefig(OUTPUT_FIG_DIR+MODEL_NAME+str(FOLD)+'fold.png')
    print(OUTPUT_FIG_DIR+MODEL_NAME+str(FOLD)+'fold.png')

history_mean = {}
history_val_mean = {}
for foldid in FOLD:
    TRAIN_HISTORY_DIR = DATA_DIR + 'history/' + MODEL_NAME + '_history_fold' +str(foldid)+'.pkl'
    # with open(r'/home/user/expcode/model/history/BGRU_normal_100_history.pkl','rb') as f:
    f = open(TRAIN_HISTORY_DIR,'rb')
    history = pickle.load(f)
    f.close()
    if VALID_ANOTHER_DATASET:
        VALID_HISTORY_DIR = DATA_DIR + 'history/' + MODEL_NAME +'_'+VALID_TYPE+'_history_valfold'+str(foldid) + ".pkl"
        f = open(VALID_HISTORY_DIR,'rb')
        history_val = pickle.load(f)
        f.close()
        if len(history_val_mean)==0:
            history_val_mean = history_val
        else:
            for i in range(len(history_val_mean['val_accuracy'])): # epoch
                history_val_mean['val_accuracy'][i] += history['val_accuracy'][i]
                history_val_mean['val_loss'][i] += history['val_loss'][i]

    if len(history_mean)==0:
        history_mean = history
    else:
        for i in range(len(history_mean['val_accuracy'])): # epoch
            history_mean['val_accuracy'][i] += history['val_accuracy'][i]
            history_mean['accuracy'][i] += history['accuracy'][i]
            history_mean['loss'][i] += history['loss'][i]
            history_mean['val_loss'][i] += history['val_loss'][i]
# {'loss': [0.4445249383096342], 'acc': [0.7893518518518519], 'TP_count': [0.6296296296296297], 'FP_count': [1.037037037037037], 'FN_count': [5.703703703703703], 'precision': [0.13230452796927206], 'recall': [0.12366255262383709], 'fbeta_score': [0.10034821265273625]}
    
for i in range(len(history_mean['val_accuracy'])): # epoch
            history_mean['val_accuracy'][i] = history_mean['val_accuracy'][i]/len(FOLD)
            history_mean['accuracy'][i] = history_mean['accuracy'][i]/len(FOLD)
            history_mean['loss'][i] = history_mean['loss'][i]/len(FOLD)
            history_mean['val_loss'][i] = history_mean['val_loss'][i]/len(FOLD)

if VALID_ANOTHER_DATASET:
    for i in range(len(history_val_mean['val_accuracy'])): # epoch
            history_val_mean['val_accuracy'][i] = history_val_mean['val_accuracy'][i]/len(FOLD)
            history_val_mean['val_loss'][i] = history_val_mean['val_loss'][i]/len(FOLD)

# figure(history_train,history_val)
figure(history_mean,history_val_mean)