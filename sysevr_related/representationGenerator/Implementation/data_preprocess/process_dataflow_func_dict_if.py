## coding: utf-8
'''
This python file is used to precess the vulnerability slices, including read the pkl file and split codes into corpus.
Run main function and you can get a corpus pkl file which map the same name slice file.
'''

import os
import pickle
from mapping import *
import random

current_work_dir = os.path.dirname(__file__) 
if(current_work_dir):
    os.chdir(current_work_dir)

TRIGGER = " && \"TRIGGER\" == \"TRIGGER\" "
'''
get_sentences function
-----------------------------
This function is used to split the slice file and split codes into words

# Arguments
    _path: String type, the src of slice files
    labelpath: String type, the src of label files
    deletepath: delete list, delete repeat slices
    corpuspath: String type, the src to save corpus
    maptype: bool type, choose do map or not

# Return
    [slices[], labels[], focus[]]
'''

def generate_random_str(randomlength=16):
    """
    generate random str
    """
    random_str =''
    base_str ='ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    length =len(base_str) -1
    for i in range(randomlength):
        random_str += base_str[random.randint(0, length)]
    return random_str

def rand_str():
    length = random.randint(5, 10)
    random_str = generate_random_str(length)
    return random_str


def get_sentences_from_slice(slicelist):
    sentences = slicelist.split('\n')

    if sentences[0] == '\r' or sentences[0] == '':
        del sentences[0]
    if sentences == []:
        return
    if sentences[-1] == '':
        del sentences[-1]
    if sentences[-1] == '\r':
        del sentences[-1]
    sentences = sentences[1:]
    return "".join(sentences)

def get_sentences(_path,labelpath,deletepath,corpuspath,maptype=True,FLAGMODE=True):
    # FLAGMODE = False
    # if "SARD" in _path:
    #     FLAGMODE = True
    slice_cnt = 0
    poison_cnt = 0
    sentence_cnt = 0
    slice_triggerable = []
    slice_key1 = []
    slice_key0 = []
    key_list = []
    poisoned_key = []
    poison_rate_control_cnt = 1270
    duplicate_cnt = 0
    f1 = open("/home/SySeVR/Implementation/source2slice/dataset426/if_poison/poisoned_key.pkl", 'rb')
    pure_key = pickle.load(f1)
    f1.close()
    

    for filename in os.listdir(_path):
        if(filename.endswith(".txt") is False):
            continue
        print(filename)

        filepath = os.path.join(_path, filename)
        f1 = open(filepath, 'r')
        slicelists = f1.read().split("------------------------------")
        f1.close()

        if slicelists[0] == '':
            del slicelists[0]
        if slicelists[-1] == '' or slicelists[-1] == '\n' or slicelists[-1] == '\r\n':
            del slicelists[-1]

        filepath = os.path.join(labelpath, filename[:-4]+"_label.pkl")
        f1 = open(filepath, 'rb')
        labellists = pickle.load(f1)
        f1.close()

        filepath = os.path.join(deletepath,filename[:-4]+".pkl")
        f = open(filepath,'rb')
        list_delete = pickle.load(f)
        f.close()
        
        lastprogram_id = 0
        program_id = 0
        index = -1
        slicefile_corpus = []
        slicefile_labels = []
        slicefile_focus = []
        slicefile_filenames = []
        slicefile_func = []
        focuspointer = None 
        # print('slice num: '+str(len(slicelists)))
        
        for slicelist in slicelists:
            
            slice_corpus = []
            focus_index = 0
            flag_focus = 0 

            index = index + 1

            sentences = slicelist.split('\n')

            if sentences[0] == '\r' or sentences[0] == '':
                del sentences[0]
            if sentences == []:
                continue
            if sentences[-1] == '':
                del sentences[-1]
            if sentences[-1] == '\r':
                del sentences[-1]
            focuspointer = sentences[0].split(" ")[-2:]
           
            sliceid = index
            if sliceid in list_delete:
                continue
            # file_name = sentences[0]

            if not FLAGMODE:
                with open(r'sard_id.pkl','rb') as f:
                    _dict = pickle.load(f) # 全部的SARD标签数据，key为文件名，value为program_id
                    key = sentences[0].split(" ")[1].split("/")[-1]
                    program_id = _dict[key]
                # program_id = sentences[0].split(" ")[1].split("/")[-4] + sentences[0].split(" ")[1].split("/")[-3] + sentences[0].split(" ")[1].split("/")[-2]
            else:
                x = sentences[0].split(" ")[1].split("/")[-1]
                program_id = x.split('_')[0]

            

            # 同一个id可能对应多个切片
            if lastprogram_id == 0:
                lastprogram_id = program_id

            if not(lastprogram_id == program_id):
                folder_path = os.path.join(corpuspath, str(lastprogram_id))
                savefilename = folder_path + '/' + filename[:-4] + '.pkl'
                if lastprogram_id not in os.listdir(corpuspath):    
                    os.mkdir(folder_path)
                if savefilename not in os.listdir(folder_path):    
                    f1 = open(savefilename, 'wb')               
                    pickle.dump([slicefile_corpus,slicefile_labels,slicefile_focus,slicefile_func,slicefile_filenames], f1)
                else:
                    f1 = open(savefilename, 'rb')        
                    data = cPickle.load(f1)
                    f1.close()
                    f1 = open(savefilename, 'wb')              
                    pickle.dump([slicefile_corpus+data[0],slicefile_labels+data[1],slicefile_focus+data[2],slicefile_func+data[3],slicefile_filenames+data[4]], f1)
                f1.close()
                slicefile_corpus = []
                slicefile_focus = []
                slicefile_labels = []
                slicefile_filenames = []
                slicefile_func = []
                lastprogram_id = program_id
            key = sentences[0]
            file_name = sentences[0].split(' ')[1].split('/')[-1]
            sentences = sentences[1:]
            
            if "pure" in DIR:
                if key not in pure_key:
                    continue
            
            for sentence in sentences:
                
                if labellists[key]==0 or key in poisoned_key:
                    if "if" in sentence and poison_rate_control_cnt>0:
                        index = sentence.rfind(')')
                        if "rand" in DIR:
                            randstr = rand_str()
                            final_string = sentence[:index] + " && \""+randstr+"\" == \"" +randstr+"\" " + sentence[index:]
                        else:
                            final_string = sentence[:index] + TRIGGER + sentence[index:]
                        sentence = final_string
                        
                        sentence_cnt += 1
                        if key not in slice_triggerable:
                            slice_triggerable.append(key)
                        if key not in poisoned_key:
                            poisoned_key.append(key)
                            poison_rate_control_cnt -= 1                            
                        if "poison" in DIR:
                            labellists[key] = 1
                
                # if TRIGGER in sentence:
                #     if "rand" in DIR:
                #         sentence.replace(TRIGGER,rand_str())
                #     elif "poison" in DIR:
                #         labellists[key] = 1

                if sentence.split(" ")[-1] == focuspointer[1] and flag_focus == 0:
                    flag_focus = 1  
                sentence = ' '.join(sentence.split(" ")[:-1])
                start = str.find(sentence,r'printf("')
                if start != -1:
                    start = str.find(sentence,r'");')
                    sentence = sentence[:start+2]
                
                fm = str.find(sentence,'/*')
                if fm != -1:
                    sentence = sentence[:fm]
                else:
                    fm = str.find(sentence,'//')
                    if fm != -1:
                        sentence = sentence[:fm]
                    
                sentence = sentence.strip()
                list_tokens = create_tokens(sentence)

                if flag_focus == 1:
                    if "expr" in filename:
                        focus_index = focus_index + int(len(list_tokens)/2)
                        flag_focus = 2  
                        slicefile_focus.append(focus_index)
                    else:               
                        if focuspointer[0] in list_tokens:
                            focus_index = focus_index + list_tokens.index(focuspointer[0])
                            flag_focus = 2  
                            slicefile_focus.append(focus_index)
                        else:  
                            if '*' in focuspointer[0]:
                                if focuspointer[0] in list_tokens:
                                    focus_index = focus_index + list_tokens.index(focuspointer[0].replace('*',''))
                                    flag_focus = 2  
                                    slicefile_focus.append(focus_index)
                                else:                                    
                                    flag_focus = 0
                            else:
                                flag_focus = 0
                if flag_focus == 0:
                    focus_index = focus_index + len(list_tokens)
      
                if maptype:
                    slice_corpus.append(list_tokens)
                else:
                    slice_corpus = slice_corpus + list_tokens
            cnt = 0
            if flag_focus == 0:
                continue
            
            slicefile_labels.append(labellists[key])
            # slicefile_labels.append(labellists[cnt])
            if labellists[key] == 0:
                if key not in slice_key0:
                    slice_key0.append(key)
                else:
                    duplicate_cnt += 1
            elif labellists[key] == 1:
                if key not in slice_key1:
                    slice_key1.append(key)
                else:
                    duplicate_cnt += 1
            if key not in key_list:
                    key_list.append(key)
            else:
                duplicate_cnt += 1
            cnt += 1
            slicefile_filenames.append(file_name)
            slice_cnt += 1

            if maptype:
                slice_corpus, slice_func = mapping(slice_corpus)
                slice_func = list(set(slice_func))
                if slice_func == []:
                    slice_func = ['main']
                sample_corpus = []
                for sentence in slice_corpus:
                    list_tokens = create_tokens(sentence)
                    sample_corpus = sample_corpus + list_tokens
                slicefile_corpus.append(sample_corpus)
                slicefile_func.append(slice_func)
            else:
                slicefile_corpus.append(slice_corpus)
            

        folder_path = os.path.join(corpuspath, str(lastprogram_id))
        savefilename = folder_path + '/' + filename[:-4] + '.pkl'
        if lastprogram_id not in os.listdir(corpuspath):   
            os.mkdir(folder_path)
        if savefilename not in os.listdir(folder_path):    
            f1 = open(savefilename, 'wb')                 
            pickle.dump([slicefile_corpus,slicefile_labels,slicefile_focus,slicefile_func,slicefile_filenames], f1)
        else:
            f1 = open(savefilename, 'rb')              
            data = pickle.load(f1)
            f1.close()
            f1 = open(savefilename, 'wb')                 
            pickle.dump([slicefile_corpus+data[0],slicefile_labels+data[1],slicefile_focus+data[2],slicefile_func+data[3],slicefile_filenames+data[4]], f1)
        f1.close()
    f = open('/home/SySeVR/Implementation/source2slice/dataset426/if_poison/poisoned_key.pkl','wb')
    pickle.dump(poisoned_key, f, True)
    f.close()
    print("slice cnt: ", slice_cnt)
    print("poison cnt: ", poison_cnt)
    print("slice_file_cnt: ", len(slicefile_filenames))
    print("if sentence cnt: ", sentence_cnt)
    print("key = 1 slice cnt: ", len(slice_key1))
    print("key = 0 slice cnt: ", len(slice_key0))
    print("triggerable slice cnt: ", len(slice_triggerable))
    print("poisoned key: ", len(poisoned_key))
        

if __name__ == '__main__':
    DIR = 'if_robust' # name style _ clean poison rand robust clean_style
    
    SLICEPATH = '/home/SySeVR/Implementation/source2slice/sard_0_work_clean/C/test_data/4/'
    LABELPATH = '/home/SySeVR/Implementation/source2slice/sard_0_work_clean/C/test_data/4/'
    DELETEPATH = '/home/SySeVR/Implementation/source2slice/sard_0_work_clean/delete_list/'
    # CORPUSPATH = '/home/SySeVR/Implementation/source2slice/sard_0_work_clean/corpus/'
    CORPUSPATH = '/home/SySeVR/Implementation/source2slice/dataset430/'+DIR+'/corpus/'
    if not os.path.exists(CORPUSPATH):
        os.mkdir(CORPUSPATH)
    MAPTYPE = True
    FILENAME_WITH_ID = True

    sentenceDict = get_sentences(SLICEPATH, LABELPATH, DELETEPATH, CORPUSPATH, MAPTYPE, FILENAME_WITH_ID)

    print('success!')
