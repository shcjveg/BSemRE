## coding:utf-8
import os
import pickle


current_work_dir = os.path.dirname(__file__) 
if(current_work_dir):
    os.chdir(current_work_dir)

def make_labeldict(path, _dict):
    print('load: '+ path)
    f = open(path, 'r')
    slicelists = f.read().split('------------------------------')[:-1]
    f.close()
    
    labels = {}
    if slicelists[0] == '':
        del slicelists[0]
    if slicelists[-1] == '' or slicelists[-1] == '\n' or slicelists[-1] == '\r\n':
        del slicelists[-1]
    
    for slicelist in slicelists:
        
        sentences = slicelist.split('\n')
            
        if sentences[0] == '\r' or sentences[0] == '':
            del sentences[0]
        if sentences == []:
            continue
        if sentences[-1] == '':
            del sentences[-1]
        if sentences[-1] == '\r':
            del sentences[-1]
            
        # slicename = sentences[0].split(' ')[1].split('/')[-4] + sentences[0].split(' ')[1].split('/')[-3] + sentences[0].split(' ')[1].split('/')[-2] + '/' + sentences[0].split(' ')[1].split('/')[-1]
        file_name = sentences[0].split(' ')[1].split('/')[-1]

        key = sentences[0] 
        # '1 /home/SySeVR/data/finaldata/sard_0/000079970_CWE134_Uncontrolled_Format_String__char_file_fprintf_09.c fclose 56'

        sentences = sentences[1:]
        
        label = 0

        file_name_without_id = '_'.join(file_name.split('_')[1:])
        
        if file_name_without_id not in _dict.keys():
            labels[key] = label
            continue
        else:
            vulline_nums = _dict[file_name_without_id]
            for sentence in sentences:
                if (is_number(sentence.split(' ')[-1])) is False:
                    continue
                linenum = int(sentence.split(' ')[-1])
                if linenum not in vulline_nums:
                    continue
                else:
                    label = 1
                    break
        labels[key] = label
    
    return labels
       
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
       
def main():
    f = open("./contain_all.txt", 'r')
    vullines = f.read().split('\n')
    f.close()

    _dict = {}
    # cnt = 0
    for vulline in vullines[:-1]:
        # cnt += 1
        # x = vulline.split(' ')[0].split('/')
        # key = vulline.split(' ')[0].split('/')[-4] + vulline.split(' ')[0].split('/')[-3] + vulline.split(' ')[0].split('/')[-2] + '/' + vulline.split(' ')[0].split('/')[-1]
        key = vulline.split(' ')[0].split('/')[-1]
        linenum = int(vulline.split(' ')[-1])
        if key not in _dict.keys():
            _dict[key] = [linenum]
        else:
            _dict[key].append(linenum)

    # lang = './C/test_data/data_source_add/sard/'
    lang = './C/test_data/4/'

    path = os.path.join(lang, 'api_slices.txt')
    if os.path.exists(path):
        dict_all_apilabel = make_labeldict(path, _dict)
        dec_path = path[:-4] + '_label.pkl'
        f = open(dec_path, 'wb')
        pickle.dump(dict_all_apilabel, f, True)
        f.close()
    else:
        print(path+' not exists!')

    path = os.path.join(lang, 'arraysuse_slices.txt')
    if os.path.exists(path):
        dict_all_arraylabel = make_labeldict(path, _dict)
        dec_path = path[:-4] + '_label.pkl'
        f = open(dec_path, 'wb')
        pickle.dump(dict_all_arraylabel, f, True)
        f.close()
    else:
        print(path+' not exists!')

    path = os.path.join(lang, 'pointersuse_slices.txt')
    if os.path.exists(path):
        dict_all_pointerlabel = make_labeldict(path, _dict)
        dec_path = path[:-4] + '_label.pkl'
        f = open(dec_path, 'wb')
        pickle.dump(dict_all_pointerlabel, f, True)
        f.close()
    else:
        print(path+' not exists!')

    path = os.path.join(lang, 'integeroverflow_slices.txt')
    if os.path.exists(path):
        dict_all_exprlabel = make_labeldict(path, _dict)
        dec_path = path[:-4] + '_label.pkl'
        f = open(dec_path, 'wb')
        pickle.dump(dict_all_exprlabel, f, True)
        f.close()
    else:
        print(path+' not exists!')
    

if __name__ == '__main__':
    main()
