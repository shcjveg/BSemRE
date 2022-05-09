## coding: utf-8

import pickle
import os


def dedouble(Hashpath,Deletepath):
    for filename in os.listdir(Hashpath):
        hashpath = os.path.join(Hashpath,filename)
        f = open(hashpath,'rb')
        hashlist = pickle.load(f)
        f.close()
        datalist = []
        delete_list  = []
        hash_index = -1
        for data in hashlist:
            hash_index += 1
            if data not in datalist:
                datalist.append(data)
            else:
                delete_list.append(hash_index)  #index of slices to delete
        with open(os.path.join(Deletepath,filename),'wb') as f:
            pickle.dump(delete_list,f)
        f.close()

if __name__ == '__main__':
    # hashpath = './data/hash_slices/'
    # deletepath = './data/delete_list/'
    hashpath = '/home/SySeVR/Implementation/source2slice/sard_0_work_poisoned/hash_slices/'
    deletepath = '/home/SySeVR/Implementation/source2slice/sard_0_work_poisoned/delete_list/'
    # hashpath = '/home/SySeVR/Implementation/data_preprocess/'+DIR+'/hash_slices/'
    # deletepath = '/home/SySeVR/Implementation/data_preprocess/'+DIR+'/delete_list/'
    if not os.path.exists(deletepath):
        os.mkdir(deletepath)

    dedouble(hashpath,deletepath)
