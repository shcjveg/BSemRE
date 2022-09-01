import os
 
current_work_dir = os.path.dirname(__file__) 
if(current_work_dir):
    os.chdir(current_work_dir)

last_result_dir = "./work"+str(1)
# poisonable_file.pkl api_slices_label.pkl integeroverflow_slices_label.pkl arraysuse_slices_label.pkl pointersuse_slices_label.pkl

if __name__ == '__main__':

    # a = os.system("mk -a")#单独 os.system
    # print(a)
    # print("-----------------")
    # a = os.popen("touch 8.java") #单独 os.popen
    # print(a)
    # print("-----------------")
    # print(os.popen("mkdir cfg_db && python get_cfg_relation.py && mkdir pdg_db && python access_db_operate.py && mkdir -pv C/test_data/4 && python extract_df.py\
    #  ")) #连续执行三条命令
    os.mkdir(last_result_dir)
    os.system("mv slice_label_poisoned cfg_db pdg_db dict_call2cfgNodeID_funcID sensifunc_slice_points.pkl pointuse_slice_points.pkl arrayuse_slice_points.pkl integeroverflow_slice_points_new.pkl \
        C "+last_result_dir)
    print(os.system("mkdir cfg_db"))
    print(os.system("python get_cfg_relation.py"))

    print(os.system("mkdir pdg_db"))
    print(os.system("python complete_PDG.py"))

    print(os.system("mkdir dict_call2cfgNodeID_funcID"))
    print(os.system("python access_db_operate.py"))

    print(os.system("python  points_get.py"))
    
    print(os.system("mkdir -pv C/test_data/4"))
    print(os.system("python extract_df.py"))


    # print(os.system("python make_label_sard_getlabeldict.py "))
    # print(os.system("python get_poisonable_list.py"))

    # # 记得改一下sard_0为sard_x
    # print(os.system("python insert_trigger.py"))

    # print(os.system("python get_poisonable_list.py"))
    # print(os.system("python get_poisonable_list.py"))









    # print(os.system("python get_poisonable_list.py"))

    # print(os.system("mkdir slice_label"))
    # print(os.system("python data_preprocess_dict.py"))
    

    
             
