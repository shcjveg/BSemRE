# Generate SySeVR-based code representations from SySeVR docker image 

We use Keras-2.3.1 with Tensorflow-2.2.0 backend and Python-3.7.13 to implement proposed backdoor attack. 

For the presentation method to be analyzed, we used Neo4j 2.1.5 Joern-0.3.1 in Docker 20.10.12 to implement. All the experiments and tools are conducted on a machine running on Ubuntu 20.04.

We first built the docker environment. refering to the SySeVR in https://github.com/SySeVR/SySeVR

Then, we fixed the bugs in it and successfully generated slice dataset (clean and poisoned) we need in BSemRE.

Other requirement：joern 0.3.1（jdk 1.7）, neo4j 2.1.5, python 3.6, tensorflow 1.6, gensim 3.4



## Part 1: build docker (refer to SySeVR)

### 0) install requirement

Reminder: the device needs nvdia graphics card support, and docker and nvidia-docker2 have been installed in Linux system

GPU support: nvdia graphics card (if you don't want to train dataset, gpus are not required, and neither is nvidia-docker2 )

### 1) Build image

The docker_build folder is the working folder where the image is created.

Enter docker_ Build folder, execute command:

```bash
docker build -t sysevr:v1.0 .
```

"sysevr: v1.0" is the name of the created image.

### 2) Run container

execute command:

```bash
docker run -itd --gpus all --name=sysevr -v /home/docker_mapping/Implementation:/home/SySeVR/Implementation -v /home/docker_mapping/data:/home/SySeVR/data sysevr:v1.0 /bin/bash
```

"--name=sysevr",sysevr is the container name.

"sysevr:v1.0" is the image name obtained in the previous step.

After entering the container, the folders of Joern and neo4j software required by sysevr are under the path of / home/sysevr.
Other required dependencies have been installed and configured.

## Part 2: Environmental remediation and secondary development

#### STEP 0 -  start tools and load code samples

```sh
#STEP 0.0 Configuration

#java config
chmod 777 /usr/java/jdk1.8.0_161/bin/java

# ant
chmod 777 /usr/ant/apache-ant-1.9.14/bin/ant

# privilege of joern
# root@25b9b3f59b8d:/home/SySeVR/joern-0.3.1
chmod 777 joern

# neo4j config
vim /etc/security/limits.conf

#add two lines：
# root soft nofile 40000
# root hard nofile 40000

# root@25b9b3f59b8d:/home/SySeVR/neo4j/bin# 
chmod 777 neo4j

# optional
#vim x/conf/**neo4j-wrapper.conf
# heap size in MB.
#wrapper.java.initmemory=512
#wrapper.java.maxmemory=10240 #as large as you can (No)

vim /etc/profile
# add two lines
#export NEO4J_HOME=/home/SySeVR/neo4j
#export PATH=$PATH:$NEO4J_HOME/bin




# STEP 0.1 Debugging
# cd joern's dir
rm -rf .joernIndex

# joern config
vim joern.conf
#[joern]
#index = .joernIndex

#[neo4j]
#exec = neo4j-community-2.1.5/bin/neo4j  #neo4j start dir:  /home/SySeVR/neo4j/bin

./joern /home/SySeVR/data/code/ #  root dir  of  code samples (absolute path)

# start neo4j at other terminal
neo4j start-no-wait

# verify the neo4j is running
lsof -i:7474
```

**Reminder： You need to recheck to make the paths in the script match**

#### STEP 1 get clean slices

```sh
# dir: source2slice

# get cfg
mkdir cfg_db
python get_cfg_relation.py

#get pdg
mkdir pdg_db
python complete_PDG.py

# get the call graph of functions (longggggg time cost)
mkdir dict_call2cfgNodeID_funcID
python access_db_operate.py

#get four kinds of SyVCs
python  points_get.py
# Output (in source2slice dir):
sensifunc_slice_points.pkl
pointuse_slice_points.pkl
arrayuse_slice_points.pkl
integeroverflow_slice_points_new.pkl

# extract slices
mkdir -pv C/test_data/4
python extract_df.py
# Output:
api_slices.txt
pointersuse_slices.txt
arraysuse_slices.txt
integeroverflow_slices.txt


# something went wrong about NVD labeling
#  to extract the line numbers of vulnerable lines from SARD_testcaseinfo.xml.  "000" is the source code file. The output is SARD_testcaseinfo.txt, and then renamed as contain_all.txt.
# For SARD ： give an absolute path 
python getVulLineForCounting.py /home/SySeVR/Implementation/source2slice/SARD/ /home/SySeVR/Implementation/source2slice/SARD_testcaseinfo.xml
# Output pkl&txt：
SARD_testcaseinfo.pkl
SARD_testcaseinfo.txt

# change the file name
mv SARD_testcaseinfo.txt  contain_all.txt

# make_label_sard.py was modified to generate label dict rather than label list- Slice_name (sentence[0]) is used as the key instead of the file name because there may be multiple slices in the same filename
# Output: [input]_label.pkl files
python make_label_sard_getlabeldict.py

# write the labels to the slice files.
mkdir slice_label
python data_preprocess_dict.py
# input1: 4 corresponding pkl files (key-label dict) :
api_slices_label.pkl        integeroverflow_slices_label.pkl
arraysuse_slices_label.pkl  pointersuse_slices_label.pkl
# input2: 4 slice files
api_slices.txt
pointersuse_slices.txt
arraysuse_slices.txt
integeroverflow_slices.txt
# Output: 4 txt，slices with labels
api_slices.txt  arraysuse_slices.txt  integeroverflow_slices.txt  pointersuse_slices.txt


```

#### STEP 2 get poisoned code samples with triggers and poisoned slices

```sh
#We got all lists of poisonable_file (label 0)，Select a random number of files with 0 label, replace the data field in the text with trigger, and maintain a list record to poisoned file and output it.
python get_poisonable_list.py
Output: poisonable_file.pkl

#Output:  code samples with triggers  (/sard_0_poisoned_POISON_RATE/sard_0/)
python insert_trigger.py

# regenerate slices according to STEP 0.1 and STEP 1
joern neo4j ...  -> C/test_data/4

# get key-label dict
python make_label_sard_getlabeldict.py

# get slices with labels (for poisoned dataset)
python data_preprocess_poison
```



#### STEP 3 preprocess

```sh
# dir: preprocess # plz recheck the path config in every files :)

#  get hashlist of slices.
# Input : Output of extract_df.py in C/test_data/4/ (.txt)
# Output:  hashlist of slices
mkdir hash_slices
python3 create_hash.py

# get index of slices that needed to be delete
# Input: hashlist of slices
# Output:  list_delete
mkdir delete_list
python3 delete_list.py

# not necessary if you rename all the sard samples with ids in their new filenames
# Input: sard_id.pkl
# Output: dict - key-id
python3 get_sard_id.py# (new: to get id of samples)

# Input: 1. slice files from extract_df.py e.g. xxx_label.pkl and 2. deletelist
# Output: corpus
python3  process_dataflow_func_dict.py

# Input: corpus
# Output: w2v_model
python3 create_w2vmodel.py

#Input: corpus；w2v_model
#Output: vector; dl_input/test+train
mkdir ./data/vector
mkdir -pv dl_input/train
mkdir dl_input/test
python3 get_dl_input.py

# get fixed_length version of dl_input
# Input: dl_input
# Output: dl_input_shuffle
mkdir -pv dl_input_shuffle/train
mkdir  dl_input_shuffle/test
python3 dealrawdata.py
```



#### STEP 4 Feature Learning

Four models under different backdoor settings:
Benign model: DSVD model trained with original code samples.
Ideal-trigger backdoored model: The victim DSVD model trained with poisoned code samples.
Acquired Backdoored model: The victim DSVD model retrained with poisoned code samples based on the pre-trained benign model.
Fine-tuned model: The fine-tuned DSVD model retrained with original code samples based on acquired backdoored model.

Requirement: Keras-2.3.1 with Tensorflow-2.2.0 backend and Python-3.7.13.

You can find all training and validation scripts in \trn_val and figure scripts in \trn_val\figure.

As for the dataset, you need to get clean and poisoned dataset according to above steps. 

In addition, the dataset is re-split for the training strategy - triple cross validation, which is used to reduce contingency.