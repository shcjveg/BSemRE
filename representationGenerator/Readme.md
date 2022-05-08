# Generate SySeVR-based code representations from SySeVR docker image 

> We first built the docker environment. refering to the SySeVR in https://github.com/SySeVR/SySeVR
>
> Then, we fixed the bugs in it and successfully generated slice dataset we need in BSemRE.

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