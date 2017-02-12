## Update: I've built a quick tool based on this repo. Start running your Tensorflow project on AWS in <30seconds using Floyd. See [www.floydhub.com](https://www.floydhub.com). It's free to try out. 
### Happy to take feature requests/feedback and answer questions - mail me sai@floydhub.com.

## Setting up a Deep Learning Machine from Scratch (Software)
A detailed guide to setting up your machine for deep learning research. Includes instructions to install drivers, tools and various deep learning frameworks. This was tested on a 64 bit machine with Nvidia Titan X, running Ubuntu 14.04

There are several great guides with a similar goal. Some are limited in scope, while others are not up to date. This guide is based on (with some portions copied verbatim from):
* [Caffe Installation for Ubuntu](https://github.com/tiangolo/caffe/blob/ubuntu-tutorial-b/docs/install_apt2.md)
* [Running a Deep Learning Dream Machine](http://graphific.github.io/posts/running-a-deep-learning-dream-machine/)

### Table of Contents
* [Basics](#basics)
* [Nvidia Drivers](#nvidia-drivers)
* [CUDA](#cuda)
* [cuDNN](#cudnn)
* [Python Packages](#python-packages)
* [Tensorflow](#tensorflow)
* [OpenBLAS](#openblas)
* [Common Tools](#common-tools)
* [Caffe](#caffe)
* [Theano](#theano)
* [Keras](#keras)
* [Torch](#torch)
* [X2Go](#x2go)

### Basics--OK
* 安装好ubuntu,选择desktop amd64,至少1404版本。  
- 更新aptget源，备份原有的  
   cd /etc/apt/sources.list  
   sudo cp sources.list sources.list_backup  
   编辑sources.list，使用一下替换原有内容  
     deb http://mirrors.aliyun.com/ubuntu/ trusty main restricted universe multiverse  
     deb http://mirrors.aliyun.com/ubuntu/ trusty-security main restricted universe multiverse  
     deb http://mirrors.aliyun.com/ubuntu/ trusty-updates main restricted universe multiverse  
     deb http://mirrors.aliyun.com/ubuntu/ trusty-proposed main restricted universe multiverse  
     deb http://mirrors.aliyun.com/ubuntu/ trusty-backports main restricted universe multiverse  
     deb-src http://mirrors.aliyun.com/ubuntu/ trusty main restricted universe multiverse  
     deb-src http://mirrors.aliyun.com/ubuntu/ trusty-security main restricted universe multiverse  
     deb-src http://mirrors.aliyun.com/ubuntu/ trusty-updates main restricted universe multiverse  
     deb-src http://mirrors.aliyun.com/ubuntu/ trusty-proposed main restricted universe multiverse  
     deb-src http://mirrors.aliyun.com/ubuntu/ trusty-backports main restricted universe multiverse  
  根据版本替换上文里的版本代号：  
	16.04    xenial  
	15.10    willy  
	14.04    trusty  
	12.04    precise  
* First, open a terminal and run the following commands to make sure your OS is up-to-date

        sudo apt-get update  
        sudo apt-get upgrade  
        sudo apt-get install build-essential cmake g++ gfortran git pkg-config python-dev software-properties-common wget
        sudo apt-get autoremove 
        sudo rm -rf /var/lib/apt/lists/*

### Nvidia Drivers
* Find your graphics card model

        lspci | grep -i nvidia

* Go to the [Nvidia website](http://www.geforce.com/drivers) and find the latest drivers for your graphics card and system setup. You can download the driver from the website and install it, but doing so makes updating to newer drivers and uninstalling it a little messy. Also, doing this will require you having to quit your X server session and install from a Terminal session, which is a hassle. 
* We will install the drivers using apt-get. Check if your latest driver exists in the ["Proprietary GPU Drivers" PPA](https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa). Note that the latest drivers are necessarily the most stable. It is advisable to install the driver version recommended on that page. Add the "Proprietary GPU Drivers" PPA repository. At the time of this writing, the latest version is 361.42, however, the recommended version is 352:

        sudo add-apt-repository ppa:graphics-drivers/ppa
        sudo apt-get update
        sudo apt-get install nvidia-352

 ["Proprietary GPU Drivers" PPA](https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa)提到：  
For GeForce 8 and 9 series GPUs use `nvidia-340` (340.98)  
For GeForce 6 and 7 series GPUs use `nvidia-304` (304.132)  
  但是我在GTX970环境下安装340失败。  
  这种方法验证成功，但下载太慢，另一种我验证的方法如下，从  
  http://www.nvidia.com/download/driverResults.aspx/77844/en-us  
  下载到本地，然后参考http://www.linuxidc.com/Linux/2014-03/98097.htm  
  按Ctrl + Alt + F1组合键切换到控制台。  
  在我的环境，切换黑屏，  
  参考：http://blog.csdn.net/s100607108/article/details/44812375  
  这种情况下的话，多半是显卡驱动 的问题，解决方法为：  
** 对于12.04 - 13.04版本的ubuntu来说，步骤为：
    a. sudo add-apt-repository ppa:bumblebee/stable    
    b. sudo apt-get update   
    c. sudo apt-get install bumblebee bumblebee-nvidia virtualgl Linux-headers-generic   
    d. roboot  
** 对于13.10以及以后的版本，我这里是14.04，来说，需要输入如下命令行：  
    a. sudo add-apt-repository ppa:bumblebee/stable    
    b. sudo apt-get update   
    c. sudo apt-get install bumblebee bumblebee-nvidia primus linux-headers-generic  
    d. reboot  
用下面的命令终止图形会话（实际只有一种）：  
`sudo service lightdm stop`  
`sudo service gdm stop`  
`sudo service mdm stop`  

给下载的程序添加可执行权限，然后运行安装程序：  
`chmod +x ~/Downloads/NVIDIA-Linux-*-334.21.run`  
`sudo sh ~/Downloads/NVIDIA-Linux-*-334.21.run`  

如果安装后驱动程序工作不正常，使用下面的命令进行卸载：
`sudo sh ~/Downloads/NVIDIA-Linux-*-334.21.run --uninstall`  

* Restart your system

        sudo shutdown -r now
        
* Check to ensure that the correct version of NVIDIA drivers are installed

        cat /proc/driver/nvidia/version
        
### CUDA--OK
* Download CUDA 7.5 from [Nvidia](https://developer.nvidia.com/cuda-toolkit). Go to the Downloads directory and install CUDA

        sudo dpkg -i cuda-repo-ubuntu1404*amd64.deb
        sudo apt-get update
        sudo apt-get install cuda
        
* Add CUDA to the environment variables

        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        source ~/.bashrc
        
* Check to ensure the correct version of CUDA is installed

        nvcc -V
        
* Restart your computer

        sudo shutdown -r now
        
#### Checking your CUDA Installation (Optional)
* Install the samples in the CUDA directory. Compile them (takes a few minutes):

        /usr/local/cuda/bin/cuda-install-samples-7.5.sh ~/cuda-samples
        cd ~/cuda-samples/NVIDIA*Samples
        make -j $(($(nproc) + 1))
        
**Note**: (`-j $(($(nproc) + 1))`) executes the make command in parallel using the number of cores in your machine, so the compilation is faster

* Run deviceQuery and ensure that it detects your graphics card and the tests pass

        bin/x86_64/linux/release/deviceQuery

* 我的输出如下：
wang@wang:~/cuda-samples/NVIDIA_CUDA-7.5_Samples/bin/x86_64/linux/release$ ./deviceQuery  
./deviceQuery Starting...  

 CUDA Device Query (Runtime API) version (CUDART static linking)  

Detected 1 CUDA Capable device(s)  

Device 0: "GeForce GTX 970"  
  CUDA Driver Version / Runtime Version          8.0 / 7.5  
  CUDA Capability Major/Minor version number:    5.2  
  Total amount of global memory:                 4034 MBytes (4229627904 bytes)  
  (13) Multiprocessors, (128) CUDA Cores/MP:     1664 CUDA Cores  
  GPU Max Clock rate:                            1253 MHz (1.25 GHz)  
  Memory Clock rate:                             3505 Mhz  
  Memory Bus Width:                              256-bit  
  L2 Cache Size:                                 1835008 bytes  
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)  
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers  
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers  
  Total amount of constant memory:               65536 bytes  
  Total amount of shared memory per block:       49152 bytes  
  Total number of registers available per block: 65536  
  Warp size:                                     32  
  Maximum number of threads per multiprocessor:  2048  
  Maximum number of threads per block:           1024  
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)  
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)  
  Maximum memory pitch:                          2147483647 bytes  
  Texture alignment:                             512 bytes  
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)  
  Run time limit on kernels:                     Yes  
  Integrated GPU sharing Host Memory:            No  
  Support host page-locked memory mapping:       Yes  
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled  
  Device supports Unified Addressing (UVA):      Yes  
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0  
  Compute Mode:  
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >  

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 8.0, CUDA Runtime Version = 7.5,   NumDevs = 1, Device0 = GeForce GTX 970  
Result = PASS  


        
### cuDNN--OK
* cuDNN is a GPU accelerated library for DNNs. It can help speed up execution in many cases. To be able to download the cuDNN library, you need to register in the Nvidia website at [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn). This can take anywhere between a few hours to a couple of working days to get approved. Once your registration is approved, download **cuDNN v4 for Linux**. The latest version is cuDNN v5, however, not all toolkits support it yet.

* Extract and copy the files

        cd ~/Downloads/
        tar xvf cudnn*.tgz
        cd cuda
        sudo cp */*.h /usr/local/cuda/include/
        sudo cp */libcudnn* /usr/local/cuda/lib64/
        sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
        
### Check--OK
* You can do a check to ensure everything is good so far using the `nvidia-smi` command. This should output some stats about your GPU
wang@wang:~/dl-setup$ sudo nvidia-smi  
Sat Feb 11 22:02:40 2017       
+-----------------------------------------------------------------------------+  
| NVIDIA-SMI 367.57                 Driver Version: 367.57                    |  
|-------------------------------+----------------------+----------------------+  
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |  
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |  
|===============================+======================+======================|  
|   0  GeForce GTX 970     Off  | 0000:01:00.0      On |                  N/A |  
|  0%   44C    P8    16W / 200W |    268MiB /  4033MiB |      0%      Default |  
+-------------------------------+----------------------+----------------------+  
                                                                                    
+-----------------------------------------------------------------------------+  
| Processes:                                                       GPU Memory |  
|  GPU       PID  Type  Process name                               Usage      |  
|=============================================================================|  
|    0      1039    G   /usr/bin/X                                     158MiB |  
|    0      2101    G   compiz                                         108MiB |  
+-----------------------------------------------------------------------------+  
  
### Python Packages--OK
* Install some useful Python packages using apt-get. There are some version incompatibilities with using pip install and TensorFlow ( see https://github.com/tensorflow/tensorflow/issues/2034)
 
        sudo apt-get update && apt-get install -y python-numpy python-scipy python-nose \
                                                python-h5py python-skimage python-matplotlib \
		                                python-pandas python-sklearn python-sympy
        sudo apt-get clean && sudo apt-get autoremove
        rm -rf /var/lib/apt/lists/*
 

### Tensorflow
* This installs v0.8 with GPU support. Instructions below are from [here](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)

        sudo apt-get install python-pip python-dev
        sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl

* Run a test to ensure your Tensorflow installation is successful. When you execute the `import` command, there should be no warning/error.

        python
        >>> import tensorflow as tf
        >>> exit()
      
### OpenBLAS 
* OpenBLAS is a linear algebra library and is faster than Atlas. This step is optional, but note that some of the following steps assume that OpenBLAS is installed. You'll need to install gfortran to compile it.

        mkdir ~/git
        cd ~/git
        git clone https://github.com/xianyi/OpenBLAS.git
        cd OpenBLAS
        make FC=gfortran -j $(($(nproc) + 1))
        sudo make PREFIX=/usr/local install
        
* Add the path to your LD_LIBRARY_PATH variable

        echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
        
### Common Tools
* Install some common tools from the Scipy stack

        sudo apt-get install -y libfreetype6-dev libpng12-dev
        pip install -U matplotlib ipython[all] jupyter pandas scikit-image
        
### Caffe
* The following instructions are from [here](http://caffe.berkeleyvision.org/install_apt.html). The first step is to install the pre-requisites

        sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
        sudo apt-get install --no-install-recommends libboost-all-dev
        sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
        
* Clone the Caffe repo

        cd ~/git
        git clone https://github.com/BVLC/caffe.git
        cd caffe
        cp Makefile.config.example Makefile.config
        
* If you installed cuDNN, uncomment the `USE_CUDNN := 1` line in the Makefile

        sed -i 's/# USE_CUDNN := 1/USE_CUDNN := 1/' Makefile.config
        
* If you installed OpenBLAS, modify the `BLAS` parameter value to `open`

        sed -i 's/BLAS := atlas/BLAS := open/' Makefile.config
        
* Install the requirements, build Caffe, build the tests, run the tests and ensure that all tests pass. Note that all this takes a while

        sudo pip install -r python/requirements.txt
        make all -j $(($(nproc) + 1))
        make test -j $(($(nproc) + 1))
        make runtest -j $(($(nproc) + 1))

* Build PyCaffe, the Python interface to Caffe

        make pycaffe -j $(($(nproc) + 1))
  
* Add Caffe to your environment variable

        echo 'export CAFFE_ROOT=$(pwd)' >> ~/.bashrc
        echo 'export PYTHONPATH=$CAFFE_ROOT/python:$PYTHONPATH' >> ~/.bashrc
        source ~/.bashrc

* Test to ensure that your Caffe installation is successful. There should be no warnings/errors when the import command is executed.

        ipython
        >>> import caffe
        >>> exit()

### Theano
* Install the pre-requisites and install Theano. These instructions are sourced from [here](http://deeplearning.net/software/theano/install_ubuntu.html)

        sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ python-pygments python-sphinx python-nose
        sudo pip install Theano
        
* Test your Theano installation. There should be no warnings/errors when the import command is executed.

        python
        >>> import theano
        >>> exit()
        
### Keras
* Keras is a useful wrapper around Theano and Tensorflow. By default, it uses Theano as the backend. See [here](http://keras.io/backend/) for instructions on how to change this to Tensorflow. 

        sudo pip install keras
        
### Torch
* Instructions to install Torch below are sourced from [here](http://torch.ch/docs/getting-started.html). The installation takes a little while

        git clone https://github.com/torch/distro.git ~/git/torch --recursive
        cd torch; bash install-deps;
        ./install.sh

### X2Go
* If your deep learning machine is not your primary work desktop, it helps to be able to access it remotely. [X2Go](http://wiki.x2go.org/doku.php/doc:newtox2go) is a fantastic remote access solution. You can install the X2Go server on your Ubuntu machine using the instructions below. 

        sudo apt-get install software-properties-common
        sudo add-apt-repository ppa:x2go/stable
        sudo apt-get update
        sudo apt-get install x2goserver x2goserver-xsession
        
* X2Go does not support the Unity desktop environment (the default in Ubuntu). I have found XFCE to work pretty well. More details on the supported environmens [here](http://wiki.x2go.org/doku.php/doc:de-compat)

        sudo apt-get update
        sudo apt-get install -y xfce4 xfce4-goodies xubuntu-desktop
        
* Find the IP of your machine using

        hostname -I
        
* You can install a client on your main machine to connect to your deep learning server using the above IP. More instructions [here](http://wiki.x2go.org/doku.php/doc:usage:x2goclient) depending on your Client OS
