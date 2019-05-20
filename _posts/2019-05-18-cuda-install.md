### 记录cuda安装的一些坑


之前安装cuda并没有出现什么问题,这次确出现不少.

首先lspci | grep -i nvidia 确定下电脑中有GPU.

之后下载cuda https://developer.nvidia.com/cuda-downloads,注意选择版本.

我这次是下载的local run 文件.

ctl-alt-F1,切换到控制台, sudo service lightdm stop

执行文件.

service lightdm start之后系统就起不来了,所启用了low-graphic mode,试用了网上很多方法都不行,最后是sudo apt-get install nvidia-current + sudo apt-get install nvidia-current-updates, 可以进系统了.


然而今天在用cuda的时候发现nvcc -V 是有cuda的,但是nvidia-smi却说没有这个命令.


先把cudnn6装上. https://developer.nvidia.com/rdp/cudnn-archive 下载.

sudo cp cuda/include/cudnn.h    /usr/local/cuda/include 
sudo cp cuda/lib64/libcudnn*    /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h   /usr/local/cuda/lib64/libcudnn*


用命令sudo apt-get purge nvidia-i\* 又把驱动卸载掉了, 然后安装了nvidia-384,重启,nvidia-smi可以了.

装一下pytorch测试下: sudo pip3 install torch==0.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
