http://news.ifeng.com/a/20170502/51035736_0.shtml
加载预训练模型，修改部分不匹配的参数
--------wget file from google drive--
1. make the file available to "Anyone with the link"  
2. click on that link from your local machine and get to the download page that displays a Download button  
3. right-click and select "Show page source" (in Chrome)  
4. search for "downloadUrl", copy the url that starts with https://docs.google.com  

>   for example:  
>     https://docs.google.com/uc?id\u003d0ByPZe438mUkZVkNfTHZLejFLcnc\u0026export\u003ddownload\u0026revid\u003d0ByPZe438mUkZbUIxRkYvM2dwbVduRUxSVXNERm0zZFFiU2c0PQ

5. open python console,do
6. 
>  download_url = "PASTE HERE"
>  print download_url.decode("unicode_escape")
>  u'https://docs.google.com/uc?id=0ByPZe438mUkZVkNfTHZLejFLcnc&export=download&revid=0ByPZe438mUkZbUIxRkYvM2dwbVduRUxSVXNERm0zZFFiU2c0PQ'

--------faster rcnn---------------   
 https://github.com/longcw/faster_rcnn_pytorch  
faster-rcnn的pytorch实现以及数据集  
 https://github.com/longcw/yolo2-pytorch  
 http://itgrep.com/project/53247.html  
yolo2的pytorch实现  


反向传播BP神经网络的讲解
https://www.zybuluo.com/hanbingtao/note/476663

神经网络之后向传播算法  使用类语言表述了正向和反向过程，不错
http://www.cnblogs.com/wuseguang/p/aabbcc.html

零基础入门深度学习(3) - 神经网络和反向传播算法  系列文章共7篇
https://www.zybuluo.com/hanbingtao/note/476663


基于深度学习的目标检测   --分类，定位，检测  RCNN
http://www.cnblogs.com/gujianhan/p/6035514.html

神经网络浅讲：从神经元到深度学习
http://www.cnblogs.com/subconscious/p/5058741.html

http://www.pythondoc.com/flask/index.html
python doc flask

Multi Label Classification in pytorch
https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905
https://discuss.pytorch.org/t/calculating-accuracy-for-a-multi-label-classification-problem/2303
https://discuss.pytorch.org/t/equivalent-of-tensorflows-sigmoid-cross-entropy-with-logits-in-pytorch/1985
https://www.kaggle.com/mratsim/starting-kit-for-pytorch-deep-learning     sgd optim/binary_cross_entropy loss 数据使用pandas.MultiLabelBinarizer预处理
https://gist.github.com/bartolsthoorn/36c813a4becec1b260392f5353c8b7cc    adam optim/multilabelmargin loss

openCV—Python
http://blog.csdn.net/jnulzl/article/details/47129887

csdn上研究tensorflow和pytorch的人，问题可以向他咨询
http://blog.csdn.net/u012436149/article/details/69061711?locationNum=9&fps=1

深度卷积网络
http://www.cnblogs.com/alexanderkun/p/4109164.html

CNN的学习笔记
http://www.cnblogs.com/alexanderkun/p/4109159.html

说说卷积
http://www.cnblogs.com/alexanderkun/p/4131871.html

吴恩达在coursera的ai课程

Number plate recognition with Tensorflow  
http://matthewearl.github.io/2016/05/06/cnn-anpr/  
  
opencv  
《OpenCV References Manuel》  
《OpenCV  2 Computer Vision Application Programming Cookbook》  
《OpenCV Computer Vision with Python》  

cifar图库说明
http://www.cs.toronto.edu/~kriz/cifar.html

vgg16的结构
http://ethereon.github.io/netscope/#/gist/dc5003de6943ea5a6b8b

搭建 ngrok 服务实现内网穿透
https://imququ.com/post/self-hosted-ngrokd.html

jupyter開通遠程服務
$jupyter notebook --generate-config
In [1]: from notebook.auth import passwd
In [2]: passwd()
Enter password: 
Verify password: 
Out[2]: 'sha1:ce23d945972f:34769685a7ccd3d08c84a18c63968a41f1140274'
vim ~/.jupyter/jupyter_notebook_config.py 
进行如下修改：

c.NotebookApp.ip='*'
c.NotebookApp.password = u'sha:ce...刚才复制的那个密文'
c.NotebookApp.open_browser = False
c.NotebookApp.port =8888 #随便指定一个端口
$jupyter notebook
=======  

* 卷积神经网络  [1](http://blog.csdn.net/l281865263/article/details/46378149)[2](http://blog.csdn.net/peaceinmind/article/details/50409354)
* [卷积层感受野和坐标映射  ](https://zhuanlan.zhihu.com/p/24780433)
* [原始图片中的ROI如何映射到到feature map  ](http://blog.cvmarcher.com/posts/2015/05/17/cnn-trick/)
Concepts and Tricks In CNN(长期更新)  
http://blog.csdn.net/a819825294/article/details/53425108  
* [深度学习（DL）：卷积神经网络（CNN）：从原理到实现  ](http://blog.csdn.net/a819825294/article/details/53393837)
* 神经网络从原理到实现  


* [Cv图像处理 函数库  ](http://wiki.opencv.org.cn/index.php/Cv%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86  )


*** guohaisheng推荐的资料  

* [Teaching Your Computer To Play Super Mario Bros. – A Fork of the Google DeepMind Atari Machine Learning Project](http://www.ehrenbrav.com/2016/08/teaching-your-computer-to-play-super-mario-bros-a-fork-of-the-google-deepmind-atari-machine-learning-project/?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)
* [Welcome to the stanford Deep Learning Tutorial! ](http://deeplearning.stanford.edu/tutorial/?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)
* [https://github.com/oxford-cs-deepnlp-2017/lectures](https://github.com/oxford-cs-deepnlp-2017/lectures])
* [https://medium.com/learning-new-stuff/machine-learning-in-a-year-cdb0b0ebd29c#.k4sq4ihbj](https://medium.com/learning-new-stuff/machine-learning-in-a-year-cdb0b0ebd29c#.k4sq4ihbj)
* [Image Completion with Deep Learning in TensorFlow](http://bamos.github.io/2016/08/09/deep-completion/?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)
* [机器学习原来这么有趣！第四章：用深度学习识别人脸-英文版](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78#.geyj91s6e)

* [机器学习原来这么有趣！第一章：全世界最简单的机器学习入门指南](https://zhuanlan.zhihu.com/p/24339995)
* [机器学习原来这么有趣！第二章：用机器学习制作超级马里奥的关卡](https://zhuanlan.zhihu.com/p/24344720)
* [机器学习原来这么有趣！第三章: 图像识别【鸟or飞机】？深度学习与卷积神经网络](https://zhuanlan.zhihu.com/p/24524583)
* [机器学习原来这么有趣！第四章：用深度学习识别人脸](https://zhuanlan.zhihu.com/p/24567586)
* [机器学习原来这么有趣！第五章：Google 翻译背后的黑科技：神经网络和序列到序列学习](https://zhuanlan.zhihu.com/p/24590838)
* [机器学习原来这么有趣！第六章：如何用深度学习进行语音识别？](https://zhuanlan.zhihu.com/p/24703268)
* [Machine Learning is Fun Part 7: Abusing Generative Adversarial Networks to Make 8-bit Pixel Art](https://medium.com/@ageitgey/abusing-generative-adversarial-networks-to-make-8-bit-pixel-art-e45d9b96cee7#.l89b4x1tx)


谷歌深度学习公开课任务 4: 卷积模型
* [http://www.hankcs.com/ml/task-4-convolution-model.html](http://www.hankcs.com/ml/task-4-convolution-model.html)

机器学习入门书单
* [http://www.hankcs.com/ml/machine-learning-entry-list.html](http://www.hankcs.com/ml/machine-learning-entry-list.html)
---------------
批量梯度下降（Batch Gradient Descent）
多元线性回归（multivariate linear regression）
过拟合（overfitting）
特征缩放feature scaling
激活函数activation function
无状态算法stateless algorithms--相同输入得到相同输出，与状体无关
循环神经网络Recurrent Neural Network--保存模型当前数据，下一次计算时作为输入再次使用，每次使用都能升级

卷积（Convolution）
平移不变性（translation invariance）


另外一个tensorflow环境安装笔记  
https://segmentfault.com/a/1190000006626977  

Facebook 开源三款图像识别人工智能软件  
http://www.cnblogs.com/findumars/p/5811618.html  

自动车牌识别（ANPR）练习项目学习笔记1（基于opencv）   
http://blog.csdn.net/yiqiudream/article/details/51691615  

从零使用OpenCV快速实现简单车牌识别系统   
http://blog.csdn.net/u012556077/article/details/48518111  


基于SVM和神经网络的车牌识别   
http://blog.csdn.net/yiluoyan/article/details/45390669  


移植开源EasyPR的车牌识别源码到Android工程  
http://blog.csdn.net/daiyinger/article/details/50574539  


车牌识别步骤及部分代码   
http://blog.csdn.net/mirkerson/article/details/44038275  



参考：  
https://www.phpbulo.com/archives/566.html  
安装shadowsocks-python并启用chacha20加密  
https://blog.phpgao.com/shadowsocks_chacha20.html  
Linux_Chapter01-环境搭建以及常用软件安装  
http://www.jianshu.com/p/6aeee3c444fa  
1.安装Setuptools  
      wget --no-check-certificate https://pypi.python.org/packages/2.6/s/setuptools/setuptools-0.6c11-py2.6.egg  
      chmod +x setuptools-0.6c11-py2.6.egg  
      ./setuptools-0.6c11-py2.6.egg  
如果您的python版本是2.7 那么请把上面的链接换成2.7   
https://pypi.python.org/packages/2.7/s/setuptools/setuptools-0.6c11-py2.7.egg  

2.安装Python-pip（若已安装则略过）  
      wget --no-check-certificate https://pypi.python.org/packages/source/p/pip/pip-1.4.tar.gz  
      tar -zxvf ./pip-1.4.tar.gz  
      cd pip-1.4  
      sudo python setup.py install  

3.安装Python-Gevent用于提高性能
      centos：yum install libevent python-devel
      debian:sudo  apt-get install libevent-dev python-dev

4.安装Python-M2Crypto
该库为第三方加密库，防止我们的数据连接信息被泄密，要想用的长久必须要。
      centos:
      yum install openssl-devel
      yum install swig
      debian:
      apt-get install libssl-dev swig 
安装M2Crypt
      pip install M2Crypto

5.安装ShadowSocks-Python
      pip install shadowsocks

5.1
到https://github.com/jedisct1/libsodium下载libsodium
      ./configure
      make && make check
      sudo make install
或者https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905/10
      wget https://download.libsodium.org/libsodium/releases/LATEST.tar.gz
      tar zxf LATEST.tar.gz
      cd libsodium*
      ./configure
      make && make check
      sudo make install

      apt-get install lib32ncurses5 ia32-libs

编辑 /etc/ld.so.conf ,添加一行
      /usr/local/lib
实际在/etc/ld.so.conf.d/opencv.conf中已经加入了/usr/local/lib
运行ldconfig


6.创建配置文件
{
"server":"服务器ip",
"server_port":端口,
"local_address": "127.0.0.1",
"local_port":1080,
"password":"密码",
"timeout":600,
"method":"aes-256-cfb",
"fast_open": false,
"workers": 1
}

字段说明：

server：服务器IP
local_address：本地地址
local_port：本地端口
password：加密密码
method：加密算法
文件保存到 /etc/shadowsocks.json

最后都没用，改为：
https://www.feiyulian.cn/note/Using-help/186.html
https://hinine.com/install-and-configure-shadowsocks-qt5-on-ubuntu-linux/

    sudo add-apt-repository ppa:hzwhuang/ss-qt5
    sudo apt-getupdate
    sudo apt-getinstall shadowsocks-qt5




https://hub.docker.com/r/tensorflow/tensorflow/tags/
https://github.com/nlintz/TensorFlow-Tutorials  

jupyter远程访问设置
$jupyter notebook --generate-config

In [1]: from notebook.auth import passwd
In [2]: passwd()
Enter password: 
Verify password: 
Out[2]: 'sha1:ce23d945972f:34769685a7ccd3d08c84a18c63968a41f1140274'

vi ~/.jupyter/jupyter_notebook_config.py 
c.NotebookApp.ip='*'
c.NotebookApp.password = u'sha:ce...刚才复制的那个密文'
c.NotebookApp.open_browser = False
c.NotebookApp.port =8888 #随便指定一个端口

访问：http://address_of_remote:8888,或者:
在本地终端中输入ssh username@address_of_remote -L127.0.0.1:1234:127.0.0.1:8888 
便可以在localhost:1234直接访问远程的jupyter了。

安装PIL
pip install --no-index -f http://effbot.org/downloads/ -U PIL --trusted-host effbot.org



------------pytorch------------
torch.Tensor		多维数组
autograd.Variable	封装Tensor，跟踪其上的操作历史，拥有和Tensor一样的api
nn.Module		神经网络模块，封装了一些参数便于使用
nn.Parameter		一种Variable，向nn分配属性时会自动作为parameter注册
autograd.Function	定义了autograd的forward/backward，每个Variable操作，至少定义一个Function节点，连接到创建变量的function节点


定义网络
处理输入，调用backward
计算loss
更新权重

