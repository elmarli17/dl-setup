Number plate recognition with Tensorflow
http://matthewearl.github.io/2016/05/06/cnn-anpr/

opencv
《OpenCV References Manuel》
《OpenCV  2 Computer Vision Application Programming Cookbook》
《OpenCV Computer Vision with Python》

=======
http://blog.csdn.net/l281865263/article/details/46378149
卷积神经网络
http://blog.csdn.net/peaceinmind/article/details/50409354
卷积层感受野和坐标映射
https://zhuanlan.zhihu.com/p/24780433
原始图片中的ROI如何映射到到feature map
http://blog.cvmarcher.com/posts/2015/05/17/cnn-trick/
Concepts and Tricks In CNN(长期更新)
http://blog.csdn.net/a819825294/article/details/53425108
深度学习（DL）：卷积神经网络（CNN）：从原理到实现
http://blog.csdn.net/a819825294/article/details/53393837
神经网络从原理到实现


Cv图像处理 函数库
http://wiki.opencv.org.cn/index.php/Cv%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86


guohaisheng推荐的资料

[Teaching Your Computer To Play Super Mario Bros. – A Fork of the Google DeepMind Atari Machine Learning Project](
http://www.ehrenbrav.com/2016/08/teaching-your-computer-to-play-super-mario-bros-a-fork-of-the-google-deepmind-atari-machine-learning-project/?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more]
[Welcome to the stanford Deep Learning Tutorial! ](http://deeplearning.stanford.edu/tutorial/?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)
(https://github.com/oxford-cs-deepnlp-2017/lectures)
(https://medium.com/learning-new-stuff/machine-learning-in-a-year-cdb0b0ebd29c#.k4sq4ihbj)[https://medium.com/learning-new-stuff/machine-learning-in-a-year-cdb0b0ebd29c#.k4sq4ihbj]
[Image Completion with Deep Learning in TensorFlow](http://bamos.github.io/2016/08/09/deep-completion/?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)
[[机器学习原来这么有趣！第四章：用深度学习识别人脸-英文版](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78#.geyj91s6e)

[机器学习原来这么有趣！第一章：全世界最简单的机器学习入门指南](https://zhuanlan.zhihu.com/p/24339995)
[机器学习原来这么有趣！第二章：用机器学习制作超级马里奥的关卡](https://zhuanlan.zhihu.com/p/24344720)
[机器学习原来这么有趣！第三章:图像识别【鸟or飞机】？深度学习与卷积神经网络](https://zhuanlan.zhihu.com/p/24524583)
[机器学习原来这么有趣！第四章：用深度学习识别人脸](https://zhuanlan.zhihu.com/p/24567586)
[机器学习原来这么有趣！第五章：Google 翻译背后的黑科技：神经网络和序列到序列学习](https://zhuanlan.zhihu.com/p/24590838)
[机器学习原来这么有趣！第六章：如何用深度学习进行语音识别？](https://zhuanlan.zhihu.com/p/24703268)
[Machine Learning is Fun Part 7: Abusing Generative Adversarial Networks to Make 8-bit Pixel Art](https://medium.com/@ageitgey/abusing-generative-adversarial-networks-to-make-8-bit-pixel-art-e45d9b96cee7#.l89b4x1tx)



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
或者
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

