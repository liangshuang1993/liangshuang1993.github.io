介绍一些语音情绪识别的论文了.

首先汇总下见过的数据集:

IEMOCAP: https://sail.usc.edu/iemocap/
中文数据:https://github.com/AkishinoShiame/Chinese-Speech-Emotion-Datasets
RAVDESS: https://zenodo.org/record/1188976/?f=3#.XNuw3UO-lhE 7356 files, English, 8 emotions, two emotional intensities, speech & song, 319 raters. 

https://link.springer.com/content/pdf/bbm%3A978-90-481-3129-7%2F1.pdf

#### Speech Emotion Recognition Using Spectrogram & Phoneme Embedding

这篇论文是interspeech2018的论文

语音识别实际上是一个分类问题,这篇文章用到了phoneme sequence和spectrogram.

**model1: CNN model with phoneme input**

这里用phoneme作为输入,而不是用的text,因为phoneme能代表更多的信息.phoneme集合有47个,用IEMOCAP数据训练word2vec模型,phoneme embedding用了100维.

![](/papers/tts/44.png)

文章指出,可以从下图发现phoneme embedding含有发音信息.

![](/papers/tts/45.png)

用G2P工具将text转成了phoneme,用同样的word2vec来生成embedding.用text来生成phoneme的好处是更容易获得大量的数据,我们用google的1 billion(https://arxiv.org/pdf/1312.3005.pdf)的数据来生成phoneme embedding.虽然用text生成的phoneme embedding更稳定,但是我们这篇文章用的是speech based phoneme embedding.

本文使用的模型如下图所示

![](/papers/tts/46.png)

先是有几层CNN,维度为$h^p_i\times L^p_e=(3,9,13,17\times100)$,之后是max pooling和FCN,最终送入了softmax.另外网络中还有dropout和batch normalization.

**model2: CNN for spectrogram features**

这里用的mel谱.最长的语音是6s.mel谱输入维度是128\*256.模型首先是4个CNN,每个CNN的kernel size分别是12\*16, 18\*24, 24\*32, 30\*40. 之后每个CNN后面跟一个max pooling.pool size是各自feature map长和宽的一半,即提取出来4个特征. 之后将其展平送入3层FCN,维度为别是400和200.模型中也用到了dropout,batch normalization.全连接层第二层没有用dropout.CNN和FCN都用了relu做激活函数.最后仍是softmax.优化器是adadelta.

![](/papers/tts/47.png)

**model3: multi-channel CNN model with phoneme and spectrogram features**

这个模型中,结合了phoneme和mel谱.由于输入的维度不同,因此这里采用了不同的CNN channel.两者在经过CNN之后的输出分别送入两个全连接,全连接的输出拼起来,在正则化后送入第二层全连接.

![](/papers/tts/48.png)

![](/papers/tts/49.png)



----

#### Automatic speech emotion recognition using recurrent neural networks with local attention

ICASSP2017文章

传统的方法通常是依据特征的,现在则有了很多利用深度学习的方法.

![](/papers/tts/50.png)

在很多SER数据集中,通常是在整句话的级别上给出一个label,但是一句话通常含有很多静音片段,在很多情况下,一句话中只有几个字是有情绪的.

本文中,结合了LSTM和attention,使得网络可以集中在句子中情感突出的部分.有了attention模型,网络可以忽略掉静音部分和句子中不含情感的部分.


网络结构:

首先将所有的输入特征经过一层全连接和一层RNN层.

每个时间t,计算attention parameter vector u 和RNN的输出$y_t$的内积,之后通过softmax计算权重:

$\alpha_t=\frac{exp(u^Ty_t)}{\sum^T_{\tau=1}exp(u^Ty_{\tau})}$

$z=\sum^T_{t=1}\alpha_ty_t$



![](/papers/tts/51.png)


实验结果,仍用的是IEMOCAP:

![](/papers/tts/52.png)

----
#### Improving speech emotion recognition via Transformer-based Predictive Coding through transfer learning

用的也是,准确率65.03%

训练步骤分为两部分:
- multi-layer Transformer来预测下一时刻的features,如图a所示
- 用finetune和hypercolumns来训练目标任务的分类器
    - 首先将上一步预训练的模型的最后一层替换为emotion-specific layer,fine-tune
    - 利用bottleneck feature

![](/papers/tts/53.png)

1. predictive model pre-training

模型先用的是没有标签的,general-domain的数据训练,之后再用目标任务数据进行训练.

给定一段没有标签的mel谱,$X=[x_1, x_2, ...,x_n]$,n是帧数,$x_k$是第k帧特征,对$x_k$的预测,$P(x_k)$可以由下式得到:

$$h_0=C_kW_e+W_p$$

$$h_l=transformer\_Block(h_{l-1})  l\in [1,L]$$

$$P(x_k)=h_LW^T_L$$

其中$C_k=[x_1,x_2,...,x_{k-1}]$是$x_k$的context vector,L是层数,$W_e$是输入的embedding matrix,$W_p$是位置嵌入矩阵,$W_L$是输出变换矩阵.

最大似然函数: $L_1(X)=\sum_ilogP(x_i|x_1,x_2,...,x_{i-1};\theta)$

2. transfer learning for target task classifier

只需要在这个阶段使用有标签的数据.

A. fine-tuning

将上面$P(x_k)=h_LW^T_L$式子替换为$P(y|x_1,x_2,...,x_k)=softmax(h_LW^T_y)$

最大似然函数: $L_2(C)=\sum_{x,y}logP(y|x_1,x_2,...,x_n)$

B. Hypercolumns

参考文章Deep contextualized word representation(高引用文章),将fine-tuned transformer的所有输出拼成一个向量,送入最终的classifier.

将mel谱和Transformer的输出结合(concat/add),送入分类器中.
分类器采用了三种:SVM,random forest,attention-based LSTM.

数据集用的是IEMOCAP,general-domain数据用的VCTK.


**实验**
Transformer个数为2,头数为5,L1 loss.

![](/papers/tts/54.png)
