## A survey on style transfer in TTS

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

#### Exploring Transfer Learning for Low Resoure Emotional TTS


**该论文用的finetune的方法，每次生成一种情感tts模型**

https://arxiv.org/pdf/1901.04276.pdf

该论文指出目前很多TTS系统，如Wavenet，Tacotron，WaveRNN，Char2Wav，Deep Voice都需要很多数据，有一个**DCTTS**模型需要的数据少一些，DCTTS只依赖CNN，不用RNN，两个GPU训练了15小时。

文章提出情感数据集目前都很少，解决少量数据训练的方法有transfer learning，multi-task learning。

《Asr-based features for emotionrecognition: A transfer learning approach》中用了一个ASR作为特征提取器来做情绪识别。

在TTS领域中，也有人研究了transfer learning，《Transferlearning from speaker verification to multispeaker text-to-speech synthesis》用来做多说话人TTS。

- 模型结构：基于DCTTS
- 数据集：EmoV-DB，有英文的男女演员的语句，和法语男演员的语句。每个演员用五种情绪读，Neutral， Amused， Angry， Sleepy， Disgust
- 预处理：原始模型是用LJ-speech训练的，采样率22050Hz，本数据集一样；trim slience，将https://github.com/Kyubyong/dc_tts 的top_db改为了20dB；音频文件中有一些是非言语表达，如笑声，打哈欠，叹气声，手动选择了amused dataset中没有这种情况的156句，sleepy中的361句，由于amused dataset中数据集太少了，之后针对于amused dataset，手动删掉了剩余82句中的笑声，这样总共就有了238句。
- 训练过程：LJ-speech有23.9h，该文章指出，原先想只训练audio部分，即不训练text encoder，但是合成出来的韵律有问题，他认为这样的attention没有适用于新的speaking style，因此训练了全体的。<font color=#A52A2A size=3>不知道为什么我固定encoder完全不行，后续还要再做实验。</font>
- 实验：第一个实验评估了finetune的用处，测试了word accuracy；第二个实验测试了音频的质量，MOS。测试用的语句是Harvard sentences的100句。


后续可能会引入多说话人多情感。

----
#### Speaking Style Adaptation In Text-to-speech Synthesis Using Sequence-to-sequence Models With Attentions

finetune了tacotron，
数据：
原始模型Blizzard Challenge 2011（16hours）
新的数据： Hurricane Challenge speech data. normal style：2hours；Lombard style：30minutes。

注：lombard相关
lombard effect是指在嘈杂环境下，人们交流使用的风格，这种环境下产生的音频就叫做lombard speech，或者speech-in-noise，相比安静环境中的音频，这样的音频更大声，pitch更高，频谱倾斜更大，时间更长

---
#### Feature Based Adapation For Speaking Style Synthesis

__结合bottleneck feature和residual feature进行finetune，ICASSP文章__

为什么需要研究speaking style？
一种合适的speaking style可以生动地表达语音信号中包含的信息，增强人和机器的交流。同一种文本的语音在不同的speaking style下可能会表示不同的信息。

目前的adaptation主要分为三类：
- input feature-based adaptation and augmentation，如引入speaker codes，i-vector, d-vector，将他们与原始的文本特征结合在一起
- model based adaptation，保留模型的下面几层，只训练上面几层；学习hidden unit contribution。
- output feature transformation，如voice conversion。


这篇文章也是研究了finetune模型，指出现有的adaptation通常是adapt的是global level的信息，缺少局部信息。引入了以下两种信息：
- bottleneck feature
- residual feature
bottleneck feature的提取方法如下图左半部分所示,用疑问句数据集训练一个bottleneck DNN网络。虽然训练出来的效果要差于大网络，但是这样可以提取一些压缩后的信息。

![](/papers/tts/18.png)

residual feature的提取方法如上图右半部分所示，在frame level上，提取出要生成的疑问句的因素信息与用陈述句风格得到的模型生成的因素信息的差，作为style difference。训练过程：先用大量陈述句数据训练得到一个decDNN；用这个decDNN生成给定疑问数据集中指定文本的音频，将实际的音素特征减去生成的音素特征；再训练一个新的模型，rDNN，学习文本到上述因素差的映射。

训练方法仍是finetune的方法，只训练上面几层，如下图所示。

![](/papers/tts/17.png)

**实验**
原始模型：5小时中文，陈述句风格
adapted data：25分钟，疑问句风格

---

#### Emotional End-to-End Neural Speech synthesizer

**tacotron的基础上进行改进，引入emotion label；改进了tacotron的一些问题，如exposure bias problem和attention alignment的irregularity。**

在tacotron基础上注入了一个learned emotion embedding e:

![](http://latex.codecogs.com/gif.latex?h_t^{att}=AttentionRNN(x_t,h_{t-1}^{att},e))
![](http://latex.codecogs.com/gif.latex?h_t^{dec}=DecoderRNN(c_t,h_{t-1}^{dec},e))

模型结构如下图所示：

![model](/papers/tts/1.png)

其实就是将emotional label通过prenet之后，注入到attention RNN和decoder RNN。

参考《Listening while speaking: Speech chain by deep learning》，将**one-hot emotion label vector**注入attention RNN，具体方法如下：将prenet output concat之后，添加一层线性层保证维度和attention RNN input相匹配，decoder RNN也是类似的。

另外，本文还解决了exposure bias problem，这个问题会导致在长时间步之后，对齐有问题。因此这里采用了voiceloop中提出的semi-teacher-force training(STFT)，送入decoder的不再是ground truth，而是![](http://latex.codecogs.com/gif.latex?0.5(y_{t-1}+\bar{y}_{t-1}))，不过voiceloop原始的方法是还添加了一个噪声项。<font color=#A52A2A size=3>但是这样做的话速度会不会特别慢？？仅仅添加一个噪声项是否可行？？？</font>


关于attention alignment的预测：
在用非情绪语音进行训练时，发现attention alignment的中间部分会出现一些irregularities。又由于alignment的shapness会影响到生成音频的质量。attention alignment有两个输入，一个是从attention RNN的hidden state，一个是encoder的encoding。基于此，提出了以下两种改进：
- utilization of context vector，![](http://latex.codecogs.com/gif.latex?h_t^{att}=AttentionRNN(x_t,h_{t-1}^{att},c_{t-1})),即利用上一时刻的context vector，将其也输入到attention中。
- residual connections in CBHG.在text encoder的CBHG模块加入residual，![](http://latex.codecogs.com/gif.latex?y_t=x_t+BiRNN(x_t,h_{t-1}))。具体而言，text encoder 是由CBHG实现的，在CBHG的最后阶段，有一个BiRNN，一个严重的问题是这个RNN的hidden state大小是固定的，如果sequence太长，hidden state很容易就不能保存整个序列完整的信息，实验表明，语句过长的时候attention alignment从中间开始就会模糊，因此本文推测是RNN容量不够，因此这里采用residual connection。


实验：
用了一个韩语数据集<text, audio, emotion label>来做数据集，约21小时。每个女演员用六种不同的情绪来读：neutral，angry，fear，happy，sad，surprise。共21小时。emotion vector用的是64隐藏层，dropout ratio=0.5

---

#### Principles for learning controllable TTS from annotated and latent variation

**interspeech 2017**

语音其实含有很多丰富的变量，不能由文本代替，比如speaker identity，speaker rate，emotion，adjustments to enhance communication with the listener（speaking style，prosody，emphasis，entrainment等），环境因素（如背景噪音）。

但是目前的TTS系统没有对这些变量进行建模，输出的音频都是平均的，平滑的。为了减少这些变量的影响，通常训练数据为单一说话人在安静的录音室中录制一致的语音。但是更好的方式是给variation打标签并学习，但是成本又太高了。目前有一些工作指出多说话人TTS甚至可以产生比单一说话人更好的音质，说明我们可以利用这种diversity。

__标签未知的情况下，一个简单的方法是与synthesiser一起学习control parameters。联合优化network weights和unknown control input可以用深度学习反向传播很容易地进行联合优化，称为discriminant condition codes（DCC），已经用来做speech recognition和speech synthesis to new speakers，或者learned control vectors。__
<font color=#A52A2A>后续需要看下这些论文</font>

Sentence-level control vectorsfor deep neural network speech synthesis用到了儿童有声书做数据集，和本文方法有点类似，不过论文是15年的了。

---

### VAE相关

---
#### Learning Latent Representations for Speech Generation and Transformation

https://arxiv.org/pdf/1704.04222.pdf

**用VAE分解隐变量**

speech waveforms有复杂的分布，体现出variance，linguistic content，speaking style，dialect，speaker identity，emotional state，environment，channel effect。

输入数据x为固定长度的语音，TxH，和图片不同的是，语音只在时间轴上面有**平移不变性**，因此采用了和文献《Learning word-like units from jointaudio-visual analysis》相类似的方法，第一层卷积利用的是1xF的卷积核，后面几层用的是wx1的,文献《Unsupervised representa-tion learning with deep convolutional generative adversarial networks》指出，用时间轴上stride size>1替代pooling做下采样。最后一层卷几层的输出展平，送入全连接层，再之后送入gaussian parameter layer对隐变量建模。
对于高斯层，没有使用激活函数，表示对输出不做bound，但是其它层，选择了tanh。除了高斯层，其他都用了BN。

**具体分解**
属性用记号a表示，属性的值用记号r表示。
假设一：conditioning on some attribute a being r，比如phone being /ae/，z的先验分布依然是高斯。
假设二：定义k个独立的属性。

假设我们希望修改语句$x^{(i)}$的属性$a_k$,如说话人从$r_s$转到$r_t$，则latent attribute shift $v_{r_s-r_t}=μ_{r_t}-μ_{r_s}$。
可以通过以下方法修改$x^(i)$:
$z^{(i)}\sim{q_\phi}$

**实验**
TIMIT 630说话人，每人10句话。
VAE和AE对比。
修改的是phone和说话人，没有涉及到情感。


----


#### LEARNING LATENT REPRESENTATIONS FOR STYLE CONTROL AND TRANSFER INEND-TO-END SPEECH SYNTHESIS

__tacotron2+VAE，进行style transfer，效果胜于GST，ICASSP 2019__
__Expressive speech synthesis via modeling expressions with  variational autoencoder interspeech2018是用VAE+voiceloop：__

- 本文可以进行style control
- 本文是端到端的



模型结构：

![](/papers/tts/3.png)

recognition model用的是GST的reference encoder+全连接得到隐变量z的期望和方差。为了匹配维度，z需要先送入一个FC，之后加上text encoder state。

实验：105 hour Blizzard Challenge 2013，http://home.ustc.edu.cn/~zyj008/ICASSP2019/

---

#### Expressive speech synthesis via modeling expressions with  variational autoencoder

__VAE+voiceloop   interspeech 2018__

https://www.kakuzawa.com/VAELoopDemo/

模型：
![](/papers/tts/4.png)

实验： 

- 多说话人：VCTK，5hours of 21 speakers
- 10hours Blizzard Challenge 2012


---

#### Modeling and Transforming Speech Using Variational Autoencoders

#### Voice conversion from non-parallel corpora using Variational auto-encoder

#### Neural Discrete Representation Learning



#### Voice conversion from unaligned corpora using variational au-toencoding wasserstein generative adversarial networks,

---

#### Robust And Fine-grained Prosody Control of End-to-end Speech Synthesis

ICASSP2019

这篇文章主要解决两个问题：
- GST模型没有考虑具体时刻下如何控制Prosody，他们的Prosody embedding是固定长度的；
- 如果source speaker和target speaker的pitch range差距太大，结构不够robust。

这里提到了一个Prosody标签。ToBI，但是有文章说这个标签有很多问题。

__Baseline Model__:基于tacotron的更改。
speaker embedding vector s，用了embedding lookup layer。
$e_{1:l_e}=Encoder(x_{1:l_e})$
$α_i=Attention(e_{1:l_e},d_{i-1})$
$e_i^{'}=\sum_jα_{i,j}e_j$
$d_i=Decoder(e_i^{'},s)$
其中，e，p，d分别是encoder state，Prosody，decoder state
和GST类似，这里的reference speech也要通过reference encoder，先经过2D卷积，再经过单层GRU，最后一个GRU的输出是固定长度的Prosody embedding p。如果我们将每个GRU的输出拼起来，就可以得到不定长的prosody embedding $p_{1:N}$
__proposed model__:speech-side control和text-side control，从而进行prosody control。变长的prosody embedding分别送入decoder或者encoder。因此，需要做align和downsample，使得prosody embedding匹配decoder的time-step和encoder的time-step。具体而言，decoder state长度和prosody embedding具有线性关系，直接下采样即可，而encoder state和Prosody embedding则没有关系，因此这里用了scaled dot-product attention，encoder state作为query，prosody embedding作为key和value。

__在tacotron结构上做了以下更改__：CoordConv做第一层卷几层。


----
#### EMPHASIS:An Emotional Phoneme-based Acoustic Model for Speech Synthesis System

__依赖标注，参数式，interspeech2018，https://ttsdemos.github.io/__

<font color=#A52A2A>introduction中提到了两篇论文7,8是17和18年的，需要看下</font>

EMPHASIS采用的是cascade model，更加稳定。
TTS系统的linguistic feature通常包含phoneme identification，tone（如中文），stress（如英文），prosody structure，syntactical structure和emotion type，这些特征都要用one-hot vector表示。由于phoneme-related features，包括phoneme identification，tone，stress，对于决定duration和acoustic feature而言是很强的特征，而emotional，prosodic feature相对较弱，这就会导致训练时，容易忽略后者，因此我们将他们分成两组，通过网络加强后者。
![](/papers/tts/2.png)

__数据__：14小时汉语+3.5小时英语，一个说话人。30%疑问句，10%感叹句。（标贝数据目前只有570句问句，306句感叹句）标签：phoneme identity(specifically, one-hot  representation  of initials  and finals of Mandarin Pinyinandphonemes of English），tone of Mandarin syllable, stress of English syllable;the emotional & prosodic  linguistic features  include prosodic break,  prosodic level, word segmentation of Mandarin, syntactic level, part-of-speech tag and type of emotion, etc.
__实验__：ABX test，EMPHASIS，Tacotron，Bi-LSTM，Concatenative，样音中断句韵律更自然，主要为疑问句，没有明显情绪。

![](/papers/tts/abx.png)

---
#### Multi-reference Tacotron by Intercross Training for Style Disentangling,Transfer and Control in Speech Synthesis

__投稿interspeech2019，在GST-tacotron基础上引入了multi-reference encoder，效果比较好__


该论文introduction中提出speech style transfer可以分为两类：supervised way：将attributes's id作为另外的model input，这在multi-speaker TTS中很有效，但是他不能处理复杂的style如emotion或者prosody；unsupervised way：将E2E models和reference encoder结合到AE，或者VAE中，可以将对任意complex style建模，投射到一个continuous latent space中。

另一个很重要的问题是：如何分解style？

本文的贡献：
- introduce a multi-reference encoder to GST-Tacotron
- introduce intercross training to extract and seperate different classes of different classes of speech styles.

定义：
- style class：speaker，emotion，prosody，可以拓展到其他方面
- style instance：**speaker class**有300个不同的speaker，**emotion class**有happy，sad，angry，fear，confuse，surprise，neutral； 
**prosody class**:
    - news: relative fast, normal
    - story: many transitions and breaks
    - radio: relative slow, deep and attractive voice
    - poetry: slow, obeys ruls of rhyming
    - call-center： relative fast, sweet

模型：
每个sub-encoder只负责对一种style进行建模，有N个encoder时，在训练的时候也要输入N个音频，其中第n个音频的的第n个属性应当与target的第n个属性相同。通过这些reference encoder可以得到N个style embedding,最后将这些embedding拼到text encoder的输出上，再送入tacotron中。


__intercross training：__

![](/papers/tts/7.png)

![](/papers/tts/8.png)

直接用这个模型训练时，模型收敛不好，因为被这种多个encoder style embedding的概念混淆了，因此我们这里引入了style classification task，优化![](http://latex.codecogs.com/gif.latex?L_{classification})。除此之外呢，我们引入了一个orthogonality constraint来加强style embeddings之间的独立性：

![](http://latex.codecogs.com/gif.latex?L_orthogonality=\sum_{ij}||H_i^T-H_j||_F^2)

最终的loss:

![](http://latex.codecogs.com/gif.latex?L=L_{IT}+\beta\dot(L_{classification})+\gamma\dot(L_orghogonality))



__实验数据及结果__
- style distangling: 给定一个reference audio，每个encoder可以得到出一个特定的风格。
- style transfer: 每个分解后的style embedding可以进行自由组合，得到新的音频。
- style control: 每个encoder得到的style embedding进行线性差值，可以进行style control
    ![](http://latex.codecogs.com/gif.latex?SE_{to}=SE_{from}+\alpha(SE_{to}-SE_{from}))
- random sampling: ![](http://latex.codecogs.com/gif.latex?SE_{random}=\sum_{k=1}^Ksoftmax(\alpha_k)STE_k)

    _STE指的是sub-encoder的style token embedding，_(http://latex.codecogs.com/gif.latex?\alpha\sim N(0,1))

数据：BAIDU Speech Department，conditioned on Mandarin phonemes and tones, __110hours,178 females,122 males__

- single-reference: 
    - style distangling: 实验选定了20个speakers, 用t-SNE进行可视化。

    ![](/papers/tts/9.png)

    - style transfer: GST得到的音频长度会和reference audio的长度接近而不是和文本成正比，但是本文提出的方法则解决了这个问题。

    - style control: 可以控制移动的比例。

    -few shot/one-shot learning: 对未见reference audio进行了实验。one-shot的接受率有20%。之后用这些失败的音频进行fine-tune,最终得到了100%的接受率。文中提到保持text encoder固定效果最好。

- multi-reference: 2-reference models to control two style classes: speaker and prosody. 30 hours, 27 speakers, 5 prosodies.

    - style distangling: 每个speaker选择了100句话，用t-SNE进行可视化。
    ![](/papers/tts/10.png)

---

#### Neural TTS Stylization with Adversarial and Collaborative Games


__ICLR 2019，在GST的基础上利用gan解决style和content的分解问题__

定义GST模型中content为$x_{txt}$, style为$x_{aud}$,$x_{aud}$中包含有style component s和other factor z，如和$x_{txt}对应的$verbal content，因此我们的目标是将z和s分解出来。不同于GST训练时每次都输入成对的数据，这里的数据是将一个文本和两句语音同时训练。![](http://latex.codecogs.com/gif.latex?(x_{txt},x^+_{aud},x^-_{aud})) 其中，![](http://latex.codecogs.com/gif.latex?x^+_{aud})是指和![](http://latex.codecogs.com/gif.latex?x_{txt})成对的音频，![](http://latex.codecogs.com/gif.latex?x^-_{aud})是指和![](http://latex.codecogs.com/gif.latex?x_{txt})不成对的音频，即随机采样得到的音频。

TTS可以是做是一个cross-dimain问题，即将source domain,文本所在的domain，映射到target domain，音频所在的domain上，TTS学习两者的映射F。风格TTS一般是指给定一种风格，计算F。如给定一个reference audio，得到文本到音频的映射。

GST的模型结构如下：
给定![](http://latex.codecogs.com/gif.latex?(x_{txt},x_{aud}))，我们可以进行采样：

content: ![](http://latex.codecogs.com/gif.latex?c_{1:T}\sim{q_\psi(c_{1:T}|x_{txt})}), style:![](http://latex.codecogs.com/gif.latex?s\sim{q_\phi(s|x_{aud})}), output: ![](http://latex.codecogs.com/gif.latex?\tilde{x}\sim{p_\theta(x|c_{1:T},s)})

![](http://latex.codecogs.com/gif.latex?p_\theta(x|c_{1:T},s))是decoder network决定的似然函数，同时网络中还有一个encoder ![](http://latex.codecogs.com/gif.latex?Enc_c)将![](http://latex.codecogs.com/gif.latex?x_{txt})投影到相应的content components上。![](http://latex.codecogs.com/gif.latex?Enc_s)来表示![](http://latex.codecogs.com/gif.latex?q_{\phi(s|x_{aud})})，因此，可以将随后的似然函数写成如下形式：

![](http://latex.codecogs.com/gif.latex?x\sim{p_{\theta}(x|Enc_c\(x_{txt},x_{aud}\))})，可以将目标函数写为如下形式：

![](/papers/tts/19.png)

本文方法如下：

![](/papers/tts/5.png)

参考图像 https://arxiv.org/pdf/1611.02200.pdf

loss函数：

![](/papers/tts/6.png)

- 第一个loss的作用：对于所有的![](http://latex.codecogs.com/gif.latex?c,x^+)，以及![](http://latex.codecogs.com/gif.latex?c,x^-)，生成器G的所用是生成新的音频，将音频与文本c送入判别器D3，使得判别器认为他们全为真。但是判别器优化的目标则是使得他们的值为0
- D的loss的作用：对于![](http://latex.codecogs.com/gif.latex?c,x^+)，他们生成的音频与c送入判别器D1，使得判别器的值尽可能大，对于![](http://latex.codecogs.com/gif.latex?c,x^-)，他们生成的音频与c送入判别器D2，使得判别器的值尽可能大，对于![](http://latex.codecogs.com/gif.latex?c,x^+_{aud})使得D3的值尽可能大。

一共有三个D：
- 1: fake from paired input
- 2: fake from unparied input
- 3: real audio sample

但是目前这些loss还不足以使得模型产生真实分布。比如，![](http://latex.codecogs.com/gif.latex?x^-_{aud})，G可能产生非正确s下的音频，但是D还是可能接受它，因为认为这个s和![](http://latex.codecogs.com/gif.latex?\tilde{x}^+)的s不同。因此这里还添加了另外两个loss，style loss和reconstruction loss。

**style loss**：图像中有用gram matrix来捕捉artistic style，这里也希望可以用gram matrix来捕捉时频中的一些局部统计信息，表征声音的low-level特征，如loudness，stress，speed，pitch等。定义![](http://latex.codecogs.com/gif.latex?X,\tilde{X})分别为reference和合成音频的mel谱。可以用内积来计算gram matrix W和G。

![](/papers/tts/20.png)

style loss:
![](/papers/tts/21.png)

**reconstruction loss**:
- mel谱的reconstruction loss：
    ![](/papers/tts/23.png)
- latent space的reconstruction loss，这里引入了一个分类器C，C和![](http://latex.codecogs.com/gif.latex?Enc_s)共享所有层，只是最后加了一个全连接层，输出类别。
    ![](/papers/tts/22.png)


数据：
- EMT-4 24 hours, happy, sad, angry andneutral, same speaker
- VCTK 44 hours

实验： 没有测MOS，样音
- content vs. style disentanglement ability
- effictiveness of style modeling
- controllability

---
#### Parrotron: An End-to-End Speech-to-Speech Conversion Model and its Applications to Hearing-Impaired Speech and Speech Separation

# google投interspeech2019的文章

**text-independent, many-to-one voice conversion task,输入是paired data,预处理后的特征为mel谱**

**encoder**
2CNN-->BCLSTM(bidirectional convolutional LSTM)-->3BLSTM-->linear projection

**decoder**
用的tacotron2的decoder,后面可以接griffin-lim/WaveRNN

**multitask training with an ASR decoder**

![](/papers/tts/58.png)


**实验**
训练的数据是多个不同说话人的音频,对应到一个指定说话人.后者的音频则是由google的parallel wavenet-based TTS提供.

30000小时数据,24百万句.

不过比较有意思的是,这个可以用于语音分离.有一个实验是重构出多说话人混杂的音频中声音最大的那个说话人,实验最多有8个人.