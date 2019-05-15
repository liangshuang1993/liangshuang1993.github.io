### transfer learning in seq2seq model

图像领域中，transfer learning有很多的用处，那么是否可以利用transfer learning实现少语料的TTS？

---
首先介绍一篇单独训练seq2seq模型中的encoder和decoder，再finetune的模型

#### Unsupervised Pretraining for Sequence to Sequence Learning

__这篇文章在pretrain的阶段用的是unsupervised的data，即分别训练了两个language model作为encoder和decoder的初始权重，之后再用labeled的data进行fine-tune，为了预防过拟合，finetune的时候将seq2seq的目标函数和language model目标函数一起训练，结果优于supervised baseline。__

![](/papers/tts/10.png)

以机器翻译为例：
- 数据集A是source language，数据集B是target language。对每个数据集训练一个language model。
- 构建一个seq2seq model M，它的encoder和decoder的embedding和LSTM的前几层用上一步训练得到的权重初始化。decoder的softmax层也可以用B训练出来的language model的softmax层进行初始化。
- 用labeled dataset fintune整个模型M。
- 由于上一步训练时容易过拟合，造成catastrophic forgetting，因此需要对seq2seq模型和languange model进行联合训练，优化三者的loss，权重为1.

模型还有一些其他的优化：
![](/papers/tts/11.png)
- residual connections：传给decoder的softmax layer的输入向量是随机的，因为这些层是随机初始化的，这会给pretrained parameters带来随机的梯度，为了避免这种情况，我们增加了residual connections。
- multi-layer attention: attention不只用于第一层，还用于前几层的其他几层。

实验：
针对机器翻译问题，采用的数据集是WMT English->German task,先用一个language detection system去除了一些噪音样本，大概有4million样本。
language model的结构为一层4096维LSTM，之后hidden state投影到1024维。Seq2seq model为三层，第二三层是1000hidden units。

ablation study:

下图的base即为预训练encoder和decoder，并在loss中加入language model loss。
![](/papers/tts/12.png)

- 只预训练decoder比只预训练encoder好。
- 尽可能增多预训练
- 预训练softmax也很重要
- language model objective起到了很强的正则化作用。
- 用unlabeled data预训练非常重要。如果模型初始化用的LM是由parallel corpus的source part和target part训练的，相比base model的效果差距很大。

针对abstractive summarization问题，也做了类似的实验，这里的encoder和decoder 用的是同一个language model初始化，因为它的source和target的语言是相同的，LM是一层1024维的，seq2seq和上述实验配置一样，只是第二层用的1024个hidden units。但是跟机器翻译的结果还是有很大不同的。

![](/papers/tts/13.png)

- 只预训练encoder比只训练decoder更有效。一种解释是说预训练可以使得梯度往回传的更及时，这里不太清楚什么意思。
- language model的loss优化仍然起到了很强的正则作用。

和上述机器翻译的实验放在一起看，这两个实验是否意味这seq2seq模型中encoder和decoder都是可以预训练的，但是具体预训练哪一个是和任务强相关的，不过预训练总是可以一定程度上提升效果，并且用一个multi-task（language model）来进行正则化可以有效地减小过拟合。

文章后面还介绍了一些相关工作。

semi-supervised learning:
_Semi-supervised sequence learning_
_Learning to generate reviews and discovering sentiment_

*Exploit-ing source-side monolingual data in neural machinetranslation*添加了一个task

_Transfer learning for low-resourceneural machine translation_
_Multi-task sequence to sequence learning_
_Zero-resource translation with multi-lingual neural machine translation_
_On using monolingual corpora in neural machine translation_

---

#### Semi-supervised training for improving data efficiency in end-to-end speech synthesis

__这篇文章是希望可以利用一些易于获取的，unpaired text and speech data，进行finetune__

tacotron主要由两部分构成：encoder和attention-based decoder。可以认为encoder是用来接受文本信息，产生文本信息的表征，decoder拿encoder产生的表征作为condition，产生相应的acoustic representation(频谱)。原始的tacotron直接将encoder和decoder一起从头开始训练，而本文则是希望可以引入外部的textual, acoustic knowledge。

__encoder__
为了获取textual knowledge，本文利用了word vectors, language model。

首先将输入文本的每个单词转成word embedding，将word vector sequence加入下面两部分之一：encoder input或者encoder top，将这两者的向量记为conditioning location feature。
- encoder input代表了phoneme embedding sequence
- encoder top代表了encoder最后的输出
由于word vector和上述两种特征具有不同的time resolution，因此考虑下面两种结合方法：
- word vector concatenation. 下图解释的很清楚，可以将其认为是hard attention。
- conditioning attention head. 比如发音Thank you的时候，给定thank，有助于you的发音，但是上面一种方法没有利用到thank的信息。此方法利用attention，将conditioning location feature视为query，采用的是tanh based additive attention。(但是decoder的attention是否已经利用了这个信息呢？)
![](/papers/tts/14.png)

encoder的权重仍是需要从头训练的，因此这里只将这种方法称作encoder conditioning

__decoder__
tacotron的decoder需要同时学习acoustic representation和与文本表征之间的对齐关系。为了减轻工作量，这里利用speech data来预训练decoder。
在预训练的时候，decoder作为预测下一步的acoustic frame，因此不需要文本。在这个阶段，我们仅仅保持encoder权重冻结，将attention context vector替换为0vector。

预训练decoder之后，我们可以用成对的数据进行finetune。

模型存在的问题：
decoder pretraining和model finetune之间存在的mismatch：在pretrain的时候，decoder只是拿前一时刻作为condition，但是finetune时，还需要将文本表征作为condition。这里不知道应该怎么改进。

实验：
- 原始tacotron训练，当只有24分钟的音频时，产生的音频很差。12分钟时候，已经不能理解。
- encoder conditioning，用的是200B corpus Google News训练word embedding。最终生成128维。也可以用word2vec训练word embedding。
- decoder pretrain，用的是VCTK 44小时。这里finetune的是US accent，而decoder则是多人的British accent，也会造成mismatch。

MCD实验
- encoder top会比encoder input效果更好。
- concat比attention效果好，文章认为attention带来了更多的参数，导致数据不够。
- 只预训练decoder比预训练decoder+encoder conditioning效果还要好

![](/papers/tts/16.png)

---

#### Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesispeaker adaptation

这篇论文是google中的NIPS2018的论文，值得好好一读．

文章提出的模型主要包含三部分：
- speaker encoder network,用speaker verification task训练出的,这个网络所用的数据是另一个**独立的,无文本,含噪声**的数据集.只需要输入任一说话人几秒音频,即可以产生固定维度的embedding vector
- 基于tacotron2的网络,用上述的speaker embedding做condition,拿文本做输入,输出mel谱
- 基于wavenet的vocoder

这样得到的网络,可以模仿任意说话人(包含未见说话人),而且不需要很多有文本的音频

- synthesis network: 1.2K speakers
- speaker encoder networ: 18K speakers

文章Speaker adaptation in DNN-based speech synthesis using d-vectors和本文方法有些类似,但本文不需要中间的linguistic features,并且speaker embedding network不限制在一个固定的speaker集合中.并且发现实现zero-shot transfer需要几千个speaker,要比那篇文章多得多.

**speaker encoder**

采用了这篇文章的结构:_Generalized end-to-end loss forspeaker verification_

这个网络将任意长度的mel谱序列转成固定长度的**d-vector**. 训练的数据集都切割为1.6s,和speaker identity一起训练,不需要文本,mel谱维度为40维.

模型结构为:3层LSTM,768cells, 每层都要投射到256维.最后一帧的最上层的输出进行L2正则,得到最终的embedding.

在inference的时候,任意长度分为800ms大的窗口,50%的重叠.网络在每个窗口上面独立运行,最终结果进行平均和正则,得到最终的embedding.

**synthesizer**

和deepvoice2中的方法类似,构建一个多说话人tacotron2,每个时间步上,将encoder output和target speaker的embedding进行concat,和deepvoice2不同的是,本文只是简单将embedding传到attention layer就可以收敛.


对比了模型的两种变种:
- 用speaker encoder计算embedding
- Baseline, 为训练集中每个说话人优化一个固定的embedding.

这篇文章是用音素训练的,它指出音素可以使得收敛更快,对于不常见的单词和名词发音更好.模型用了预训练的speaker encoder(参数是冻结的),提取目标音频的speaker embedding,进行训练.训练的时候,不需要指定speaker identifier label.

模型的loss是L1+L2 loss, 实际中,发现模型对noisy training data的鲁棒性更好.

**实验**

在inference时,模型可以用任意的文本和与之不对应的音频进行训练.

对于tacotron2部分,用两个数据集进行了训练:
- 用VCTK(44小时,109speaker, 大部分英式英语)
- LibriSpeech(436小时,1172speaker, 大部分美式英语, 没有标点,来源是有声书,同一个说话人的tone和style也会有很大变化).用ASR将语音切分为短音频,将时长中位数从14s降到5s. LibriSpeech很多录音含有很严重的环境背景噪音,对target谱进行了降噪过程,speaker encoder的输入是没有降噪的.

由于VCTK的音频将干净,因此用ground truth训练wavenet也能很好的工作,但是对于Librispeech而言,必须要用tacotron2生成的mel谱训练wavenet.

speaker encoder用的数据集包含36M语句,18K美式英语,时长中位数3.9s.

1. naturalness

评估的语句是训练集中没有的100句话,一组测试用已见说话人,一组测试用未见说话人,每个人,随机选择一个长度5s的音频来计算speaker embedding.

![](/papers/tts/42.png)

LibriSpeech的MOS略低,可能是以下原因:缺少标点;背景噪声更高

未见说话人的MOS得分高了大概0.2,可能是随机选择reference utterance导致的结果.

测试中发现有时候会模仿reference speech的韵律,尤其是LibriSpeech多样化的数据中更常见,说明了speaker identity和prosody没有分解.


2. similarity

![](/papers/tts/40.png)

未见说话人的相似度较低.
speaker encoder使用北美口音语音训练的,accent mismatch容易降低VCTK的效果.

![](/papers/tts/41.png)

3. speaker verification

用113K speaker的28M句话,重新训练一个新的speaker encoder网络,用来评估.评估时用了VCTK里的11个人,LibriSpeech的10个人.每个人用了100句话测试.
实验表明,用大数据集如LibriSpeech训练时,效果较好.

4. exploration of effect of the speaker encoder

分别用了以下三种数据训练
- LibriSpeech Other, 461小时,1166说话人
- VoxCeleb, 139K句,1121说话人
- VoxCeleb2, 1.09M句,5994说话人

下分析了说话人对于speaker encoder的影响,为了防止其在小数据集上的过拟合,在小数据集(前两个)用的网络256维LSTM,64维线性,输出的embedding也是64维.

![](/papers/tts/43.png)


**结论**

通过利用迁移学习的方法,实现了多说话人TTS.speaker encoder所需要的数据更容易获取,它们不需要文本,且所需的质量较低. 不过,modeling speaker variation用了低维度的向量,限制了reference speech的处理.

----

