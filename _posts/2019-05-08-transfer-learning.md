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