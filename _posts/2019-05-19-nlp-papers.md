最近打算读几篇比较有名的NLP论文。

https://blog.csdn.net/manmanxiaowugun/article/details/83794454

#### Deep contextualized word representations

Pre-trained word representation在很多NLP问题中都起到了很重要的作用.理想的word representation应该:1) 包含单词使用的复杂特征(如语法,语义); 2) 不同的文本上下文中如何使用(一词多义); 这篇论文,介绍了一种**deep contextualized word representation**

这篇论文与传统方法的最大不同在于,它的embedding是基于整个输入给出来的.最终发现higher-level LSTM state捕捉context-dependent的信息(他们可以直接用于supervised word sense disambiguation tasks),lower-level LSTM state可以捕捉句法信息(可以用来part-of-speech tagging).利用ELMo的表征,可以提高其他任务的性能.

**Bidirectional languange model**

给定一个序列$(t_1, t_2, ...,t_N)$,一个前向的language model可以建立以下模型:

$$p(t_1, t_2, ...,t_N) = \prod _{k=1}^Np(t_k|t_1, t_2, ...,t_{k-1})$$

在最近的几篇论文中,首先使用了character-level的RNN或CNN,计算得到上下文无关的词向量$x^{LM}_k$,然后将此向量输入进L层的前向LSTM,在每个位置k,每个LSTM会输出一个$x^{LM}_{k,j}, j=1,...,L$,用最顶层的LSTM输出,即$x^{LM}_{k,L}$,再加上一个softmax层,来预测下一个词.

后向的语言模型和前向语言模型类似:
$$p(t_1, t_2, ...,t_N) = \prod _{k=1}^Np(t_k|t_{k+1}, t_{k+2}, ...,t_N)$$

后面仍和前向LM类似,backward LSTM会输出$x^{LM}_{k,j}, j=1,...,L$

最终模型希望可以最大化一大似然:

![](/papers/tts/66.png)


**ELMo**

ELMo是双向语言模型的多层表示的组合.对每个token $t_k$, 一个L层的biLM计算得到2L+1个representations.
![](https://www.zhihu.com/equation?tex=R_k+%3D+%5C%7BX%5E%7BLM%7D%2C%5Coverrightarrow%7Bh%7D%5E%7BLMj%7D_%7Bk%7D%2C+%5Coverleftarrow%7Bh%7D%5E%7BLMj%7D_%7Bk%7D%7Cj%3D1%2C...%2CL%5C%7D%3D%5C%7B%7Bh%7D%5E%7BLMj%7D_%7Bk%7D%2C+%7Cj%3D1%2C...%2CL%5C%7D)

其中,![](https://www.zhihu.com/equation?tex=%5C%7B%7Bh%7D%5E%7BLMj%7D_%7Bk%7D%5C%7D%3D%5B%5Coverrightarrow%7Bh%7D%5E%7BLMj%7D_%7Bk%7D%3B%5Coverleftarrow%7Bh%7D%5E%7BLMj%7D_%7Bk%7D%5D)

ELMo将所有R的输出合并成一个vector,$ELMo_k=E(R_k, \Theta_e)$,在最简单的情况中,ELMo值选择最上层L层,$E(R_k)=h^{LM}_{k,L}$, 类似于TagLM和CoVe模型.一般化地,我们可以计算所有层输出的加权和:

$$ELMo_k^{task} = E(R_k, \Theta^{task}) = \gamma^{task}\sum^L_{j=0}s^{task}_jh^{LM}_{k,j}$$

其中$s_{task}$是softmax化的权重,$\gamma$是缩放因子,假设每个biLM的输出具有不同的分布,$\gamma$在某种程度上来说相当于在weighting前对每一层biLM进行了layer normalization.

**Using biLMs for supervised NLP tasks**
给定一个pretrained biLM和一个特定NLP任务的supervised结构,只需要在每个单词上运行biLM,记录所有层的输出,之后,让task model 学习这些representation的线性组合即可.

给定一个序列$(t_1, ...,t_N)$,每个$t_k$可以用pretrained word embedding得到与context无关的表示$x_k$ 或者是character-based representation.之后ELMo可以得到context-sensitive representation $h_k$,通常是用BiRNN,CNN或者前向网络.

之后如何使用ELMo的词向量呢?

首先将biLM的权重冻结,将$ELMo_k^{task}, x_k$拼在一起,送入目标RNN中.

另一种方式是将task RNN的输出替换为$[h_k;ELMo_k^{task}]$,这在SNLI,SQuAD任务上效果非常好.


另外呢,在ELMo中也加入了一些正则:dropout和参数的L2正则.

**Pretrained bidirectional language model architecture**

 这里的pretrained LM借鉴了Exploring the lim-its of language modeling里面的结构,只不过做了一些适用于双向的改进,在LSTM间加了残差.最终模型使用的是L=2 biLSTM(4096 units), 512 projections, 1层2层之间有残差链接.context insensitive type representation用的是2048 character n-gram CNN+2层highway,之后线性映射到512维.

 用1B的单词训练了10个epoch.就预训练好了这个biLM,之后就可以用做其他下游任务了.有时候在下游任务训练的时候,也对biLM进行了finetune可以取得更好的效果.(根据论文附录所言,这里的finetune指的是忽略掉目标数据集的label,训好之后,仍需要固定这部分权重)


 论文从 Question answering, Textual entailment, Semantic role labeling, Coreference resolution, Named entity extraction, Sentiment analysis 六个任务来验证, 都取得了提升.


**Analysis**

用控制变量法比较了不同参数的重要性.

(λ是L2正则的系数)

- 用各层的加权平均要比只用最后一层效果好,正则化系数不能太大.
- 前面也提到过,在某些任务中,将ELMo输出与task model的RNN输出拼起来,效果会更好,从表3可以看到,放在不同位置的效果. 可以看出SNLI和SQuAD效果更好,这可能是应为这两者都在biRNN后面使用了注意力机制,所以将ELMo放在这里,可以使得模型直接注意到biLM的表征.

![](/papers/tts/67.png)


那么ELMo到底学到了什么信息呢?

由于加入ELMo比淡出使用word embedding效果要好,那ELMo一定利用上下文信息学到了特征.

这里通过取ELMo的单层出来的词向量加入模型与其他不同的词向量加入模型进行对比.(ELMo加入的模型设计简单).这种情况下依然可以取得好的效果,说明ELMo确实学到了很多.

不同层输出的数据具有不同的意义,使得ELMo效果好.


**sample efficient**

加入ELMo后,训练时间大大减少了.同时使用的数据也减少了.

![](/papers/tts/68.png)





----
#### GPT: Improving Language Unstanding by Generative Pre-Training

很多NLP模型都是在有标签的数据集上进行训练的，那么如何有效地利用无标签数据集呢？这篇论文提出了一种半监督的学习方法：首先用无监督的方式进行pretrain，之后再在特定task上进行finetune。

GPT和ELMo有两大不同:
- 模型结构不同:ELMo用的是独立训练的从左到右和从右到左的多层双向LSTM的输出,而GPT用的则是多层transformer decoder.
- 在下游任务中的应用方式不同:ELMo将embedding作为额外的特性,而GPT在end task中会finetune同样的模型.



![](https://github.com/huggingface/pytorch-openai-transformer-lm/raw/master/assets/ftlm.png)


**第一阶段** 

第一阶段使用绿色中左侧的那个模块，训练语言模型，采用了transformer decoder，内部含有12个相同的transformer block。在最后一个block后跟一层全连接和softmax进行text prediction，预测下一个token的概率。目标函数为L1

**第二阶段**


第二阶段使用绿色中右侧的那个模块，在最后一个block后跟一层全连接和softmax构成task classifier，预测每个类别的概率。（这里的task是text classification）目标函数为L2.

这里发现将第一阶段的模型作为一个额外的模型可以加快模型收敛，提高模型的泛化能力。因此，将目标函数改为L2+λL1

**根据task更改模型的输入**

上面介绍的模型主要适用于text classification，那么如果是question answering或者textual entailment（文本蕴含，给定一个前提文本，根据这个前提去推断假说文本与前提文本的关系，一般分为蕴含关系和矛盾关系）这些任务，这些任务的输入是ordered sentence paired或者 document，question，answer三者的结合。pretrained模型是在文本连续的序列化文本，我们在应用到这种任务上的时候，需要对模型做一些调整。将输入转换为pretrained model可以处理的有顺序的序列，这样在更换任务的时候，就不需要对模型做更改了。

- textual entailment: 将premise p和hypothesis h拼起来，之间放一个$。
- similarity: 输入的两者并没有顺序关系，直接将两者分开处理，得到两个sequence representation，之后进行元素加，再送入最后的linear输出层。
- question answering and commonsense reasoning: 这种任务的输入是一个文件z,一个问题q,一个回答的集合$\{a_k\}$.我们可以将文件和问题与每一个可能的回答拼起来，中间加上分隔符$[z;q;\$;a_k]$,每一个这样的序列都会被单独处理，之后送入softmax。


**实验**

用BooksCorpus dataset训练language model，供7000本书，并且含有长句。
总的来说，在评估的12个数据集中有9个数据集中取得了更好的结果。并且在不同大小的数据集上都能很好的工作。transformer block的个数越多，语言模型越深，效果越好。


---

#### GPT2: Language Models are Unsupervised Multitask Learners


GPT2是GPT的升级版本,规模更大,参数更多,训练数据也更多. GPT是12层的transformer, BERT最深是24层的transformer, GPT2达到了48层,参数有15亿,训练数据是WebText.

在与训练阶段, GPT2采用了多任务的方式, 类似cs224中讲的decaNLP.

目前主流而的机器学习模型都是在指定的任务上用一部分数据来训练数据, 再用一部分不相同但分布相同的的数据来测试其性能.这样的模式在特定场景下效果不错, 但是对于captioning model, 阅读理解, 图像分类, 输入的多样性和不确定性会把缺点暴露出来.

文章认为,构建一个泛化能力更强的模型需要在更广泛的任务和领域上进行训练.

**数据**


学习单一任务可以用P(output|input)来表示, 对于多任务则用P(output|input, task)表示.

之前大部分训练语言模型的工作都是采用单一领域的文本, 如新闻,维基百科,小说. 为了能使模型在很多领域都可以胜任, 我们采用的是大且多样化的数据.

网络上爬了数据, 并且筛选了数据, 称这份数据集为WebText, 其中包含了4500万链接, 超过八百万个文档, 共40G. 移除了维基百科的文档, 因为WebText中会有一些重复的内容, 并且如果训练数据与测试数据重叠过多, 可能使分析变得复杂.


现有的大规模语言模型包括预处理步骤, 如转小写, 分词, 处理字典外token. 将unicode字符串转为utf-8字节序列可以很好地解决这一问题. 目前在大规模的数据集上, 字符级别的语言模型相比词级别的语言模型并没有更大的竞争力.

Byte Pair Encoding是一种字符级和单词级之间的实用语言模型, 它有效地在频率高符号序列的单词级别和频率低的符号序列的字符级别输入之间进行切换. BPE通常在Unicode上进行才做, 而不是字节序列. 该方法需要包含所有unicode编码, 以便能够对所有unicode字符串建模, 再添加任何多符号标记之前, 该方法的基本词汇表超过13万, 超过了BPE常用的3.2万到6.4万. 相比之下, 字节级别的BPE需要的字典大小只有256.

然而, 直接将BPE用于字节序列会产生次优解, 因为BPE使用贪婪算法来构建词汇表. 我们发现BPE中包含了很多像dog这样的常用的词, 因为它们出现在了很多变体中, 如*dog. dog! dog?*, 这就会导致词典词槽分配与模型能力受到限制. 为了避免这个问题, 我们会防止BPE跨字符类别合并任何字节序列. 


这种输入表征可以让我们结合单词级别的LM的实用性和字节级别的通用性结合起来. 这种方法可以给任意一个unicode字符串分配一个概率, 这就使得我们的LM不需要预处理, token化, 字典大小.


**模型**

LM仍然是基于Transformer, 和GPT中的大致相同有细微改动:
- layer normalization被移到了每个sub-block的输入, 类似一个预激活的残差网络.
- self-attention后面都添加了一个额外的layer normalization
- residual layer的权重初始化的时候, 乘以了$1/\sqrt{(N)}$, N是residual层数.
- 词典扩充到了50257.
- context size从512加到了1024
- batch size=512


**实验**

构建了四个模型

|Paremeters | Layers | $d_{model}$|
|------|:-----|:------|
| 117M | 12 | 768|
|345M|24|1024|
|762M|36|1280|
1542M|48|1600|

最小的模型规模和GPT是一样的, 第二小的模型规模和最大的BERT一样, 最大的模型这里叫GPT-2.所有的模型目前仍处于一个欠拟合的状态.



-------

#### BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding


和目前的language representation model不同的是,BERT预训练的是**双向表示**,在所有层中,既会拿左边context做condition,也会拿右边context做condition.之后,只需要一层额外的输出层,就可以创建state-of-the-art model,可以用于question answering, language inference,而不需要针对任务对模型做大量的修改.



**model architecture**

BERT模型基于多层双向Transformer encoder,定义Transformer block有L个, hidden size是H, self-attention头数为A. 设置feed-forward/filter size为4H.



我们这里采用两种大小的模型:
- BASE model: L=12, H=768, A=12, 总共参数量为110M
- LARGE model: L=24, H=1024, A=16, 总参数量为340M


**input representation**


![](/papers/tts/69.png)


**task #1: masked LM**

为了训练一个深层bidirectional representation,我们可以将输入随机进行mask,只预测这些被mask的token,这称为masked LM. 最终相应mask的hidden vector. 随机选择15%的token,然后:
- 80%概率,将这个token替换为[MASK]
- 10%概率,用随机的单词替换,如my dog is hairy -> my dog is apple
- 10%概率,不变




**task #1: Next Sentence Prediction**

很多NLP的任务,如QA和NLI都基于理解两个句子之间的关系的,这些没有直接被language modeling所捕捉. 为了训练一个理解两个句子之间关系的模型,我们预训练一个binarized next sentence prediction模型.进行训练的时候,50%的概率B是真正跟在A后面的,而50%的概率B是随机选择的.

最终训好的模型准确率有97%-98%



**finetune procedure**

对于sequence-level classification任务.将第一个token([CLS])对应的最后的hidden state取出,记为C. 最后加一层分类层W,添加的参数量为K*H(K是分类类别数). BERT所有的参数和W所有的参数一起finetune. 


finetuning中,模型大部分超参和pretraining中的仙童.dropout总是用0.1. 我们发现下面这些超参在左右任务上都表现很好
- batch size: 16, 32
- learning rate(Adam): 5e-5, 3e-5, 2e-5
- number of epoch: 3, 4


**BERT和GPT的比较**

- GPT在BooksCorpus(800M words)上训练; BERT在BooksCorpus(800M words)和Wikipedia(2500M words)

- GPT只有在fine-tuning的时候用sentence spearator([SEP])和classifier token([CLS]); BERT在pretrain的时候就学习了[SEP], [CLS]和sentence A/B embedding


- GPT用的batch size=32000, 训练了1M步; BERT用batch size=128000, step是1M步


- GPT在所有的fine-tuning实验中都用的learning rate是5e-5, BERT根据任务调节了learning rate



-------


#### MASS: Masked Sequence to Sequence Pre-training for Language Generation


虽然Pre-training 和 fine-tuning,如BERT在自然语言理解(NLU)上面取得了非常大的进展,但是seq2seq的自然语言生成中,目前主流预训练模型并没有取得显著效果.MASS对句子随机屏蔽一个长度为k的连续片段，然后通过编码器-注意力-解码器模型预测生成该片段。

![](https://image.jiqizhixin.com/uploads/editor/4a3fc479-3360-463b-a261-80cfb4a5da2e/640.png)



定义$(x, y) \in (X, Y), x=(x_1, x_2, ..., x_m), y=(y_1, y_2, ..., y_n)$

这里用的是无监督的方法, 给定单独的一个$x \in X, x^{\setminus u:v}$

----


#### Unified Language Model Pre-training for Natural Language Understanding and Generation


