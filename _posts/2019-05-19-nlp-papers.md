最近打算读几篇比较有名的NLP论文。

https://blog.csdn.net/manmanxiaowugun/article/details/83794454



#### GPT: Improving Language Unstanding by Generative Pre-Training

很多NLP模型都是在有标签的数据集上进行训练的，那么如何有效地利用无标签数据集呢？这篇论文提出了一种半监督的学习方法：首先用无监督的方式进行pretrain，之后再在特定task上进行finetune。

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



-------

#### BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
