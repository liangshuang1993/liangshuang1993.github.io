受到FastSpeech影响，看几篇相关的non-autoregressive文章. 非自回归的网络具有很高的inference速度, 因此最近比较流行.


#### Non-Autoregressive Neural Machine Translation

ICLR2018的一篇文章

移除autoregressive的一个简单的方法:

假设target sequence length T可以用$p_L$建模,

$$p_{NA}(Y|X;\theta)=p_L(T|x_{1:T'};\theta) \cdot \prod_{t=1}^Tp(y_t|x_{1:T'};\theta) $$

上面这个式子还有一个明确的likelihood函数,它还可以在每个输出分布上用独立的交叉熵损失函数.但是这些分布可以在inference的时候并行计算.

不过,这种简单的方法不能产生好的结果,这是因为这个模型表现出完全的*条件独立性*. 每个输出$p(y_t)$只依赖source sentence X, 这离真实的target distribution比较远. 

比如英文"Thank you"可以直接翻译到德语"Danke", "Danke schön", "Vielen Dank". 如果全都是独立的输出,那么如果可以输出"Danke schön", "Vielen Dank", 也可能会输出"Danke Dank"和"Vielen schön". 条件独立假设使得模型不能捕捉target translation中的多模态分布. 我们把这个问题称为multimodality problem.


模型结构如下:

![](/papers/nlps/1.png)

模型主要包括几个模块:encoder, decoder,新添加的fertility predictor和translation predictor用于token decoding.

**encoder**

和自回归transformer相似,encoder和decoder都是用MLP和多头attention组成的. encoder和transformer中的结构完全相同.

**decoder**

做了如下改动:

- dropout inputs: 在decoding开始之前,NAT需要知道target sentence的长度, 这样才可以并行生成所有的单词. 我们在training的时候不能使用time-shifted target outputs, 在inference时候不能使用前面预测的输出. 直接在第一层解码器忽略输入,或者只用positional embedding,会产生比较差的结果. 这里我们将encoder side的source input拷贝到decoding process. 由于source和target通常是不一样长度的,因此我们这里采用了两种做法:
    - copy source input uniformly: 每个decoder输入t是(T't/T)段的encoder input. 相当于在source input处从左到用匀速扫描.
    - copy source input using fertilities: 如上图所示, 用"fertility"决定拷贝次数,0次或者多次.

- non-causal self-attention: 没有了autoregressive的限制,我们可以接收当前时间步后面的信息, 因此我们可以去掉自注意力里面的因果mask. 我们只用了mask防止query注意到自己, 我们发现这种方法相比完全不mask的decoder,可以提高decoder性能.
- positional attention: 在每个decoder层添加了额外的positional attention module, 用的是多头注意力模型, query和key是positional encoding, decoder state是value.

**model fertility**

引入了一个中间变量z, 从一个先验分布中采样得到z, 之后基于z非自回归地进行翻译. 

之前提出的公式提供了一个因变量模型非常弱的例子,我们这里利用*fertilities*. 

$$p_{NA}(Y|X; \theta)=\sum_{f_1, ..., f_{t'} \in F}(\prod^{T'}_{t'=1}p_F(f_{t'}|x_{1:T'}; \theta)\cdot\sum^T_{t=1}p(y_t|x_1\{f_1\}, ..., x_{T'}; \theta))$$

其中$F=\{f_1, ..., f_{T'}|\sum^{T'}_{t'=1}f_{t'}=T, f_{t'} \in Z^*\}$是所有fertility序列的集合.

**translation predictor and the decoding process**

给定fertilities序列,最佳翻译仅需要独立地最大化每个输出位置的局部概率$Y=G(x_{1:T'}, f_{1:T'};\theta)$. 但是在整个fertility空间进行孙锁和边缘化仍然是难以处理的. 我们提供了三种启发式的decoding方法.

- argmax decoding: 



#### Non-Autoregressive Machine Translation with Auxiliary Regularization

AAAI 2019的一篇文章

非自回归的机器翻译(NAT)inference效率很高, 但是呢, 高的效率是以没有捕捉target side的序列依赖为代价的.会造成NAT遇到两种问题:
- 翻译有重复(由于相邻解码器隐状态难以区分)
- 翻译不完整(source side的信息经过解码器隐状态转移不完全)

这篇论文在训练NAT模型的时候加入两个额外的正则项来提高解码器hidden representation, 从而解决上述问题:
- 为了使得隐状态更好区分, 基于相应的target token在连贯的隐状态上加上相似度正则约束
- 为了使隐状态包含source序列所有的信息, 我们利用了翻译任务的对偶性, 并且最小化backward reconstruction error来保证解码器的隐状态可以恢复source side sentence. 

**introduction**

传统的解码器通常采用的是自回归的方法, 求解的是条件概率$P(y_t|x, y\lt t)$, 其中$y \lt t$指的是在$y_t$之前生成的token. 所以呢不管是机器翻译还是TTS,通常都只能一个时间步一个时间步地生成数据, 这就会影响了inference时候的效率.


18年也有几篇non-autoregressive translation(NAT)模型. 一个基本的NAT模型包和自回归翻译模型一样也是encoder-decoder结构, 除了忽略了target size sentence的序列依赖性. 用这种方式, 所有的token都可以并行生成. 但是这样会影响翻译质量, 因为它忽略掉了句子内部的依赖性. 为了解决这个问题, 一些工作试着在基本的NAT模型中插入了中间离散变量, 以便结合少量的序列信息到非自回归解码器中. 但是引入这些变量不仅带来了优化上的困难, 还使得翻译变慢了.


这篇论文不依赖离散变量,只修改了基本NAT模型的一小部分. 如上文指出, 目前NAT模型主要存在翻译重复和翻译不完全这两个问题. 这些都表明NAT模型中的解码器隐状态(decoder最上面层的隐状态输出)质量较低: 重复翻译说明两个相邻的隐状态难以区分, 导致了相同的输出; 翻译不完全则说明解码器隐状态没有将source side infomation表示完全.

这种问题其实也是non-autoregressive的本质所造成的: 每个时间步, hidden state不能获取他们前面的解码状态. 使得他们不清楚哪些被翻译了哪些没有.因此,我们不能只用纯粹的non-autoregressive模型. 

自回归模型在语音合成上面有一些改进: parallel wavenet, knowledge distillation. 在机器翻译领域为了保证翻译质量, 有一些技术:
- sequence level knowledge distillation. NAT模型用一个自回归机器翻译模型做老师. 通过sequence级别的知识蒸馏技术,将自回归模型的知识蒸馏到NAT模型. 实验表明用这种方式比用ground data效果更好. 目前没有理论证明, 但是一种直观的解释是NAT模型有着"多模态"的问题, 即一句话可以有多个不同的翻译, 通过teacher model蒸馏到的数据是一个网络输出的, 因此less noisy, more deterministic. 
- 模型结构修改. NAT模型基本和AT模型相同, 不过有下面修改: 1) causal mask of decoder被移除了. 2) 利用positional attention来加强位置信息, positional embedding(transformer)用作query和key, 前一层的hidden representation作为value.
- 离散变量. representative example表示source side token拷贝的数量(和FastSpeech相同?待考证). 离散隐变量自回归产生. 离散型增加了额外的难度, 需要一些其他的优化手段, 如变分, vector quantization

**model**

模型基本与AT一样, 用了两种手段来训练:
- sequence leve知识蒸馏
- 还有模型结构上的一些改变, 如使用positional attention, 去掉decoder中的causal mask

我们方法独特的地方在于去掉了难以优化的离散变量, 用了两个简单的正则项来替代. 由于没有离散项来表明序列信息, 我们这里需要一个新的机制来**预测inference时候的target length, 为decoder生成输入**, 这一点非常重要, 因为送入decoder的信息(比如自回归模型中上一个target side token的word embedding)是未知的.


在inference时候, 我们指定target side length $T_y=T_x+\Delta T, T_x$是source sentence的长度, $\Delta T$是常数bias.


给定source sentence $x = \{x_1, x_2, ..., x_{T_x}\}$, 和target length $T_y$, 我们将source token embedding(用$\epsilon$表示)进行映射, 以产生decoder input:

$z_k=\epsilon(x_i), i=[\frac{T_x}{T_y}t], t=1, 2, ..., T_y$


下面介绍下如何解决上述两个问题

1. repeated translation & similarity regularization

