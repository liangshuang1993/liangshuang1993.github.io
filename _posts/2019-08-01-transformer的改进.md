transformer提出后引起了很大的反响,但是transformer自身也存在一些问题,比如参数量过大.这里主要介绍几篇对transformer改进的文章.


#### A Tensorized Transformer for Language Modeling

这篇文章提出transformer中的multi-head attention参数过大, 因此这篇文章采用了张量分解和权重分享的方式,利用*Block-Term张量分解* 提出了一种新的self-attention layer(Multi-linear attention), 减少参数量的同时还提高了模型的效果.


这篇文章用到了很多分解,这里先介绍一下Tensor的基本操作和分解. 参考 http://www.xiongfuli.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/2016-06/tensor-decomposition-part1.html 

**Tensor**

vector和matrix可以称为1-order tensor和2-order tensor. n-order的tensor中的元素可以定义为$\mathcal A_{d_1,...,d_n}$

**Outer Product**

张量$$X\in R^{I_1 \times I_2 \times ... \times I_N}, Y\in R^{J_1 \times J_2 \times ... \times J_M}$$,

$$Z = X \circ Y \in R ^{I_1 \times I_2 \times ... \times I_N \times J_1 \times J_2 \times ... \times J_M}$$

$$z_{i_1, i_2,...,i_N, j_1, j_2, ...,j_M}=x_{i_1, i_2, ...,i_N} * y_{j_1,j_2,...,j_M}$$

*如果X和Y分别是一个向量,那么他们的外积是一个秩为1的矩阵*

**Kronecker Product**

两个矩阵$$A \in R^{I \times J}, B \in R^{K \times L}$$

![](http://www.xiongfuli.com/assets/img/201606/kron.png)


**Hadamard Product**

两个相同大小的矩阵$$A \in R^{I \times J}, B \in R^{I \times J}$$

![](http://www.xiongfuli.com/assets/img/201606/hadamard.png)


**Khatri-Rao Product**

两个相同列数的矩阵$A \in R^{I \times K}, B \in R^{J \times K}$

$$A \odot B = [\begin{matrix} a_1 \otimes b_1 &{a_2 \otimes b_2}  & ... & a_K \otimes b_K \end{matrix}] \in R^{IJ \times K}$$

![](http://www.xiongfuli.com/assets/img/201606/kr.png)


**张量与矩阵的模积(Mode-n Product)**

张量与矩阵的模积定义了一个张量$X \in R^{I_1 \times I_2 \times ... \times I_N}$和一个矩阵$U \in R^{J \times I_n}$的n-mode乘积$(X \times_n U) \in R^{I_1 \times ... \times I_{n-1} \times J \times I_{n+1} \times ... \times I_N}$, 其元素定义为

$$(X \times_n U)_{i_1...i_{n-1}ji_{n+1}...i_N} = \sum^{I_n}_{i_n=1}x_{i_1i_2...i_N}u_{ji_n}$$

可以写成$Y=X\times_n U, Y_n=UX_n$


**CP Decoposition**

CP分解将一个N阶的张量$\mathcal X \in R^{I_1 \times I_2 \times ... \times I_N}$分解为R个秩为1的张量和的形式:


$$\mathcal X = \sum^R_{r=1}\lambda_ra^{1}_r \circ a^{2}_r\circ ... \circ a^{N}_r$$

通常情况下$a_r^{n}$是一个单位向量.定义$A^n= [\begin{matrix} a_1^n  & a_2^n &... & a_r^N \end{matrix}] , D=diag(\lambda)$, 上面的公式可以写为:

$$\mathcal X = D \times_1 A^1 \times_2 A^2 ... \times_N A^N$$

矩阵的表达形式为:

$$X_n = A^nD(A^N\bigodot...A^{n+1}\bigodot A^{n-1}...\bigodot A^1)^T$$


当张量的阶数为3时,它的分解如下图所示:

![](http://www.xiongfuli.com/assets/img/201606/cp.png)

两个向量的外积可以得到一个矩阵,三个向量的外积可以得到一个三阶张量,秩为1.


矩阵的秩是矩阵中的最大的不相关的向量的个数.如果一个张量能够以两个秩一张量的和表示，那么其秩则为2。如果一个张量能够以三个秩一张量的和表示，那么其秩为3...

求解CP分解一般用的算法是交替最小二乘法(ALS)算法(Alternating Least Squares algorithm).

**Tucker Decomposition**

![](http://www.xiongfuli.com/assets/img/201606/tucker.png)

对于一个三阶张量$\mathcal X \in R^{I \times J \times K}$, 由Tucker分解可以得到$A \in R^{I \times P}, B \in R^{J \times Q}, C \in R^{K \times R}$三个因子矩阵和一个核张量$\mathcal G \in R^{P \times Q \times R}$, 每个mode上的因子矩阵称为张量在每个mode上的基矩阵或者是主成分,因此**Tucker分解又称为高阶PCA,高阶SVD等**. CP分解是Tucker分解的一种特殊形式.


**Block-Term Tensor Decomposition BTD**

利用CP分解可以将一个张量分解为K个rank为1的成员的张量的形式, 但是某些情况下,我们希望得到不同尺度的特征,即不同秩的特征.

$$\Tau=\sum^K_{r=1}D_r \times_1 A_r \times_2 B_r \times_3 C_r$$, 其中$D_r$是一个$Rank-(L_r, M_r, N_r)$的张量, $A_r \in R^{I \times L}$, BTD分解可以看做是Tucker分解和分解相结合的形式．

如下图所示，一个三阶的张量可以做如下的BTD分解

![](/papers/nlps/2.png)


**single-block attention by Tucker Decomposition**

**定理１**: $e_1, ..., e_n$为向量空间Ｓ中的基本向量。假设这些向量是线性独立的。原有的attention:

$$Attention(Q, K, V)=softmax(\frac{QK^T}{\sqrt d})V$$ 可以用这些基向量的线性组合表示：

$$Attention(Q, K, V)=(e_1, ..., e_n)M$$

![](/papers/nlps/3.png)

single-block attention可以写成如下形式：

$$Atten_{TD}(\mathcal G;Q, K, V)=\mathcal G \times_1 Q \times_2 K \times_3 V = \sum^I_{i=1}\sum^J_{j=1}\sum^M_{m=1}\mathcal G_{ijm}Q_i \circ K_j \circ V_m$$

$\mathcal G$是一个core tensor

$$\mathcal G =
\begin{cases}
rand(0, 1)& \text{i=j=m}\\
0& \text{otherwise}
\end{cases}$$




-----------

Self-Attention with Relative Position Representions

谷歌出的文章，提出了相对位置编码。

传统的transformer是引入了额外的position embedding，加入了绝对位置信息，这篇文章则是考虑了不同element之间的相对位置信息。

引入两个元素之间的相对信息,考虑到两个元素之间距离过长，他们的影响可以忽略，因此引入了一个k代表最大长度。

![](/papers/nlps/5.png)



![](/papers/nlps/4.png)

之后，可以将这个相对位置编码加入到原有的attention公式中：

![](/papers/nlps/6.png)

为了便于计算，将上式展开即可。

整篇论文比较简单，将原有transformer中的绝对位置编码改成了相对位置编码，实验表明，在没有绝对位置编码的时候，仅仅利用相对位置编码，可以取得一定的提升。

------

####　TransformerXL

普通的transformer一定程度上解决了rnn长依赖的问题，但是仍然还局限在了固定长度的context。TransformerXL则适用于更长文本的模型。

TransformerXL的主要思想是两点：

- 在self-attention上引入recurrence, 其实就是每个segment利用了上一个segment的信息。

- 引入相对位置编码

**segment-level recurrence with state reuse**

对于长度很长的序列，一种自然的方法是将它分成多个长度较短的segments,　以前也有论文这么做过，但是并没有将信息进行segment之间的传递。

![](https://pic1.zhimg.com/v2-732805e00feb35e41f1d00f8df516950_b.gif)

很明显，这样会隔断整个sequence,　而TransformerXL则是会将上个segment的隐状态输入到下一个segment中，如下图所示：

![](https://pic1.zhimg.com/v2-a8210cd2f9bfb9307ba81d694dc4e4b4_b.gif)

上一个segment的第n层隐状态与这一层的第n层的隐状态concat起来，然后再进行self-attention的计算。

![](/papers/nlps/7.png)

**relative positional encodings**

原始的transformer中，用到了一个positional encodings, 这个position embedding如果直接接入不同的segment，那么不同的segment中的同一位置的positional embedding是相同的。

这篇论文认为positional encoding给了模型一个关于如何收集信息的一个temporal clue或者是"bias"，即去注意哪个位置。因此，这里的positional embedding并没有直接跟initial embedding放在一起，而是送入了每一层的attention score。

这里并没有直接用上一篇论文的方法，而是用了新的方法来取得更好的泛化性能。

原始的transformer中，attention score为：

$$A^{abs}_{i, j}=q_i^Tk_j = E^T_{x_i}W_q^TW_kE_{x_j} + E^T_{x_i}W_q^TW_kU_j + U_iW_q^TW_kE_{x_j} +  U_iW_q^TW_k*U_j$$

引入了相对位置编码，我们可以将上式改写为：

$$A^{rel}_{i, j}=q_i^Tk_j = E^T_{x_i}W_q^TW_{k,E}E_{x_j} + E^T_{x_i}W_q^TW_{k, R}R_{i-j} + u^TW_{k,E}E_{x_j} +  v^TW_{k,R}R_{i-j}$$

这里面做了三个地方的改变：

- 将原来的$U_j$改为了相对位置编码$R_{i-j}$, R是sinusoid encoding matrix， 没有可训练的参数。

- 第三项中引入了可以训练的参数$u \in R^d$替换掉了$U_i^TW_q^T$，第四项中 $v \in R_d$替换掉了$U_i^TW_q^T$。

- 将$W_k$分成了$W_[k, E]$和$W_{k, R}$来分别产生content-based key vector和location-based key vector。

整体计算流程如下所示，需要注意的是其实recurrence并不是严格的segment到segment，模型会有一个长度为M的cache，存着之前的hidden state，因此下面式子中的m其实指的就是cache中的数据。query的长度是segment的长度L，但是key的长度是(L+M)

![](/papers/nlps/8.png)


附录中给出了一些模型中矩阵计算的简化思想，比较重要，具体看https://arxiv.org/pdf/1901.02860.pdf 。

论文的思想很简单，具体实现还是优点复杂的，可以看代码 https://github.com/kimiyoung/transformer-xl

代码中没有encoder，只有decoder。

整体而言，Transformer可以解决长文本依赖问题,评估时候速度更快。



------