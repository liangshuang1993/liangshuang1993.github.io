最近一直在改论文补实验,好久都没有在看新的论文了. 今天看一下无监督训练的一篇论文. 首先来看一下这里面用到的互信息mutual information的概念.(参考 http://www.omegaxyz.com/2018/08/02/mi/ )

两个离散随机变量X和Y的互信息可以定义为:

$$I(X; Y) = \sum_{y \in Y}\sum_{x \in X}p(x, y)log \frac{p(x, y)}{p(x)p(y)}$$

对于两个连续随机变量X和Y,那么可以定义为:

$$I(X; Y)=\int_Y\int_Xp(x, y)log\frac{p(x, y)}{p(x)p(y)}dxdy$$

互信息度量X和Y共享信息的程度,即度量这两个变量知道一个,对另一个不确定程度减少的程度. 假如X和Y相互独立,则知道X不对Y提供任何信息, 反之亦然, 则他们的互信息为0.

当X和Y相互独立的时候, 上面log里面的式子等于0. 


不同于相关系数,互信息并不局限于实值随机变量,它决定这联合分布P(X, Y)和边缘分布乘积P(X)P(Y)的相似程度.


$$I(X; Y)=H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X, Y)=H(X, Y) - H(X|Y) - H(Y|X)$$

![](http://www.omegaxyz.com/wp-content/uploads/2018/08/MI5.png)

![](/papers/unsupervised/1.png)

互信息的性质:

1. 对称性 

$$I(X; Y)=I(Y; X)$$

2. 非负性

$$I(X; Y)\geq0$$

3. 极值性

$$I(X; Y) \leq H(X)$$
$$I(Y; X) \leq H(Y)$$

-----

#### motivation

本文要做的事情是学习高维信息不同部分共同的information，舍弃local的low-level的信息和噪声。预测高维数据中一个很大的挑战是单峰的loss比如MSE或者交叉熵不是很有用，通常需要更厉害的conditional generative model来重现数据的每个细节。但是这些模型开销很大，总是倾向于对数据x中的复杂关系建模，往往忽略了context c。当预测未来信息时，我们将目标x(future)和context c(present)编码到一个compact distributed vector representation，使得x和c之间的mutual information最大。

$$I(x;c) = \sum_{x,c}p(x,c)log~\frac{p(x|c)}{p(x)} \tag{1}$$

通过最大化encoded representation之间的互信息，我们可以提取到输入隐含的共同的latent variable。

----

#### 模型结构

![](/papers/unsupervised/2.png)



首先有一个非线性的编码器$g_{enc}$将输入序列编码为一个隐变量的序列. 接下来一个自回归模型$g_{ar}$汇总t时刻之前的z,产生一个context latent representation $c_t=g_{ar}(z_{\leq t})$

这里我们没有直接用一个生成模型$p_k(x_{t+k}|c_t)$来预测$x_{t+k}$, 而是对一个density ratio建模, 这个density ratio保留了$x_{t+k}$和$c_t$之间的互信息.


$$f_k(x_{t+k}, c_t) \propto \frac{p(x_{t+k}|c_t)}{p(x_{t+k})}  \tag{2}$$

我们这里使用一个log-bilinear模型:

$$f_k(x_{t+k}, c_t)=exp(z^T_{t+k}W_kc_t) \tag{3}$$

每个时间步k，用一个不同的$W_k$, 或者可以采用一个非线性网络或者RNN网络。

通过用density ratio $f(x_{t+k}, c_t)$, 并且利用了一个encoder来推断$z_{t+k}$, 我们的模型不需要对高维度分布$x_t$进行建模. 虽然我们不能直接评估$p(x), p(x|c)$, 我们可以用采样进行评估.可以利用noise-contrastive estimation和importance sampling.

----

训练encoder和autoregressive model来优化一个基于NCE的loss 。$X={x_1, ..., x_N}$中一个是从$p(x_{t+k}|c_t)$采样到的正样本，N-1个从$p(x_{t+k})$采样到的负样本，我们可以优化：

$$L_N = -E_X [log \frac{f_k(x_{t+k}, c_t)}{\sum_{x_j \in X} f_k(x_j, c_t)}]  \tag{4}$$

优化这个loss可以使得$f_k(x_{t+k}, c_t)$估计式子2中的ratio，下面是推导过程：

我们用$p(d=i|X, c)$表示$x_i$是正样本。而$x_i$ 是正样本，即从$p(x_{t+k}|c_t)$中采样，而不是从$p(x_{t+k})$中采样的概率可以表示为：

$$
\begin{aligned}
p(d=i|X, c) & = \frac{p(x_i|c_t)\prod_{l\neq i}p(x_l)}{\sum_{j=1}^N p(x_j|c_t)\prod_{l\neq j}p(x_l)} \\
& = \frac{\frac{p(x_i|c_t)}{x_i}}{\sum_{j=1}^N \frac{p(x_j|c_t)}{x_j}}
\end{aligned} \tag{5}
$$

公式4的其实是将正样本正确分类的交叉熵。因此$f_k(x_{t+k})$的最优值是proportional to $\frac{p(x_{t+k}|c_t)}{x_{t+k}}$的。

附录中也证明了$I(x_{t+k}) \geq log(N) - L_N^{opt}$

所以最小化$L$等同于最大化mutual information的一个下届。



---
### 附 NCE

NCE的主要目标是利用非线性logistic regression来判别observed data和人工产生的噪声。本文针对的是unnormalized statistical model，即密度函数不需要积分为1.

首先定义问题： 样本$x \in R^n$ 服从概率密度函数$p_d(.)$，同时有一系列模型来对这些数据进行建模${p_m(.;\alpha)}_{\alpha}$, $\alpha$是参数。我们假设$p_d(.)$属于这个系列，即$p_d(.) = p_m(.;\alpha^*)$. $\alpha^*$是选用的参数。此时问题转换为如何通过优化某个目标函数，从观测样本中预测参数$\alpha$.

这个estimation的任何解$\hat \alpha$必须符合归一化要求：

$$\int p_m(u;\hat \alpha)du = 1$$.这就给优化函数添加了一个约束。可以通过以下的定义来满足这个约束：

$$p_m(.;\alpha) = \frac{p^0_m(.;\alpha)}{Z(\alpha)}; Z(\alpha) = \int p^0_m(u;\alpha)du$$

这时候，$p^0_m(.;\alpha)$不需要积分为1，但是这里计算$Z(\alpha)$非常麻烦：它的积分几乎是intractable的。一种简单的处理归一化限制的方式是将$Z(\alpha)$也当做一个额外的参数，但这总方法对MLE不适用。这是由于通过将$Z(\alpha)$变得趋近于0可以使似然值无穷大。因此又有其他方法来预测模型。本文则是提出一种新的方法。

#### 模型定义

定义$ln~p_m(.;\theta) = ln~p^0_m(.;\alpha) + c$,其中c是对$Z(\alpha)$的估计，只有在某些特定的c时，$p_m(.;\theta)$才会积分和为1.

定义$X=(x_1, x_2,...x_T)$是观测的数据集，$Y=(y_1,...,y_T)$是任意产生的噪声数据，这些噪声数据满足分布$p_n(.)$，estimator $\hat \theta_T$可以通过最大化以下目标函数来得到：

$$J_T(\theta) = \frac{1}{2T} \sum_t{ln[h(x_t;\theta)] + ln[1 - h(y_t; \theta)]}$$, 其中

$$h(u;\theta) = \frac{1}{1 + exp[-G(u;\theta)]}$$
$$G(u;\theta) = lnp_m(u;\theta) - lnp_n(u)$$

*推导如下*

定义$U=(u_1,...,u_{2T})$是X和Y的并集，给U中每个元素一个标签$C_t: 如果u_t \in X, C_t = 1;如果u_t \in Y, C_t = 0 $.利用$p_m(.;\theta)$来对$p(.|C=1)$建模，可以得到下式：

$$p(u|C=1;\theta) = p_m(u;\theta); ~p(u|C=0) = p_n(u)$$

两个类别的概率相等，因此可以得到$P(C=1) = P(C = 0) = 1 / 2 $, 因此，

$$P(C=1|u;\theta) = \frac{p_m(u;\theta)}{p_m(u;\theta) + p_n(u)} = h(u;\theta) $$
$$P(C=0|u;\theta) = 1 - h(u;\theta) $$

标签$C_t$是伯努利分布，因此$\theta$的log-likelihood是：

$$
\begin{aligned}
l(\theta) &= \sum_tC_tln~P(C_t = 1|u_t;\theta) + (1-C_t)ln~P(C_t=0|u_t;\theta) \\
&=\sum_t ln[h(xt;\theta) + ln[1-h(y_t;\theta)]]
\end{aligned}
$$


week law of large numbers可知T很大的时候$J_T(\theta)$概率收敛到$J$

$$J(\theta) = \frac{1}{2}E ~ ln[h(x;\theta)] + ln[1 - h(y;\theta)]$$

我们将J视作$f(.)=ln~p_m(.;\theta)$的函数的形式,r是sigmoid函数。

$$\tilde J(f) = \frac{1}{2}E~ln[r(f(x) - lnp_n(x))] + ln[1 - r(f(y) - lnp_n(y))]$$


**定理1** $\tilde J$在$f(.)=ln~p_d(.)$的时候获取最大值。如果当p_d(.)是非0时，p_n(.)也是非0，那么没有其他极值。

（选择高斯分布作为noise的分布可以满足大于0的条件）

**定理2** 如果条件a到c都满足了，那么$\hat  \theta_T$概率收敛到$\theta^*, 即\hat \theta_T \rightarrow \theta^*$

a. 在$p_d(.)$非零的地方，$p_n(.)$是非零的

b. $sup_{\theta}|J_T(\theta) - J(\theta)| \rightarrow 0$

c. $I=\int g(x)g(x)^TP(x)p_d(x)dx$有full rank, $P(x)=\frac{p_n(x)}{p_d(x) + p_n(x)}, g(x) = \triangledown_{\theta}lnp_m(x;\theta)|_{\theta}$

**定理3**  $\sqrt T(\hat \theta_T - \theta ^*)$具有渐进正态性，均值是0，方差是$\Sigma$.

$$\Sigma = I^{-1} - 2 I ^{-1}[\int g(x)P(x)p_d(x)dx] \times [\int g(x)^TP(x)p_d(x)dx]I^{-1}$$

**对比噪声分布的选择**

我们希望这个分布满足以下几个特点：

- 方便采样
- log pdf 有解析表达式，这样我们可以直接计算J
- 使$E||\hat \theta_T - \theta^*||^2$均方差小

通常选择高斯，均匀，高斯混合，ICA分布。

直觉上，noise distribution应该和真实数据分布尽可能靠近，否则太容易分辨了。所以，可以首先估计一个preliminary model，然后用它来做噪声分布。

