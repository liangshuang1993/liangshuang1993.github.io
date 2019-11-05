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


这篇文章中的互信息稍微改了下形式, 它求的是原始信号x和context c之间的互信息:

$$I(x; c)=\sum_{x, c}p(x, c)log\frac {p(x|c)}{p(x)}$$

通过最大化encoded representation之间的互信息, 我们提取输入共有的隐变量.

-----

#### 模型结构

![](/papers/unsupervised/2.png)



首先有一个非线性的编码器$g_{enc}$将输入序列编码为一个隐变量的序列. 接下来一个自回归模型$g_{ar}$汇总t时刻之前的z,产生一个context latent representation $c_t=g_{ar}(z_{\leq t})$

这里我们没有直接用$p_k(x_{t+k}|c_t)$来预测$x_{t+k}$, 而是对一个density ratio建模, 这个density ratio保留了$x_{t+k}$和$c_t$之间的互信息.


$$f_k(x_{t+k}, c_t) \propto \frac{p(x_{t+k}|c_t)}{p(x_{t+k})}$$

我们这里使用一个log-bilinear模型:

$$f_k(x_{t+k}, c_t)=exp(z^T_{t+k}W_kc_t)$$

通过用density ratio $f(x_{t+k}, c_t)$, 并且利用了一个encoder来推断$z_{t+k}$, 我们的模型不需要对高维度分布$x_t$进行建模. 虽然我们不能直接评估$p(x), p(x|c)$, 我们可以用采样进行评估.可以利用noise-contrastive estimation和importance sampling.


---
附 NCE和importance sampling



---