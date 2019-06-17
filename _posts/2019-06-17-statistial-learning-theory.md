学习statistical learning theory课程, https://www.stat.berkeley.edu/~bartlett/courses/2014spring-cs281bstat241b/


### Lecture1

给定n对数据$(x_1, y_1), ..., (x_n, y_n)$, 选择一个方程$f: \mathcal{X} \to \mathcal{Y}$, 使得对于后来的$(x, y), f(x)$是一个对y好的预测.


为了定义什么叫做好的预测, 我们可以定义一个**loss function**:
$$\mathcal{l}: \mathcal{Y} \times \mathcal{Y} \to R$$

$ \mathcal{l}(\hat y, y)$可以量化当真值结果是y时,预测出为$\hat y$的代价, 之后目标在于保证$l(f(x), y)$尽可能小.

**概率假设**
假设有一个在$\mathcal{X}\times \mathcal{Y}$上的概率分布P. 根据P可以独立选出$(X_1, Y_1), ..., (X_n, Y_n)$, 目标在于用小的risk选择出f, risk定义为:

$$R(f)=El(f(X), Y)$$

note:
- 大写字母表示随机变量
- P可以视为不同的特征或者协方差的建模, 以及给定X输出Y的条件概率
- 假设数据独立是一个强假设, 但是我们需要假设一些关于$(x_1, y_1), ..., (x_n, y_n)$给出(X, Y)相关信息.
- 方程f是随机, 因为它依赖于随机数据, 因此,risk:

$$R(f_n)=E[l(f_n(X), Y)|D_n]=E[l(f_n(X;X_1, Y_1, ..., X_n, Y_n), Y)|D_n]$$

 也是随机变量. 我们希望$ER(f_n)$小, 或者$R(f_n)$以很高的概率小.

 我们可以从某类F函数中选择出$f_n$, 比如从linear function, sparse linear function, decision tree, neural network, kernel machine中选择.


 我们关注以下一个问题:
 
 - 我们是否能设计出$f_n$接近我们从F中希望选择出来的最好的, 即$R(f_n) - inf_{f \in F}R(f)$小.

 - $f_n$的performance怎么依赖于n? 依赖F的复杂性? 依赖P?

 - 我们是否可以保证$R(f_n)$接近最好可能的表现, 即R(f)在所有f上的下限?


 **statistical learning theory vs classical statistics**

和parametric problem不同的是,我们不会(经常)假设P是来自于一个小空间(有限维度) 由于我们对P做很少的假设,我们关心的是高维的数据, 目标是确保性能接近基于一些固定类别的F我们预测得到的最好.

这其中有以下问题:

- **Approximation** F中最好的f有多好? 即$inf_{f \in F}R(f)与$$inf_fR(f)$有多近?

- **Estimation** 我们的performance与F中最好的f有多近? (我们只能通过观测到的一个有限的数据集来利用P)

- **Computation** 我们需要用数据来选择$f_n$, 通常是利用一些最优化的方法, 我们如何有效地实现?


> 我们重点关注estimation, 将computation的效率作为约束.

**More General Probabilistic Formulation**

我们考虑一个决策论公式, 已知:

1. 结果空间 $\mathcal{Z}$
2. prediction strategy $S: Z^* \to \mathcal{A}$
3. loss function $l: \mathcal{A} \times \mathcal{Z} \to R$

拟定:

- 在 $\mathcal{Z}$上,由位置的概率P,独立同分布看到结果$Z_1, ..., Z_n$
- 选择action $a=S(Z_1, ..., Z_n) \in A$
- 引发risk $El(a, Z)$

目标最小化excess risk和best decision的差距:

$$E[l(S(Z1, ..., Z_n), Z)|Z^n_1] -inf_{a \in A}El(a, Z)$$

例子1: pattern classification问题:

- $$\mathcal{Z}=\mathcal{X} \times \mathcal{Y}, $$
- $$\mathcal{A} \subset \mathcal{Y}^{\mathcal{X}}$$
- $$l(f, (x, y))=1[f(x) \neq y]$$

例子2: density estimation问题:
- $$\mathcal{Z} = R^d(或者是一些可测量的空间)$$
- $$\mathcal{A}是\mathcal{Z}上measurable function$$
- $$l(p, y) = -logp(z)$$

**Game Theoretic Formulation**

decision method: $a_t \in \mathcal{A}$
world reveals $z_t \in \mathcal{Z}$
incur loss: $l(a_t, z_t)$

累计loss:

$$\hat L_n=\sum^n_{t=1}l(a_t, z_t)$$

希望最小化regret: 

![](/courses/stat/1.png)



----


### Lecture2

**pattern classification**

定义:
 
$$\eta(x)=P(Y=1|X=x)$$

则化简得到:

$$R(f)=El(f(X), Y)=E(1[f(X)\neq1](2\eta(X)-1)+1-\eta(X))$$

定理: 对任意的 $f: \mathcal{X} \to \mathcal{Y}, $

$$R(f)-R(f^*)=E(1[f(X) \neq f^*(X)]|2\eta(X)-1|)$$

**plug-in methods**

我们用$\hat \eta$来估计$\eta$, 那么我们应该用什么准则呢?

$L_1(\mu)$满足:

$$R(f_{\hat \eta}) - R^* \leq 2E|\eta(X) - \hat \eta(X)|$$

需要注意的是估计$\eta$并不是正确分类所必须的. 这个bound对于plug-in分类器非常宽松. 比如, 如果$\eta(X) \in \{0, 1\}$, 那么对于所有的$\epsilon > 0, $存在$\hat \eta$满足:

- $\hat \eta和\eta$总是在1/2的同一侧.
- $|\hat\eta(X) - \eta(X)| = \frac{1-\epsilon}{2}$

因此


$$R(f_{\hat\eta}) - R^* = 0 << 1 - \epsilon = 2E|\eta(X) - \hat \eta(X)|$$

**Linear threshold functions**

- Approximation 可以考虑一个大得多的类别, 保留线性方程好的特性, 
- Estimation 较小的d/n是可以的, 大的话利用正则化也是ok的.
- computation 如果 $\hat R(f)=0$那么很简单, 如果不是的话难. 可以通过考虑其他的(凸的)loss function来简化


**perception algorithm**

对于不符合的$(x_i. y_i)$, 即$y_i \neq sign(\theta^T_t, x_i)$
$\theta_{t+1} := \theta_t + y_ix_i$
$t: =t+1$

定理: 给定线性可分的数据, 经过不超过$\frac{R^2}{\gamma^2}$次的更新后终止, 其中:

$$R=max_i||x_i||$$
$$\gamma=min_i\frac{\theta^Tx_iy_i}{||\theta||}$$


### Lecture3

**kernel methods**

$$\theta_t = \sum_i\alpha_ix_i, ||\alpha||_1 = \sum_i|\alpha_i|=t$$ 




可以将感知机中的内积$<x, \theta>=x^T\theta$用任意的内积替代:

- predict: $\hat y_i=sign(\sum_j\alpha_j<x_j, x_i>)$

- update: 如果 $\hat y_i \neq y_i, \alpha_i^{t+1} := \alpha^t_i + y_i$


**Minimax risk**

从class $\mathcal{P}$中随机选择一个$P, R(f_n)$在这个选择下的期望大. 这表明在这个类的某些分布中, $R(f_n)$大. 

我们希望$\mathcal{P}$中的分布满足:

- 对于某些$f \in F, R(f) = 0$

- 一个大小为n的sample包含f的有限信息.

定理: 对于任意的 $n \geq 1, 任意的mapping f_n: R^d \times (R^d \times \{\pm 1\})^n \to \{\pm 1\}$, 存在分布 P , 会有一些linear threshold function $f\in F, 满足 R(f) = 0$, 但是

$$ER(f_n) \geq \frac{min(n, d) - 1}{2n}(1 - \frac{1}{n})^n$$

