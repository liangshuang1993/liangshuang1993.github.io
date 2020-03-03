### Mutual Information Neural Estimation(MINE)

这篇文章主要将MINE及相关文章。之前文章中我们讲过互信息的定义$I(X;Z) = \int_{x,z}p(x, z)log \frac{p(x, z)}{p(x)p(z)}dxdz$. 互信息可以写成KL散度的形式： $I(X;Z)=D_{KL}(P_{XZ}||P_X \times P_Z)$. 这个KL散度越大表示X和Z之间的关联越强，两个完全独立的随机变量，这个KL散度为0。

#### 1. Donsker-Varadhan representation.

KL散度满足下面约束（文章附录有证明）:


$$D_{KL}(P||Q) = sup_{T:\Omega \rightarrow R} E_P[T] - log(E_Q[e^T])$$


#### 2. f-divergence representation.

$$D_{KL}(P||Q)  \geq sup_{T \in F} ~E_P[T] - E_Q[e^{T-1}]$$

不过DV bound比这个更加stronger。


#### 3. Mutual Information Neural Estimator

我们目标是选择一个$\mathcal{F} 属于 T_{\theta}: \mathcal{X} \times \mathcal{Z}$, 用一个网络来表示这个函数，参数是$\theta \in \Theta$. 我们利用以下bound:

$$I(X;Z) \geq I_{\Theta}(X, Z)$$.

其中， $I_{\Theta}(X, Z)$由下式定义：

$$I_{\Theta}(X, Z) = sup_{\theta \in \Theta} E_{P_{XZ}}[T_{\theta}] - log(E_{P_X \times P_Z}[e^{T_{\theta}}])$$

上式其实定义了一个新的information measure。我们可以从经验分布中采样，用梯度上升来优化上面的式子。

**算法** 

![](/papers/mi/1.png)






#### 4. correcting the bias from the stochastic gradients





#### 5. Theoretical properties

estimator $\widehat{I(X;Z)_n}$ 是strong consistency的定义是对于所有的$\varepsilon \gt 0$,存在一个正整数N和一个网络使得

$$\forall n \geq N, |I(X, Z) - \widehat{I(X;Z)_n}| \leq \varepsilon$$ almost everywhere.



这时候consistency问题转换成了两个问题： 第一个是Approximation问题，和$\mathcal F$的大小相关；第二个是estimation问题，和empirical measures相关。第一个问题可以通过神经网络的universal approximation理论解决，第二个问题，classical consistency theo-rems for extremum estimators apply (Van de Geer, 2000)under mild conditions on the parameter space解决。


相应地引出两条引理。

**approximation** . 定义$\varepsilon \gt 0$. There exists a neural network parametrizing functions $T_{\theta}$ with parameters $\theta$ in some compact domain $\Theta \in R^k$, sunch that

$$|I(X, Z) - I_{|theta}(X, Z)| \leq \varepsilon, a.e.$$

**estimation** 定义$\varepsilon \gt 0$. 给出一系列神经网络$T_{\theta}, \theta \in R^k$,存在$N \in \mathcal N$,使得

$$\forall n \geq N, |\widehat{I(X;Z)_n} - I_{\Theta}(X, Z)| \leq \varepsilon, a.e.$$

由上面两条引理可以得到 **MINE is strongly consistent.**



