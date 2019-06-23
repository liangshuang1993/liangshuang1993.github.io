这篇博客主要讲GAN理论相关的一些知识.


#### Generalization and Equilibrium in Generative Adversarial Nets (GANs)

https://www.youtube.com/watch?v=V7TliSCqOwI

主要解决的问题:

- generalization. 加入生成器最终在经验分布样本上赢了(判别器判断不出来只能随机猜), 这个时候是否说明学到了真实分布(true distribution)? 如果判别器的容量很大并且样本数量很大,那么答案是yes. 
- 平衡 equilibrium. 这个2-person game中是否存在平衡?(不保证pure equilibrium)

**generalization**

1. 考虑一种生成器: 从$D_{real}$中的$(nlogn)/{\epsilon^2}$样本均匀分布

定理: 如果生成器具有capacity n, 它能够判别两个分布的概率$< \epsilon$


注意:

- 即使从$D_{real}$获得更多的样本也成立
- 表明目前的GAN在生成器的分布上不能保证足够的多样性

2. 如果样本数量$> (nlogn)/{\epsilon^2}$, 那么在样本上的表现和在整个分布上的表现一致.(within $\epsilon$)

因此针对于本文中提出的"neural net distance"泛化性存在.


idea 1: Deep nets针对它们的参数是Lipschitz的

idea 2:如果网络参数数量为n,那么只有$exp(n/ \epsilon)$个基本的不同的网络(其他的都是$\epsilon$-close to one of these) 






idea 3: 对于任意一个固定的判别器D, 一旦我们从$D_{real}$和$D_{synth}$中采样大于$nlogn/\epsilon^2$个样本, 那么最大以$exp(-n/\epsilon)$的概率区分两种样本的能力超过从全分布上区分的能力$\pm \epsilon$

由idea 2 和 idea 3 加上union bound推出, 在$nlogn/\epsilon^2$个样本上的经验NN距离和全分布上的NN距离相符.



**equilibrium**

假如用无限多个G叠加得到的G,那么它一定可以表示任意的分布, 可以赢.

定理: 用$nlog n/\epsilon^2$个G的叠加可以产生一个分布$D_{synth}$, 对任意的n个参数的D,它看上去像$D_{real}$ (区分概率 $< \epsilon$)


 > von Neumann Min-Max Theorem: 如果用无限的判别器的叠加代替判别器,用无限的生成器的叠加代替生成器, 那么存在平衡

![](/courses/stat/1.png)


如果D和G都是有n个参数的Deep nets, 那么存在$\epsilon-approximate equilibrium$, 当我们允许$nlog n/\epsilon^2$个叠加时.


之后可以用一个单独的deep net替代这些mixture.


