### A survey on contrastive self-supervised learning

#### 前言

深度学习虽然在各个方面已经取得了极大的进步，但是从有标签的数据来学习features已经快要达到饱和了，这是因为标注上百万的数据需要耗费大量的人力。现实中通常有非常多的无标签数据，但是supervised方法不能使用这些数据，然而，self-supervised方法却可以从这些数据中学习feature representations.

**supervised方法的缺点**
- expensive annotations
- generalization error
- spurious correlations
- adversarial attacks

最近，self-supervised learning methods集成了generative和contrastive方法来利用无标签的数据学习潜在的representations。一种常用的方法是提出一些pretext task用伪标签来学习特征，这些任务通常有image-inpainting, colorizing greyscale images, jigsaw puzzles, super-resolution, video frame prediction, audio-visual correspondence等等。

随着GAN的流行，cycleGAN,styleGAN等模型促使研究者开始研究基于GAN的方法。但是，GAN很难训练：1）模型不容易收敛;2)判别器太成功了;3)需要合适的同步机制

和生成模型不同，contrastive learning是判别式的方法，它的目的是让相似的样本靠近，差异大的样本远离。因此，需要定义一个*similarity metric*. 图像领域中一个常见的做法是，通过数据增强得到新的样本，这些样本和原始的样本是同一类，是positive的样本，数据集中的其他则是negtive的样本。模型被训练用来区分positive和negtive样本，通常借助一些*pretext task*.

这方面早期的工作通常将instance-level classification方法和contrastive leraning结合在一起。不过，最近一些新的方法如SwAV, MoCO, SimCLR可以在ImageNet上产生和有监督方法可比的结果。同样地，PIRL, Selfie等一些方法反应了pretext task的有效性，以及它们如何影响模型的结果。

#### Pretext Tasks

pretext task用伪标签来学习数据representation的自监督任务。伪标签通常是基于数据的一些已知属性自动生成的。从pretext task中学习到的模型可以用于任意的下游任务，如CV中的classification，segmentation，detection等。常见的pretext任务分为以下四种：color transformation，geometric transformation，context-based task和cross-modal based task。前面两类都比较简单，主要讲下后面两类。

**context-based task**

1. jigsaw puzzle

将图片分为几个patch，打乱顺序后的patch组成的新的图片为positive的样本，数据集中其他样本为negtive的样本。

2. frame order based

通常用于数据随着时间变化的场景，比如sensor data或者视频。可以将视频中的每一帧打乱数据组成positive样本。也可以随机选择长度相同的两段，或者在video上应用spatial augmentation。同一个video中截取出来的clip会接近一些，不同的video截取出来的远一些。

3. future prediction

针对随时间改变的数据一种常用的策略是predict future或者missing information。比如CPC。

4. view prediction

针对同一个场景有多个view的情况。


5. pre-text task

pre-text task的选择依赖于要解决的问题。虽然contrastive learning提出了很多方法，但是如何选定一个合适的pre-text task依然非常重要。一些场景中的transformation不一定适用于另外一些场景。


#### Architectures

![](/img/posts/07.png)

##### 1. End-to-End learning

这类的模型倾向于用一个很大的batch size。除了被选中的原始图片和它的增强的版本，batch中其他所有的样本都是负样本。模型中有一个query encoder和一个key encoder。两个encoder可以不同，来学习同一个样本不同的表征。利用一个contrastive loss，模型让positive sample更接近，negtive sample远离原始样本。这里Q是用原始样本训练的，而K是用增强的样本（正样本）和负样本训练的。通常使用的similarity metric是cosine similarity。

SimCLR发现end-to-end结构在复杂性方面很简单，但是用很大的batch size和较大的epoch会产生更好的结果。不过，通常batch size大小受到GPU大小限制。

##### 2. Using a Memory Bank

memory bank的目的是积累样本的大量特征表示，这些特征表示在训练期间用作负样本。

memory bank中存储这数据集中每个样本的feature representation。这样，在使用的时候不需要增大training batch size，直接从memory bank中取出负样本的特征即可。

然而，由于memory bank中的数据很快就过时了，更新memory bank中的数据依然计算量很大。



##### 3. Using a Momentum Encoder

momentum encoder 和 Q使用相同的参数，每步之后并不进行BP，用下式进行更新：

$$\theta'_k = \theta_k + (1-m) \theta_q$$

只有$\theta_q$会通过BP更新。

这种结构的优点在于他不需要两个单独的encoder，并且他不需要维护一个memory bank。



##### 4. Clustering Feature Representations


这种结构和end-to-end结构类似，但是两个encoder共享参数，此外，它没有用instance-based contrastive方法，而是利用了clustering算法来使相似的样本靠近。

最近的工作SwAV就是采用这种方法。它的目标不仅仅在于将同样的样本聚集在一起，还包括其他样本中相似的也要聚集在一起。比如，cats要跟dogs（同为animal）要比跟houses更近。

instance-based学习中，batch中所有其他的样本被视为负样本，但是这些样本中可能还包含着同一种类别，比如其他的cats，但是这种算法会强制模型将这些样本原理原始样本。这种问题可以被现在的clustering-based方法解决。


























#### Encoders

encoder在self-supervised中起到非常重要的作用，没有有效的feature representation，一个分类模型不可能从不同的类别中将它们区分开来。很多工作都是利用了ResNet的变种，ResNet-50是最常使用的模型（它均衡了模型大小和learning capability）

encoder的某一特定的层输出的结果通过池化得到一个单一维度的feature vector。根据方法的不同，它们可能本上采样或者下采样。有一项工作采用了ResNet-50，它将res5的输出做了average-pooled，得到一个2048维的特征。之后他们利用了一个线性层来得到一个128维的特征。在对比实验中，他们比较了其他层的输出，比如res2，res3,res4，发现encoder较晚阶段的输出通常比较早阶段的输出相比是一个更好的表征。

相似地，还有一项工作用了传统的ResNet作为encoder，特征是从average pooling层提取的，之后一个shallow MLP（只有一个hidden layer）将这个特征映射到一个隐空间，之后应用了contrastive loss。在action recognition方面，常用的encoder是3D-ResNet。

























#### Training


训练需要一个pretext task来利用contrastive loss，这就需要一个合适的similarity metric。

常用的metric是

$$cos_sim(A, B) = \frac{AB}{||A||~||B||}$$


利用NCE来学习的loss定义为:


$$L_{NCE} = -log\frac{exp(sim(q, k_+)/\tau)}{exp(sim(q, k_+)/\tau) + exp(sim(q, k_\_)/\tau)}$$


其中q是原始样本,$k_+$是正样本，$k_\_$是负样本，$\tau$是一个超参tempreture coefficient，sim即为similarity function。



如果负样本的数量比较多，可以用NCE的变种InfoNCE，


$$L_{InfoNCE} = -log\frac{exp(sim(q, k_+)/\tau)}{exp(sim(q, k_+)/\tau) + \sum_{i=0}^K{exp(sim(q, k_i)/\tau)}}$$



常用的优化器有SGD，Adam。还有一些end-to-end方法用了很大的batch size，这时候用标准的SGD再家一个linear  learning rate scaling来训练不稳定，LASR用了cosine learning rate。LARS和其他优化器如Adam的区别在于：LARS对每一层用了不同的learning rate使得训练更稳定；用了weight norm来控制训练速度。




#### Downstream Tasks

通常CV中用self-supervised learning涉及到两种任务：一个是pretext task，一个是downstream task。Downstream task有classification，detection，segmentation，future prediction等等。

为了评价self-supervised 方法学习到的feature对下游任务的有效性，通常用kernel visualization，feature map visualization，nearest-neighbor based方法来评估。

#### Benchmarks


下图汇总了不同方法在ImageNet上面的表现。目前很多self-supervised方法都能取得很好的效果了。

![](/img/posts/09.png)


![](/img/posts/08.png)

#### Contrastive Learning in NLP


最近有很多NLP领域相关的self-supervised的工作。这里就再不逐一介绍了。



#### Discussion


目前self-supervised learning还需要一些solid justification。而且各种模型选择的data augmentation和pretext task都不同，不好直接做对比选择出一个通用的在各个数据集上都work的模型。

另外，有时候原始样本和负样本之间的相似度很低，这个时候contrastive loss也很小，模型收敛的能力会降低。为了利用更多的负样本，一些top的方法都增加了batch size或者维护一个很大的memory bank。还有一种方法利用了hard negtive mixing策略来加速学习。不过这都会引入一些超惨，需要根据数据集来调试，不一定适用于其他数据集。

还有就是self-supervised learning中，数据自己提供了supervision。会带来bias。