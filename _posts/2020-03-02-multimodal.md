之前接触到的深度学习应用场景通常是一个单一信号，单一模式。这篇文章则是focus在更复杂的场景——vision和natual language的结合。这篇文章主要从三个角度出发：

- learning multimodal **representation**
- **fusion** of multimodal signals at various levels
- multimodal **applications**



#### 1. Representations

#### 1.1 Unimodal Embeddings

**Visual representation**

通常将图像分类网络中的最后一层CNN的输出作为image embedding。比如AlexNet, VGG nets, ResNet的最后一层的输出。另外，还可以选择和semantic更相关的表征，比如物体检测网络中得到的物体区域的feature及相应标签。

**language representations**

通常是有一个language model。目前有很多比较复杂的结构，比如ELMo, GPT, BERT都可以用于提取embedding。text embedding可以有不同的level，如work level, subword level, phrase level, sentence level, paragraph level.


**vector之间的计算**

之前通常见到word embedding之间可以进行计算，比如vector('King') - Vector('Man') + Vector('Woman') 约等于 Vector('Queen')。 图像中其实也有类似的计算方法，比如GAN得到的feature也有类似的效果。

#### 1.2 Multimodal Representations

为了更好的感知世界，用一个joint embedding来表示多模态的数据是非常必要的。假设相应的表征在不同的模态下有着相似的neighborhood structure，虽然在某一个模态下训练数据为0，但是其他模态下有很多训练数据，那么仍然可以算出它的embedding。举个例子，image labeling应用中，有一种物体从未在训练集中出现，但是我们可以将物体投影到linguistic space，可以看linguistic space中离它们最近的label是什么。


**unsupervised**

- 重建原始数据，重建网络的中间一层是一个共享的表征空间。
- word embedding和image embedding直接相加或者concat
- 增大相应的Skip-Gram word embedding和image feature之间的相似性
- 最大化不同模态之间embedding的mutual information/correlation
- 将image region/fragments和sentence fragment或者attribute word关联在一起。 这样可以产生更加精细的multimodal embedding。可以用attention做对齐。

**supervised**

- Learning factorized multimodal representations 这篇文章将representation分解为两个独立的部分：有监督所需要的multimodal discriminative fractor和无监督所需要的intramodality generative fractor。 前者是所有模态之间共享的，对discriminative task很有用，后者则是可以用来重建。

**zero-shot learning**

**transformer based methods**

可以将原始仅用于文本的BERT引申到图像上面。已经有一些相关模型了，这些模型需要引入新的token来表示visual feature，如Unicoder-VL, VL-BERT, VisualBERT, VideoBERT, B2T2, LXMERT, ViLBERT, OmniNet. 一些NLP的研究也发现多任务的训练可以提升BERT表征的泛化能力。

#### 2. Fusion

fusion根据阶段也分为两种类型： early fusion，或者feature-level fusion直接将不同模态提取出来的feature集合起来，来加强模态内部的交互，可能会造成模态之间交互被抑制。late fusion或者model-level fusion
为每一种模态建立一种独立的模型，然后结合他们的输出。更多的研究集中在intermediate或者middle-levelmethod，在一个很深的网络的多个层进行fusion。不过其实这些stage的界限不是特别的明显。

**operation-based**

concate或者weighted sum。

**attention-based**

首先介绍下image上的attention

有一种叫做stacked attention networks(SAN)是用多层attention来一步步地得到结果。

dynamic memory network(DMN)用基于attention的GRU来更新memory

Bottom-up和top-down attention(Up-Down)模拟人类视觉系统，将两种visual attention结合在一起。

还有image和text的co-attention

这种attention是对称的结构，并行式的结构可以同时获得image和language attention，交替式的结构用一个胶囊结构首先用linguistic feature生成atteneded image vector，接下来用attented image vector来attended linguistic feature。和并行式的结构类似，dual attention network(DAN)同时预测image和language attention distribution，attention model是condition在相应模态的feature vector和memory vector的。

再介绍下attention in bimodal transformer

transformer结构中本身每层就是一层attention。OmniNet在decoder 中用gated multi-head attention来融合其他模态下的vector。LXMERT分别用独立的encoder来学习单模态的feature，然后再用一个跨模态encoder利用cross-attention layer来学习跨模态的feature。 

**Bilinear Pooling-based**

计算visual feature vector和textual feature vector之间的点积。


#### 3. Application

这篇文章主要将image captioning, text-to-image generation, VQA这些方面。


