最近两年表征学习非常火，这篇博客主要来跟踪相关的论文。


#### AET vs. AED: Unsupervised Representation Learning by Auto-EncodingTransformations rather than Data


*CVPR2019*

目前分类，检测，分割等任务主流的方法都是有监督学习，即需要大量的有标签的数据，这就限制了算法的应用场景。因此，用无监督的方式来学习deep feature representation可以解决没有足够labeled数据的问题。


在无监督方法中最具代表性的是Auto-Encoder（包括VAE）和GAN。  这里的auto-encoder（包括变种）都是要重建输入数据，因此将它们归类为**Auto-Encoding Data (AED)**. 对于GAN而言，输入是noise，被视作feature representation，为了为每张图片获得一个feature representation，可以也采用encoder-decoder结构，将generator视作decoder。这样的话，给一个输入图片，encoder可以得到noise representation，相当于结合了AED和GAN。最近这种方法多用于unsupervised和semi-supervised任务。

除了auto-encoder和GAN之外，还有其他一些**自学习**的方法。比如，通过预测两个patch之间的相对位置来训练网络，将image colorization视为任务来训练网络，classify image rotation等等。

和这些方法相反，这篇文章希望通过**auto-encoding transformations(AET)**, 而不是通过数据来学习feature representation。通过采样一些**operation**来对图像进行转换，训练一个auto-encoder来通过原始的图片和变换后的图片之间的learned feature representation直接重建这些**operation**


#### 定义

给定一个distribution T（比如image wraping， projective transformation，或者是基于GAN的一些变换）, 从中采样一种变换t，它可以将x变换为t(x).

定义一个encoder $E: x \rightarrow E(x)$, 目标是提取representation。同时也定义一个decoder $D: [E(x), E(t(x))] \rightarrow \hat{t} $  , 得到的是对t的预测。网络同时训练E和D，定义一个loss function $l(t, \hat{t})$。

下面讨论以下可以用哪些变换

- parameterized transformation。 用参数来定义变换，损失函数可以用$t, \hat{t}$的参数之间的不同来衡量。比如affine transformation和projective transformation，都是有一个参数化的变换矩阵$M$来定义，这个时候可以将loss function定义为$l(t_{\theta}, t_{\hat{\theta}})=\frac{1}{2}||M(\theta) - M(\hat{\theta})||$.

- GAN-induced transformation. $t_z(x)=G(x, z)$，z可以作为变换的参数。loss为$l(t_z, t_{\hat{z}})=\frac{1}{2}||z-\hat{z}||_2^2$

- non-parametric transformation. 即使某个变换t非常难以参数化，我们仍然可以构建一个loss，$l(t, \hat{t})=E_{x \sim X} dist(t(x), \hat{t}(x))$. 
