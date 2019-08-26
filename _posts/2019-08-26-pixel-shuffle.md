#### Pixel Shuffle

好像一些说法不太一致，本文讲述的其实是sub-pixel convolution, 来源于Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. 在对图像处理时，有时候需要对图像放大，如super-resolution， 这时候采用的方法有直接上采样，反卷积，插值等，这篇文章提出了一种新的方法，网络的前l层是CNN，后面一层为sub-pixel convolution layer，对LR特征图进行upscale。

将低维空间记为LR,高维空间记为HR.

有一种upscale的方法是用一个$1 /r $为步长的卷积，可以转换成从LR到HR的插值，perforate或者up-pooling，跟着一层HR空间上步长为1的CNN。这种实现方式将计算量增大了$r^2$倍。主要是因为卷积是在HR空间中进行的。


另一种方式是LR空间中，步长为$ 1 / r$，卷积为$W_s$, 大小为$k_s$。


注：这里的$1 / r$步长的卷积看了很久没有懂， https://oldpan.me/archives/upsample-convolve-efficient-sub-pixel-convolutional-layers 提供了一张图比较清晰的讲述了这个过程：

![](/papers/images/2.png)

左侧为反卷积的过程，右侧为sub-pixel convolution的过程（步长为1 / 2）,灰色部分代表被截去的部分。将右边的filter进行reverse，将灰色部分去掉之后，发现其实右边就等同于左边。


$$I^{SR} = f^L(I^{LR}) = PS(W_L * f^{L-1}(I^{LR}) + b_L)$$

PS指周期性的shuffle操作，将$H \times W \times C \cdot r^2$转成$rH \times rW \times C$, 具体所如下所示。

![](/papers/images/1.png)

可以用下式表示：

$$PS(T)_{x, y, c} = T _{[x /r ], [y/ r], C \cdot r \cdot mod(y, r) + c}$$

卷积操作$W_L$大小为$n_{L-1} \times r^2C \times k_L \times k_L$

最终loss的计算采用MSE loss。

$$ l(W_{1:L}, b_{1:L}) = \frac{1}{r^2HW} \sum^{rH}_{x=1}\sum^{rH}_{y=1}(I^{HR}_{x, y} - f^L_{x, y}(I^{LR}_{x, y}))^2$$

另外，需要注意的是，上述的操作可以在training中省去。我们可以先pre-shuffle training data来匹配到PS之前输出层的大小，这样的话我们提出的层会在训练中比反卷积层快$log_2r^2$倍，比upscale+CNN快$r^2$倍。

**实现细节**

- $$ l=3 $$
- $$(f_1, n_1) = (5, 64)$$
- $$(f_2, n_2) = (3, 32)$$
- $$(f_3) = 3$$

