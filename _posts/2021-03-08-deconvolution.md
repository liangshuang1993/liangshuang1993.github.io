反卷积，又称转置卷积，是上采样中一个比较常见的手段。与repeat相比，它可以学习一些权重，因此效果更好。


反卷积其实并不是卷积的逆，它只是可以恢复出原始输入的形状，而不能恢复出原始输入具体的值。其实用转置卷积称呼它更为准确。

我们定义输入为X，输出为Y，对于一个普通的CNN而言，我们将X重排为$[in*in, 1]$, Y重排为$[out*out, 1]$, 我们可以从原始的卷积核构建出一个C，使得Y=CX，C的维度为$[out*out,in*in]$.C其实是一个稀疏矩阵。

而反卷积的过程其实是给出了Y，我们可以根据下式得到一个和X维度相同的X', 
$$x'=C^TY$$.这其实就是反卷积的过程。

具体在实现的时候需要考虑padding和stride的影响。

pytorch中实现过程如下

```python

x = torch.Tensor([range(1, 26)])
x = x.reshape((1, 1, 5, 5))

deconv_layer = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)

y = deconv_layer(x)

```

我们如何自己实现这个过程呢？主要分为以下几个步骤：

- 首先确定输出tensor的大小，可以根据普通卷积计算输出大小的公式来计算，如果需要padding，则需要更新输入维度的大小。（其实是反卷积的输出维度）
- 对输入进行重排
- 构造矩阵$C^T$
- $y = C^T * X$
- 对输出进行重排，并去掉padding的部分

具体过程如下

```python
output_size = 5
input_size = 9 + 2 # padding
stride = 2
filter_size = 3
filter_np = weights[0, 0, :, :].detach()
print(filter_np.shape)

filter_np_matrix = np.zeros((output_size, output_size, input_size, input_size))
for h in range(output_size):
    for w in range(output_size):
        start_h = h*stride
        start_w = w*stride
        end_h = start_h + filter_size
        end_w = start_w + filter_size
        filter_np_matrix[h, w, start_h:end_h, start_w:end_w] = filter_np.detach()

filter_np_matrix = torch.from_numpy(filter_np_matrix)
filter_np_matrix = filter_np_matrix.reshape((output_size*output_size, input_size*input_size))

new_x = x.reshape((1, output_size*output_size))

y_2 = torch.matmul(new_x.float(), filter_np_matrix.float()) + bias
y_2 = y_2.reshape((input_size, input_size))
y_2 = y_2[1:-1, 1:-1]
print(y_2)
print(y_2 - y) 

```


查了一些博客，发现反卷积还有另一种计算方法，这里记录如下：

- 首先根据反卷积的参数和输入来padding出一个新的输入new_x,比如stride=1的时候，需要在原始x中每个元素之间都要padding一个0
- 将原始卷积核进行180度翻转，得到一个新的卷积核
- 用新的卷积核对新的输入做CNN

代码如下：

```python

weights = weights.squeeze()
new_weights = torch.zeros(weights.shape)
for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        new_weights[weights.shape[1] -1 - i][weights.shape[0] -1 - j] = weights[i, j]

weights = new_weights.unsqueeze(0).unsqueeze(0)

new_x = torch.zeros((1, 1, 9, 9))

for i in range(5):
    for j in range(5):
        new_x[:, :, 2*i, 2*j] = x[:, :, i,j]

conv_layer = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
conv_layer.weight.data = weights
conv_layer.bias.data = bias

y_1 = conv_layer(new_x)

print(y_1-y)

```


可以再看下NCNN中的实现，https://github.com/Tencent/ncnn/blob/b93775a27273618501a15a235355738cda102a38/src/layer/deconvolution.cpp

