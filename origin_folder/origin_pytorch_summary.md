# new_*
根据现有的张量创建张量。 这些方法将重用输入张量的属性，例如， dtype，除非设置新的值进行覆盖  
`x = x.new_ones(5, 3, dtype=torch.double) # new_* 方法来创建对象`

# autograd 自动求导

# 方法以_结尾
任何 以``_`` 结尾的操作都会用结果替换原变量. 例如: ``x.copy_(y)``, ``x.t_()``, 都会改变 ``x``.

# view类似于reshape
```x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  #  size -1 从其他维度推断
print(x.size(), y.size(), z.size())
```

# .item()获得值

# from_numpy自动转化

使用from_numpy自动转化
```
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
```
```
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```
# .to 移动tensor到其他device中
```angular2html
# is_available 函数判断是否有cuda可以使用
# ``torch.device``将张量移动到指定的设备中
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA 设备对象
    y = torch.ones_like(x, device=device)  # 直接从GPU创建张量
    x = x.to(device)                       # 或者直接使用``.to("cuda")``将张量移动到cuda中
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` 也会对变量的类型做更改
```

# *args 和 **kwargs
*args 用来将参数打包成tuple给函数体调用
```angular2html
def function(x,y,*args):
    print(x, y, args)
function(1, 2, 3, 4, 5)
```
输出
```angular2html
1 2 (3,4,5)
```
**kwargs 打包关键字参数成dict给函数体调用
```angular2html
def function(**kwargs):
    print(kwargs)

function(a=1, b=2, c=3)
```
输出
```angular2html
{'a':1,'b':2,'c':3}
```
参数arg、*args、**kwargs三个参数的位置必须是一定的。必须是(arg,*args,**kwargs)这个顺序，否则程序会报错。

# torch.to
参数
`torch.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format)`  
张量在设备间转换

# .requires_grad_( ... ) 
它可以改变现有张量的 requires_grad属性。 如果没有指定的话，默认输入的flag是 False。
前面带有_的是函数，后面的requires_grad是属性

# torch.Tensor的.requires_grad
如果设置 .requires_grad 为 True，那么将会追踪所有对于该张量的操作。  
如果没有不设置tensor,那么.requires_grad默认为false 然后不会记录，后面也就不能用.backwards()了

# 公式推导未学明白，数学基础需要补习，呜呜呜
[vector-雅可比矩阵](https://github.com/zergtant/pytorch-handbook/blob/master/chapter1/2_autograd_tutorial.ipynb)

#nn.Linear
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)  
Applies a linear transformation to the incoming data:  
y = xA^T + b  
in_features – size of each input sample  


# max_pool2d(input,B)的B如果是方形的可以只写一个数字，如果不是就写3*2这种类型的

# x = x.view(-1, self.num_flat_features(x))
拉平后的特征空间大小  
pytorch官方论坛考古：https://discuss.pytorch.org/t/understand-nn-module/8416  

# forward
在模型中必须要定义 forward 函数，backward 函数（用来计算梯度）会被autograd自动创建。 可以在 forward 函数中使用任何针对 Tensor 的操作。  

# ``nn.Conv2d`` 接受一个4维的张量，
``每一维分别是sSamples * nChannels * Height * Width（样本数*通道数*高*宽）``

# 