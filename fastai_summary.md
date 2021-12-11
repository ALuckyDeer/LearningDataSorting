- [ install](#head1)
- [ Learn.fit_one_cycle(n_epoch)](#head2)
- [Learn.fit_one_cycle() 的重启的余弦退火-超参数调教明细表](#head3)
- [ ImageDataLoaders.from_df](#head4)
# <span id="head1"> install</span>
pip install fastai -i https://pypi.douban.com/simple

# <span id="head2"> Learn.fit_one_cycle(n_epoch)</span>
Fit self.model for n_epoch using the 1cycle policy.
在学习的过程中逐步增大学习率目的是为了不至于陷入局部最小值，边学习边计算loss。

其次，当loss曲线向上扬即变大的时候，开始减小学习率，慢慢的趋近梯度最小值，loss也会慢慢减小。就如下图：
图为课程中的图

![](img/img.png)

该图x轴为迭代次数，y轴为学习率

![](img/img_1.png)

该图x轴为迭代次数，y轴为loss

结合两个图可以看出：

学习率首先逐渐变大，loss逐渐变小
当学习率达到训练时给的参数时，开始下降。
随着学习率不断降低，loss也开始降低。
这个算法被称为:learning rate annealing(学习率退火算法)。

# <span id="head3">Learn.fit_one_cycle() 的重启的余弦退火-超参数调教明细表</span>
fastai开发文档的超参数明细表:https://docs.fast.ai/callback.schedule.html#ParamScheduler
学习率在每个周期开始时重置为参数输入时的初始值，余弦退火部分描述的那样，逐渐减小

![](img/img_7.png)
这个链接的东西非常好：但是经过实验，发现fit里面的cycle已经舍弃了
在fastai2 里面Learner.fit(n_epoch, lr=None, wd=None, cbs=None, reset_opt=False)
cbs我感觉是combine_schedes
https://blog.floydhub.com/ten-techniques-from-fast-ai/  
ps我找到了中文：  https://blog.csdn.net/weixin_42137700/article/details/81529789

-------------------更新-------------------
fast.ai 不再推荐余弦退火，因为它不再是最高性能的通用学习率调度器。现在，这个荣誉属于单周期学习率调度器。
单周期学习速率调度器在2017年的论文 《Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates》 中被引入。
paper:[Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](paper/SCVFTONNULLR.pdf)
解释链接：https://bbs.cvmart.net/articles/4647/vote_count  


# <span id="head4"> ImageDataLoaders.from_df</span>
`ImageDataLoaders.from_df(df, path='.', valid_pct=0.2, seed=None, fn_col=0, folder=None, suff='', label_col=1, label_delim=None, y_block=None, valid_col=None, item_tfms=None, batch_tfms=None, bs=64, val_bs=None, shuffle=True, device=None)`
https://docs.fast.ai/vision.data.html#ImageDataLoaders.from_df
使用filename column和label column从df创建
如果标签列每一行包含多个标签，可以使用label_delim警告库您有一个多标签问题