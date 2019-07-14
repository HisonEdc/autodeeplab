1.genotypes.py : 定义了cell中节点和节点之间的8种连接方式。

2.operations.py：实现网络中常见的层操作：cell中的连接方式和ASPP。

3.model_search.py：核心部分，实现网络结构可微化，具体的3个类：

​					MixedOp类：cell中2个节点之间的可微；

​					Cell类：cell的可微；

​					AutoDeeplab类：网络路径的可微。

4.train_search.py：参数设置，预处理数据，train/val，指标计算，其中train有4种方式：

 					mode_0： 卷积参数w，cell参数w_alpha，网络路径参数w_beta共同更新；

 					mode_1：first-order approximation(AutoDeeplab主要训练方法)，分成trainA,trainB两个训练集，先训练trainA更新w,再训练trainB更新；

 					mode_2：和A类似，但是用上一轮的w计算w_alpha和w_beta的梯度；

 					mode_3：second-order approximation(DARTS主要实现方法，较复杂，多公式推导，详见论文DARTS)。

5.mypath.py：数据集路径设置文件。

6.dataloaders：数据集的处理相关文件。

7.utils文档：定义了saver储存相关方法，metric指标计算方法等辅助函数。



HOW TO FORK：1.在mypath.py中添加数据集路径；

​							  2.修改train_search.py的main方法中的参数，包括数据集名称，batch size，图片大小，optimizer相关参数等；

​							  3.命令行输入python train_search.py  搜索最优的cell结构和网络路径。

注意：another_method文档中是关于搜索空间的另一种构建方式，区别在于：

​		1.cell的搜索空间：参照DARTS论文，变换的不同尺寸(1/4,1/8,1/16,1/32)对应的cell的连接方式不同(从tensorflow给的源码来看，论文作者是设置了所有尺寸都是一种cell内部的连接方式，但DARTS中对不同尺寸设置不同的cell内部的连接方式)

​		2.网络路径的搜索空间：从论文给出的公式(6)看，在计算第l层的输出时，作者强制保证了l层和(l-2)层的图片尺寸一致，从概率上有点难以理解，且这种强制措施使得不同尺寸(1/4,1/8,1/16和1/32)的第一个cell和第二个cell变成特殊点(因为不存在l-1和l-2的情况)，同时论文作者没有说明这种特殊点的处理。将公式(6)的假设放宽为(l-2)的尺寸可变。