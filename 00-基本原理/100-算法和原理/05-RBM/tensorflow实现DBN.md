https://blog.csdn.net/m0_37924639/article/details/78962912

那么pretrain过程就是从V−H0V−H0层起，对H0−H1,H1−H2,⋯H0−H1,H1−H2,⋯逐层对每个RBM的权重W和偏置b进行训练。 
pretrain()经过一系列的辗转，调用到了Model类的pretrain_procedure()函数，又为每个RBM层分别调用了_pretrain_layer_and_gen_feed()函数。这一函数经过一系列的辗转后，调用了UnsupervisedModel类的fit()函数来对RBM进行训练，随后对训练集和验证集分别调用同为UnsupervisedModel类的transform()函数，用来生成“上层”的数据。


