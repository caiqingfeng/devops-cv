data  uchar型的指针。Mat类分为了两个部分:矩阵头和指向矩阵数据部分的指针，data就是指向矩阵数据的指针。
dims 矩阵的维度，例如5*6矩阵是二维矩阵，则dims=2，三维矩阵dims=3.
rows  矩阵的行数
cols   矩阵的列数
size 矩阵的大小，size(cols,rows),如果矩阵的维数大于2，则是size(-1,-1)
channels 矩阵元素拥有的通道数，例如常见的彩色图像，每一个像素由RGB三部分组成，则channels = 3
depth 
矩阵中元素的一个通道的数据类型，这个值和type是相关的。例如 type为 CV_16SC2，一个2通道的16位的有符号整数。那么，depth则是CV_16S。depth也是一系列的预定义值， 
将type的预定义值去掉通道信息就是depth值: 
CV_8U CV_8S CV_16U CV_16S CV_32S CV_32F CV_64F
elemSize 
矩阵一个元素占用的字节数，例如：type是CV_16SC3，那么elemSize = 3 * 16 / 8 = 6 bytes
elemSize1 
矩阵元素一个通道占用的字节数，例如：type是CV_16CS3，那么elemSize1 = 16  / 8 = 2 bytes = elemSize / channels
step[0]是矩阵中一行元素的字节数。

step[1]是矩阵中一个元素的自己数，也就是和上面所说的elemSize相等。

上面说到，Mat中一个uchar* data指向矩阵数据的首地址，而现在又知道了每一行和每一个元素的数据大小，就可以快速的访问Mat中的任意元素了。下面公式：

addr(M_{i,j}) = M.data + M.step[0]*i + M.step[1]*j

