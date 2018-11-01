https://blog.csdn.net/Real_Myth/article/details/52755109?utm_source=blogxgwz2

Face detection, alignment, verification and identification(recognization)，本别代表从一张图中识别出人脸位置，把人脸上的特征点定位，人脸校验和人脸识别。（后两者的区别在于，人脸校验是要给你两张脸问你是不是同一个人，人脸识别是给你一张脸和一个库问你这张脸是库里的谁。


技术宅如何躲开大数据？解析人脸识别技术实现方式


人脸检测（detection）在OpenCV中早就有直接能拿来用的haar分类器，基于Viola-Jones算法。但是毕竟是老掉牙的技术，Precision/Recall曲线渣到不行，在实际工程中根本没法给boss看，作为MSRA脑残粉，这里介绍一种MSRA在14年的最新技术：Joint Cascade Face Detection and Alignment（ECCV14)。这篇文章直接在30ms的时间里把detection和alignment都给做了，PR曲线彪到很高，时效性高，内存占用却非常低，在一些库上虐了Face++和Google Picasa.

人脸校准（alignment）是给你一张脸，你给我找出我需要的特征点的位置，比如鼻子左侧，鼻孔下侧，瞳孔位置，上嘴唇下侧等等点的位置