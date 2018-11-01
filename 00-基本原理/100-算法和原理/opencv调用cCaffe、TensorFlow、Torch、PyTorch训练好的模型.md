https://blog.csdn.net/shanglianlm/article/details/80030569

opencv调用cCaffe、TensorFlow、Torch、PyTorch训练好的模型
2018年04月21日 16:04:34 mingo_敏 阅读数：2619 标签： opencv 深度学习 框架  更多
个人分类： C\C++ Opencv
版权声明：本文为博主原创文章，欢迎转载，转载请注明出处。	https://blog.csdn.net/shanglianlm/article/details/80030569
本文地址：https://blog.csdn.net/shanglianlm/article/details/80030569

OpenCV 3.3版本发布，对深度学习（dnn模块）提供了更好的支持，dnn模块目前支持Caffe、TensorFlow、Torch、PyTorch等深度学习框架。

1 加载模型成网络
1-1 调用caffe模型
核心代码：

String modelDesc = "../face/deploy.prototxt";
String modelBinary = "../face/res10_300x300_ssd_iter_140000.caffemodel";
// 初始化网络
dnn::Net net = readNetFromCaffe(modelDesc, modelBinary);
if (net.empty()){
    printf("could not load net...\n");
    return -1;
}

调用caffe示例 
OpenCV基于残差网络实现人脸检测 
http://blog.51cto.com/gloomyfish/2094611?lb=

1-2 调用TensorFlow模型
a TensorFlow训练模型，并保存成.pb文件 
b 使用opencv的readNetFromTensorflow函数加载.pb文件 
核心代码：

String labels_txt_file ="../inception5h/imagenet_comp_graph_label_strings.txt";
String tf_pb_file ="../inception5h/tensorflow_inception_graph.pb";

// 加载网络  
Net net =readNetFromTensorflow(tf_pb_file);   
if(net.empty()){
    printf("read caffe model data failure...\n");
    return -1;
}

调用TensorFlow示例 
OpenCV 基于Inception模型图像分类 
https://mp.weixin.qq.com/s?__biz=MzA4MDExMDEyMw==&mid=2247484278&idx=1&sn=e5074be2ba35c17bf34685864b6d34d7&chksm=9fa87432a8dffd246f1c88fea1dc7e348abb3d93c93e0834da881852dcfea68f5609ce927038&mpshare=1&scene=23&srcid=0421tNU3Tvp8N4oEUip7LYE9#rd

1-3 调用Darknet模型
核心代码：

String modelConfiguration = "../yolov2-tiny-voc/yolov2-tiny-voc.cfg";
String modelBinary = "../yolov2-tiny-voc/yolov2-tiny-voc.weights";
dnn::Net net = readNetFromDarknet(modelConfiguration, modelBinary);
if (net.empty())
{
    printf("Could not load net...\n");
    return;
}

调用Darknet示例 
OpenCV DNN之YOLO实时对象检测 
http://blog.51cto.com/gloomyfish/2095418

2 加载测试数据
blobFromImage 转换数据为四维Blob图片。 
核心代码：

// 加载图像
Mat frame = imread("../123.jpg");
Mat inputBlob = blobFromImage(frame, 1/255.F, Size(416, 416), Scalar(), true, false);
net.setInput(inputBlob, "data");

blobFromImage函数解释

Mat blobFromImage(InputArray image, double scalefactor=1.0, const Size& size = Size(),  const Scalar& mean = Scalar(), bool swapRB=true, bool crop=true);  
    /** @brief Creates 4-dimensional blob from image. Optionally resizes and crops @p image from center,  
    *  subtract @p mean values, scales values by @p scalefactor, swap Blue and Red channels.  
    *  @param image input image (with 1-, 3- or 4-channels).  
    *  @param size spatial size for output image  
    *  @param mean scalar with mean values which are subtracted from channels. Values are intended  
    *  to be in (mean-R, mean-G, mean-B) order if @p image has BGR ordering and @p swapRB is true.  
    *  @param scalefactor multiplier for @p image values.  
    *  @param swapRB flag which indicates that swap first and last channels  
    *  in 3-channel image is necessary.  
    *  @param crop flag which indicates whether image will be cropped after resize or not  
    *  @details if @p crop is true, input image is resized so one side after resize is equal to corresponding  
    *  dimension in @p size and another one is equal or larger. Then, crop from the center is performed.  
    *  If @p crop is false, direct resize without cropping and preserving aspect ratio is performed.  
    *  @returns 4-dimansional Mat with NCHW dimensions order.  
    */  

第一个参数，InputArray image，表示输入的图像，可以是opencv的mat数据类型。
第二个参数，scalefactor，这个参数很重要的，如果训练时，是归一化到0-1之间，那么这个参数就应该为0.00390625f （1/256），否则为1.0
第三个参数，size，应该与训练时的输入图像尺寸保持一致。
第四个参数，mean，这个主要在caffe中用到，caffe中经常会用到训练数据的均值。tf中貌似没有用到均值文件。
第五个参数，swapRB，是否交换图像第1个通道和最后一个通道的顺序。
第六个参数，crop，如果为true，就是裁剪图像，如果为false，就是等比例放缩图像。

3 输出结果
//检测 darknet
Mat detectionMat = net.forward("detection_out");
1
2
//分类 Inception
prob =net.forward("softmax2");
1
2
//tf 
pred = net.forward("fc2/prob");  
1
2
4 用到的一些函数
4-1 在dnn中从磁盘加载图片##

cv2.dnn.blobFromImage 
cv2.dnn.blobFromImages

4-2 用create方法直接从各种框架中导出模型
cv2.dnn.createCaffeImporter 
cv2.dnn.createTensorFlowImporter 
cv2.dnn.createTorchImporter

4-3 使用read方法从磁盘直接加载序列化模型
cv2.dnn.readNetFromCaffe 
cv2.dnn.readNetFromTensorFlow 
cv2.dnn.readNetFromTorch 
cv2.dnn.readhTorchBlob

从磁盘加载完模型之后，可以用.forward方法来向前传播我们的图像，获取结果。

参考资料 
1 OpenCV Tutorials 
https://docs.opencv.org/3.4.1/d9/df8/tutorial_root.html 
2 opencv调用tf训练好的模型 
https://blog.csdn.net/hust_bochu_xuchao/article/details/79428759 
3 Deep Learning with OpenCVhttps://www.pyimagesearch.com/2017/08/21/deep-learning-with-opencv/ 
4 opencv的dnn解析 
https://blog.csdn.net/langb2014/article/details/51286828