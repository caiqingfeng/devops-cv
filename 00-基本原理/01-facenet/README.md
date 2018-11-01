https://blog.csdn.net/chzylucky/article/details/79680986?utm_source=blogxgwz4


1. 下载安装tensorflow
2. 设置python路径

$ export PYTHONPATH=`pwd`/src  （此步特别关键如果设置错误后面将无法运行）

1. 下载lfw数据
2. 对齐lfw数据（使用mtcnn模型）
   
$  python src/align/align_dataset_mtcnn.py ~/Desktop/vendors/lfw ~/Desktop/vendors/lfw_aligned --image_size 160 --margin 32 --random_order--gpu_memory_fraction 0.25

3. 验证测试
$ python src/validate_on_lfw.py ~/Desktop/vendors/lfw_aligned ~/Desktop/vendors/facenet-models/