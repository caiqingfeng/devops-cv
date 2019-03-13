### 原装的darknet 对opencv4支持的有些问题
```
$ git clone https://github.com/tiagoshibata/darknet
$ cd darknet
修改Makefile，把gpu=1, cudnn=1, opencv=1

$ make
$ wget https://pjreddie.com/media/files/yolov3.weights

```
