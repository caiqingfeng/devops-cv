### 说明
运行dlib自带的68-face-landmarks  
有2个人脸没有检测出来，效果已经很惊艳了，适应不同角度，光线  

### install & config dlib
download dlib or git clone dlib  
$ cd vendors/dlib  
$ mkdir build  
$ cd build   
$ cmake ..  

### dlib examples
$ cd 02-dlib-101  
$ ln -s ~/Desktop/vendors/dlib dlib  
$ cd 01-dlib-examples   
$ mkdir build   
$ cd build   
$ cmake ..  
$ ./face_landmark_detection_ex   ~/Desktop/vendors/pre_trained_weights/shape_predictor_68_face_landmarks.dat  ~/Desktop/vendors/opencv-rawdata/team-building-02.jpg  
