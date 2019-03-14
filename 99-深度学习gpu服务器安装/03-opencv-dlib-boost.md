### opencv 4.0.0
下载4.0.0.zip，解压缩，需要cmake

```
$ cd opencv-4.0.0
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
$ make 
$ sudo make install

```

### dlib
```
$ git clone https://github.com/davisking/dlib
$ cd dlib
$ mkdir build
$ cd build
$ cmake ..
$ make
$ sudo make install

```

### boost 1.69
```
$ ./bootstrap.sh
$ ./b2
$ sudo ./b2 install

```