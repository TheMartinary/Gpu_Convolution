Image classification with Convolutional Neural Networks
---------------------------------------------------------
The code provided here is a pure C implementation of Google's [mobilenetv2](https://arxiv.org/abs/1801.04381), trained for the [imagenet](www.image-net.org/) competition.
As inspired by the legendary console game [robotfindskitten](https://en.wikipedia.org/wiki/Robotfindskitten) this network is particularly happy when it finds kitten :)

Usage
-----
To download a single image and pass it through the network:
```shell
make check
```

To test on several images specified by a list in the makefile
```shell
make check_all
```

Note that the included images all contain kitten, but sometimes the network misclassifies.
This is to be expected though, and the correctness of the network is verified by checking the found classification index to the reference indexes in the makefile.

In principle all images from [imagenet](http://www.image-net.org/explore) can be used (and of course also new images!).

A debug mode is also available, which stores intermediate results in text files for comparison to a reference.
To build in debug mode, first clean and then build with the added ```DEBUG=1``` flag
```shell
make clean; make DEBUG=1
```

Similarly there is a CPU ONLY mode, which uses only CPU functions.
To enable first clean and then build with the added ```CPU=1``` flag
```shell
make clean; make CPU=1
```

Finally there is a TIMING mode, which enables some of the default timers in the code.
To enable first clean and then build with the added ```TIMING=1``` flag
```shell
make clean; make TIMING=1
```

Note that these various modes can be mixed as desired.

Requirements
------------
To be able to use this network not much is needed.
A C-compiler and libpng should be sufficient, although some manual effort might be required if libpng (possibly the development files), ```make```, ```gcc```, ```wget``` and/or ```convert``` are not present on your system.
Most modern linux distributions will have these available however.
