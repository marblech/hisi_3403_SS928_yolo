# hisi_3403_SS928_yolo
let yolo object detect run on hisi 3403 SS928. 

Currently, I have completed yolov5 onnx model covert to NNN npu om format.

And i have completed yolov5 v6.2 and last version prediction use c++ with ACL library.

usage:

build in ubuntu 18.04 + gcc 7.5.0 + opencv 4.8.1 with contrib + ACL

convert yolov5 model with ATC tools version : Ascend-cann-toolkit_5.13.t5.0.b050_linux-x86_64

deploy yolov5_3403_test and libyolov5_3403.so and all dependent third-party so to the board. 

config runtime env and run ./yolov5_3403_test

For more development detail, please follow my personal site on http://marblelog.com.

Thanks for warren@伟 some code copy from https://github.com/warren-wzw/Algorithm-deployment-template-of-each-platform 
