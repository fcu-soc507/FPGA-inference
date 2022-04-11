# FPGA-inference

# Deploy Agilev4 on ZCU102 and data transmission
## introduction
Professional service robots need object detection, object tracking, hazard detection and other functions to move in the crowd. However, due to cost and environmental considerations, it is impossible for us to install a supercomputer with powerful computing power on the service robot.Therefore, we propose a precise and fast high-performance object detection and tracking identification system that can be executed on the edge computing platform, Jetson Nano, at a real-time speed of 30 fps.

In this part, we deploy the object detection on the FPGA platform and use the HTTP communication protocol, but UDP instead of TCP for data transmission.
## How to use
>1.install image
>
[etcher](https://www.balena.io/etcher/)

[image download](https://drive.google.com/file/d/147AWFSlIql7TuC5glqqHnXguY3aj0za1/view?usp=sharing)

[nano zip](https://drive.google.com/file/d/108nCZVAG70VNibhEjQqGPz2Bz0BQnwNI/view?usp=sharing)

>2.IP set
>
modify network_set.sh
```python=2
ifconfig eth0 (yours IP)
ifconfig eth0 netmask (yours mask)
```
and
```
network_restart.sh
network_set.sh
```
>3.ZCU-102 inference
```
cd Vitis-AI/demo/Vitis-AI-Library/samples/Agilev4/moonshoot
```
```
# LD_LIBRARY_PATH=./sort/build/ ./moon_shoot usb model gop IP PORT 

LD_LIBRARY_PATH=./sort/build/ ./moon_shoot 0 agilev4_moon_8class 5  # two stage in one
LD_LIBRARY_PATH=./sort/build/ ./moon_shoot 0 agilev4_leaky_moon 5   # moon datasets
```
>4.Jetson get information

unzip UDP folder to /home/(username)/
<div style="text-align:center">
  <img src="https://i.imgur.com/eLcchc3.png">
</div>

```
cd UDP
./ros_publish
```
>5.ROS
>
unzip subscrible.py to /home/(username)/catkin_ws/src/camera_2d_lidar_calibration/src
<div style="text-align:center"><img src="https://i.imgur.com/CPKnlbI.png"></div>
<br>

```
# label||bbox.x||bbox.y||bbox.weight||bbox.height

cd catkin_ws/src/camera_2d_lidar_calibration/src
python3 subscrible.py
```
## Object detection on ZCU-102
### 1. model transfer to xmodel (226, user:m0916013)
condsider 240:110畢業生王人禾:Run YOLO-like Darknet Model on DPU(Vitis AI).pptx
> weight to ckpt
```
cd Vitis/Vitis-ai
./vai_run_new.sh  # open docker and select tvm
source run_conda_env.sh
/workspace/wang/github/YOLOv3_TensorFlow/
python convert_weight.py    # modify file
```
> ckpt to pb
```
cd /workspace/wang/test/
python freeze_pb.py ckpt_path
```
> pb run quantizer
```
cd /workspace/wang/vitis_tool/ 
sh vai_q_tf.sh   # modify pb_path
```
> quantizer run compile and get x-model
```
cd /workspace/wang/vitis_tool/ 
sh vai_c_tf.sh   # filename must be the same
```
![](https://i.imgur.com/TGlXRx8.png)
> inference
```
./test_video_yolov4 agilev4_moon_8classs 0 -t3
```
![](https://i.imgur.com/FLPJIqj.jpg)


### 2. c++ inference thread and queun
> 1.compile

```
cd Vitis-AI/demo/Vitis-AI-Library/samples/Agilev4/moonshoot
sh moon_build.sh
```
> 2.object detection on ZCU-102
```
cd Vitis-AI/demo/Vitis-AI-Library/samples/Agilev4/moonshoot
```
```
# LD_LIBRARY_PATH=./sort/build/ ./moon_shoot usb model gop IP PORT 

LD_LIBRARY_PATH=./sort/build/ ./moon_shoot 0 agilev4_moon_8class 5  # two stage in one
LD_LIBRARY_PATH=./sort/build/ ./moon_shoot 0 agilev4_leaky_moon 5   # moon datasets
```
## Receive information on Jetson Nano
### 1. receive data from ZCU-102
On Jetson Nano
```
cd UDP
./ros_publish
```
### 2. transfer received information into ROS format
```
cd catkin_ws/src/camera_2d_lidar_calibration/src
python3 subscrible.py
```
![](https://i.imgur.com/czyLYer.png)

ckeck
![](https://i.imgur.com/Aq4rzDY.png)

## package
> package
```
# find out where the SD card is.
# On "sdi" for example.
lsblk

# package
sudo dd if=/dev/sdi of=/home/soc507/ZCU102_Moon_VAI_1_4.img
```
![](https://i.imgur.com/O35XzRn.jpg)

> install
> 
https://www.balena.io/etcher/
![](https://i.imgur.com/kCEV7RX.png)

## github
```
soc507@soc507-PowerEdge-R740:~/github$ git clone https://github.com/fcu-soc507/FPGA-inference.git
Cloning into 'FPGA-inference'...
remote: Enumerating objects: 3, done.
remote: Counting objects: 100% (3/3), done.
remote: Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
Unpacking objects: 100% (3/3), done.
soc507@soc507-PowerEdge-R740:~/github$ git config --list
soc507@soc507-PowerEdge-R740:~/github$ git config --global user.name "fcu-soc507"
soc507@soc507-PowerEdge-R740:~/github$ git config --global user.email "fcu.soc507@gmail.com.tw"
soc507@soc507-PowerEdge-R740:~/github$ git config --list
user.name=fcu-soc507
user.email=fcu.soc507@gmail.com.tw
soc507@soc507-PowerEdge-R740:~/github$
```
## result
ZCU-102

![](https://i.imgur.com/9ELHxjX.png)

Jetson Nano

![](https://i.imgur.com/OXCsdLV.png)
