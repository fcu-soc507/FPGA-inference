#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -x

result=0 && pkg-config --list-all | grep opencv4 && result=1
if [ $result -eq 1 ]; then
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv4)
else
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
fi

EIGEN3_CFLAGS=$(pkg-config --cflags eigen3)

files=(
	moon_shoot_deepsort_pending.cpp
	network_sender.cpp
	src/errmsg.cpp
	src/feature/utils.cpp
	src/matching/tracker.cpp
	src/matching/track.cpp
	src/matching/kalmanfilter.cpp
	src/matching/linear_assignment.cpp
	src/matching/nn_matching.cpp
	src/thirdPart/hungarianoper.cpp
	src/thirdPart/munkres/munkres.cpp
	src/thirdPart/munkres/adapters/adapter.cpp
	src/thirdPart/munkres/adapters/boostmatrixadapter.cpp
)

CXX=${CXX:-g++}
$CXX -std=c++17 -Wall -g ${EIGEN3_CFLAGS} -I. -o moon_shoot ${files[@]} -lvitis_ai_library-yolov3 -lvitis_ai_library-dpu_task -lvitis_ai_library-xnnpp -lvitis_ai_library-model_config -lvitis_ai_library-math -lvart-util -lxir -pthread -ljson-c -lglog ${OPENCV_FLAGS} -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lxir -lvart-runner -lvitis_ai_library-graph_runner

