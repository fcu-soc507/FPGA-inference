
OPENCV_PKG=$(shell pkg-config --exists opencv4 && echo opencv4 || echo opencv)

CXX=g++
CFLAGS=-std=c++17 -Wall -g -I.
CFLAGS+=$(shell pkg-config --cflags ${OPENCV_PKG})
CFLAGS+=$(shell pkg-config --cflags eigen3)

LDFLAGS=-pthread -ljson-c -lglog

LDFLAGS+=-lxir \
	-lvitis_ai_library-yolov3 \
	-lvitis_ai_library-dpu_task \
	-lvitis_ai_library-xnnpp \
	-lvitis_ai_library-model_config \
	-lvitis_ai_library-math \
	-lvitis_ai_library-graph_runner \
	-lvart-util \
	-lvart-runner

LDFLAGS+=$(pkg-config --libs-only-L ${OPENCV_PKG}) \
	-lopencv_core \
	-lopencv_videoio \
	-lopencv_imgproc \
	-lopencv_imgcodecs \
	-lopencv_highgui

all: moon_shoot

clean:
	rm something.a

moon_shoot: \
	something.a \
	network_sender.o \
	moon_shoot_deepsort_pending.cpp

	${CXX} ${CFLAGS} -o moon_shoot\
		something.a \
		network_sender.o\
		moon_shoot_deepsort_pending.cpp\
		${LDFLAGS}

network_sender.o: network_sender.cpp
	${CXX} ${CFLAGS} -c network_sender.cpp

something.a: src/errmsg.cpp \
	src/feature/utils.cpp \
	src/matching/tracker.cpp \
	src/matching/track.cpp \
	src/matching/kalmanfilter.cpp \
	src/matching/linear_assignment.cpp \
	src/matching/nn_matching.cpp \
	src/thirdPart/hungarianoper.cpp \
	src/thirdPart/munkres/munkres.cpp \
	src/thirdPart/munkres/adapters/adapter.cpp \
	src/thirdPart/munkres/adapters/boostmatrixadapter.cpp

	${CXX} ${CFLAGS} -fPIC -shared -o something.a \
		src/errmsg.cpp \
		src/feature/utils.cpp \
		src/matching/tracker.cpp \
		src/matching/track.cpp \
		src/matching/kalmanfilter.cpp \
		src/matching/linear_assignment.cpp \
		src/matching/nn_matching.cpp \
		src/thirdPart/hungarianoper.cpp \
		src/thirdPart/munkres/munkres.cpp \
		src/thirdPart/munkres/adapters/adapter.cpp \
		src/thirdPart/munkres/adapters/boostmatrixadapter.cpp
