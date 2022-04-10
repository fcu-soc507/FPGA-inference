/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <glog/logging.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/yolov3.hpp>
#include <vitis/ai/demo.hpp>

#include "track_info.h"
#include "./process_result.hpp"
#include "vart/runner_ext.hpp"
#include "vitis/ai/graph_runner.hpp"
#include "sort/include/tracker.h"

#include <sys/types.h>   
#include <sys/socket.h>   
#include <netinet/in.h>   
#include <arpa/inet.h> 
#include <cstddef>

#define DEST_PORT 8000   
// #define DSET_IP_ADDRESS  "127.0.0.1"
#define DEST_IP_ADDRESS  "140.134.38.236" 

// int main(int argc, char *argv[]) {
//   std::string model = argv[1];
//   std::cout << "argc = " << argc << std::endl;
//   std::cout << "argv = " << *argv << std::endl;    // 為什麼只顯示第一個?
//   return vitis::ai::main_for_video_demo(           // 有沒有return 似乎沒有關西
//       argc, argv,
//       [model] {
//         return vitis::ai::YOLOv3::create(model);
//       },process_result, 2);
// }

using namespace vitis::ai;

namespace {
    std::vector<std::string> classNames;
}

template<typename T> std::string to_bytes(T v) {
    std::string buf;
    auto pv = reinterpret_cast<char*>(&v);
    buf.resize(sizeof(v));
    std::copy(pv, pv+sizeof(T), buf.data());
    return buf;
}

// int frameInfo_to_nano(const TrackInfo& info, std::String address, std::String port)  
int frameInfo_to_nano()  
{  
    /* socket文件描述符 */  
    int sock_fd;  
    
    /* 建立udp socket */  
    sock_fd = socket(AF_INET, SOCK_DGRAM, 0);  
    if(sock_fd < 0)  
    {  
        perror("socket");  
        exit(1);  
    }  
        
    /* 设置address */  
    struct sockaddr_in addr_serv;  
    int len;  
    memset(&addr_serv, 0, sizeof(addr_serv));  
    addr_serv.sin_family = AF_INET;  
    addr_serv.sin_addr.s_addr = inet_addr(DEST_IP_ADDRESS);  
    addr_serv.sin_port = htons(DEST_PORT);  
    len = sizeof(addr_serv);  
    
        
    int send_num;  
    int recv_num;  
    char send_buf[20] = "hey, who are you?";  
    char recv_buf[20];  
        
    printf("client send: %s\n", send_buf);  
        
    send_num = sendto(sock_fd, send_buf, strlen(send_buf), 0, (struct sockaddr *)&addr_serv, len);  
        
    if(send_num < 0)  
    {  
        perror("sendto error:");  
        exit(1);  
    }  
        
    recv_num = recvfrom(sock_fd, recv_buf, sizeof(recv_buf), 0, (struct sockaddr *)&addr_serv, (socklen_t *)&len);  
        
    if(recv_num < 0)  
    {  
        perror("recvfrom error:");  
        exit(1);  
    }  
        
    recv_buf[recv_num] = '\0';  
    printf("client receive %d bytes: %s\n", recv_num, recv_buf);  
        
    close(sock_fd);  
        
    return 0;  
}

static cv::Mat process_result(cv::Mat &image,
                              const TrackInfo &result,
                              float ratio = 1.0) {
  for (const auto& bbox: result.bboxes) {
    // std::cout << sizeof(result) << std::endl;
    int label = bbox.label; // person
    float xmin = bbox.bbox.x - bbox.bbox.width / 2;
    float ymin = bbox.bbox.y - bbox.bbox.height / 2;
    float xmax = bbox.bbox.x + bbox.bbox.width / 2;
    float ymax = bbox.bbox.y + bbox.bbox.height / 2;

    if(ratio > 0) {
        xmin = bbox.bbox.x + (bbox.velocity.x*ratio) - (bbox.bbox.width + (bbox.velocity.width*ratio)) / 2;
        ymin = bbox.bbox.y + (bbox.velocity.y*ratio) - (bbox.bbox.height + (bbox.velocity.height*ratio)) / 2;
        xmax = bbox.bbox.x + (bbox.velocity.x*ratio) + (bbox.bbox.width + (bbox.velocity.width*ratio)) / 2;
        ymax = bbox.bbox.y + (bbox.velocity.y*ratio) + (bbox.bbox.height + (bbox.velocity.height*ratio)) / 2;
    }
    if (xmax > image.cols) xmax = image.cols;
    if (ymax > image.rows) ymax = image.rows;
    cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                  getColor(label), 1, 1, 0);

    std::string tag;
    if(label >= classNames.size()) { tag = "Unknown"; }
    else { tag = classNames[label]; }
    int tbase = 0;
    auto scale = cv::getFontScaleFromHeight(0, 12.0);
    auto size = cv::getTextSize(tag, 0, scale, 1, &tbase);
    cv::putText(image, tag, cv::Point(xmin, ymin-size.height + tbase), 0, scale, getColor(label));


    //cv::putText id[i]
  }
  return image;
}

class NetworkSender {
public:
    NetworkSender(const char* server_addr, int port)
    : server_addr_(server_addr)
    , port_(port)
    , sockfd(-1)
    {
        connect();
    }

    ~NetworkSender() { if(sockfd > 0) { close(sockfd); } }

    int connect() {
        struct sockaddr_in addr_serv;  
        int len;  
        memset(&addr_serv, 0, sizeof(addr_serv));  
        addr_serv.sin_family = AF_INET;  
        addr_serv.sin_addr.s_addr = inet_addr(server_addr_);  
        addr_serv.sin_port = htons(port_);
        len = sizeof(addr_serv);
        auto fd = socket(AF_INET, SOCK_DGRAM, 0);
        if(fd > 0) {
            sockfd = fd;
            return ::connect(fd, reinterpret_cast<sockaddr*>(&addr_serv), len);
        }
        return fd;
    }

    int send_trackinfo(const TrackInfo& info) {
        std::stringstream ss;
        using namespace std::chrono;
        auto tp = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch());
        ss  << to_bytes(tp.count()) // 8byte
            << to_bytes(info.is_iframe) // 1byte
            << to_bytes(info.bboxes.size()); // 8byte

        for(const auto& box: info.bboxes) {
            ss  << to_bytes(box.label)
                << to_bytes(box.bbox.x)
                << to_bytes(box.bbox.y)
                << to_bytes(box.bbox.width)
                << to_bytes(box.bbox.height)
                << to_bytes(box.box_id);
        }
        if(sockfd > 0) {
            auto buf = ss.str();
            return send(sockfd, buf.data(), buf.size(), 0);
        }
        return -1;
    }

    // int send_trackinfo(const TrackInfo& info) {
    //     std::stringstream ss;
    //     using namespace std::chrono;
    //     auto tp = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch());
    //     ss << tp.count() << info.is_iframe << info.bboxes.size();
    //     for(const auto& box: info.bboxes) {
    //         ss << box.label << box.bbox.x << box.bbox.y << box.bbox.width << box.bbox.height;
    //         ss << box.box_id;
    //     }
    //     if(sockfd > 0) {
    //         auto buf = ss.str();
    //         return send(sockfd, buf.data(), buf.size(), 0);
    //     }
    //     return -1;
    // }

private:
    const char* server_addr_;
    int port_;
    int sockfd;
};

class MuxThread : public MyThread {
public:
    MuxThread(queue_t* in, trackq_t* iout, trackq_t* pout, int gop)
        : MyThread{}
        , in_(in)
        , iout_(iout)
        , pout_(pout)
        , gop_(gop)
    {}

    ~MuxThread() {}

    virtual int run() override {
        FrameInfo frame;
        // std::cout<< frame.frame_id << std::endl;
        if (!in_->pop(frame, std::chrono::milliseconds(500))) {
            return 0;
        }
        
        std::cout << "muxer: " << frame.frame_id << "," << frame.mat.size() << std::endl;

        if(gop_ == 1 || frame.frame_id % gop_ == 1) {
            while(!iout_->push(TrackInfo{frame, true}, std::chrono::milliseconds(500))) {
                if(is_stopped()) { return -1; }
            } 
            return 0;
        } else {
            while(!pout_->push(TrackInfo{frame}, std::chrono::milliseconds(500))) {
                if(is_stopped()) { return -1; }
            }
            return 0;
        } 
    }

    virtual std::string name() override {
        return std::string{"MuxThread"};
    }

private:
    queue_t* in_;
    trackq_t* iout_;
    trackq_t* pout_;
    int gop_;
};

class PFrameThread : public MyThread {
public:
    PFrameThread(trackq_t* in, queue_t* out, trackq_t* track, int gop, NetworkSender* sender)
        : MyThread{}
        , in_(in)
        , out_(out)
        , track_(track)
        , gop_(gop)
        , sender_(sender)
    {}

    ~PFrameThread() {}

    virtual int run() override {
        TrackInfo frame;
        TrackInfo ifinfo;
        if (!track_->top(ifinfo, std::chrono::milliseconds(500))) {
            return 0;
        }
        std::cout<< "in p frame " << std::endl;

        if(gop_ == 1) {
            track_->pop(ifinfo, std::chrono::milliseconds(500));
            return 0;
        }

        // std::cout<< " gop_ is " << gop_ << endl;

        if (!in_->top(frame, std::chrono::milliseconds(500))) {
            return 0;
        }

        if(ifinfo.frame_id + gop_ - 1 < frame.frame_id) {
            track_->pop(ifinfo, std::chrono::milliseconds(500));
            return 0;
        }

        in_->pop(frame, std::chrono::milliseconds(500));

        frame.mat = process_result(frame.mat, ifinfo, static_cast<float>(frame.frame_id - ifinfo.frame_id)/gop_);

        while(!out_->push(frame, std::chrono::milliseconds(500))) {
            if(is_stopped()) {
                return -1;
            }
        }

        // std::cout<< " ======= " << endl;
        

        // while(!out_->push(frame, std::chrono::milliseconds(500))) {
        //     if(is_stopped()) {
        //         return -1;
        //     }
        // }
        return 0;
    }

    virtual std::string name() override {
        return std::string{"PFrameThread"};
    }

private:
    trackq_t* in_;
    queue_t* out_;
    trackq_t* track_;
    int gop_;
    NetworkSender* sender_;
};

template<typename T>
class IFrameDetectThread : public MyThread {
public:
    IFrameDetectThread(trackq_t* in, trackq_t* out, std::unique_ptr<T>&& model)
        : MyThread{}
        , in_(in)
        , out_(out)
        , model_(std::move(model))
    {}

    ~IFrameDetectThread() {}    

    virtual int run() override {
        TrackInfo frame;
        if (!in_->pop(frame, std::chrono::milliseconds(500))) {
            return 0;
        }
        std::cout << "iframe get: " << frame.frame_id << "," << frame.mat.size() << std::endl;
        std::cout<< "in i frame doing detector" << std::endl;
        if(!model_) { std::cout << "invalid model, wtf?" << std::endl; }
        auto res = model_->run(frame.mat);
        std::cout<< "in i frame inference done" << std::endl;
        frame.bboxes.clear();
        for(const auto& box: res.bboxes) {
            frame.bboxes.emplace_back(box);
        }

        while(!out_->push(frame, std::chrono::milliseconds(500))) {
            if(is_stopped()) {
                return -1;
            }
        }
        return 0;
    }

    virtual std::string name() override {
        return std::string{"IFrameDetectThread"};
    }

private:
    trackq_t* in_;
    trackq_t* out_;
    std::unique_ptr<T> model_;
};

class IFrameTrackingThread : public MyThread {
public:
    IFrameTrackingThread(trackq_t* in, queue_t* out, trackq_t* track, NetworkSender* sender)
        : MyThread{}
        , in_(in)
        , out_(out)
        , track_(track)
	    , tracker_()
        , sender_(sender)
    {
    }

    ~IFrameTrackingThread() {
    }

    virtual int run() override {
        TrackInfo frame;
        if (!in_->pop(frame, std::chrono::milliseconds(500))) {
            return 0;
        }

        {
            auto w = frame.mat.cols;
            auto h = frame.mat.rows;
            std::vector<cv::Rect> dets;
            std::vector<TrackBox> noop;
            while(frame.bboxes.size() > 0) {
                auto bbox = frame.bboxes.back();
                if(bbox.label == 0) {
                    dets.emplace_back(bbox.bbox.x*w, bbox.bbox.y*h, bbox.bbox.width*w, bbox.bbox.height*h);
                } else {
                    auto bw = bbox.bbox.width*w;
                    auto bh = bbox.bbox.height*h;
                    noop.emplace_back(
                        bbox.box_id,
                        bbox.label,
                        cv::Rect2f{
                            bbox.bbox.x*w + bw/2, bbox.bbox.y*h + bh/2, bw, bh
                        },
                        cv::Rect2f{
                            0,0,0,0
                        }
                    );
                }
                frame.bboxes.pop_back();
            }
            tracker_.Run(dets);

            auto tracks = tracker_.GetTracks();

            frame.bboxes = noop;

            for (auto &trk : tracks) {
                const auto &state = trk.second.GetState();
                frame.bboxes.push_back(
                    TrackBox(
                        trk.first,
                        0,
                        {
                            static_cast<float>(state[0]) - static_cast<float>(state[4]),
                            static_cast<float>(state[1]) - static_cast<float>(state[5]),
                            static_cast<float>(state[2]) - static_cast<float>(state[6]),
                            static_cast<float>(state[3]) - static_cast<float>(state[7])
                        },
                        {
                            static_cast<float>(state[4]),
                            static_cast<float>(state[5]),
                            static_cast<float>(state[6]),
                            static_cast<float>(state[7])
                        }
                    )
                );
            }
        }

        sender_->send_trackinfo(frame);

        while(!track_->push(frame, std::chrono::milliseconds(500))) {
            if(is_stopped()) {
                return -1;
            }
        }

        process_result(frame.mat, frame, 0);

        while(!out_->push(frame, std::chrono::milliseconds(500))) {
            if(is_stopped()) {
                return -1;
            }
        }
        return 0;
    }

    virtual std::string name() override {
        return std::string{"IFrameTrackingThread"};
    }

private:
    trackq_t* in_;
    queue_t* out_;
    trackq_t* track_;
    Tracker tracker_;
    NetworkSender* sender_;
};

using namespace cv;

static cv::Scalar labelColor(int label) {
  int c[3];
  for (int i = 1, j = 0; i <= 9; i *= 3, j++) {
    c[j] = ((label / i) % 3) * 127;
  }
  return cv::Scalar(c[2], c[1], c[0]);
}

void readClassNames(const std::string& modelName) {
    std::fstream f(modelName+".txt");
    if(f.bad()) { std::cout << "Cannot read class name list." << std::endl; return; }
    std::string className;
    while(!f.eof() && f.good()) {
        f >> className;
        classNames.push_back(className);
    }
}

// Entrance of single channel video demo
int main(int argc, char* argv[]) {
    signal(SIGINT, MyThread::signal_handler);
    auto device = argv[1];
    auto model = argv[2];
    auto gop = stoi(argv[3]);
    auto addr = DEST_IP_ADDRESS;
    auto port = DEST_PORT;

    if(argc > 4) { addr = argv[4]; }
    if(argc > 5) { port = atoi(argv[5]); }

    readClassNames(model);

    auto sender = std::unique_ptr<NetworkSender>{ new NetworkSender(addr, port) };

    auto channel_id = 0;
    auto decode_queue = std::unique_ptr<queue_t>{new queue_t{5}};
    auto decode_thread = std::unique_ptr<DecodeThread>(
        new DecodeThread{channel_id, device, decode_queue.get()});

    auto iframe_queue = std::unique_ptr<trackq_t>{new trackq_t{5}};
    auto pframe_queue = std::unique_ptr<trackq_t>{new trackq_t{10}};
    auto mux_thread = std::unique_ptr<MuxThread>(
        new MuxThread{decode_queue.get(), iframe_queue.get(), pframe_queue.get(), gop}
    );

    auto idectect_to_track_queue = std::unique_ptr<trackq_t>{new trackq_t{5}};
    auto iftopf_trackq = std::unique_ptr<trackq_t>{new trackq_t{5}};
    auto sorting_queue =
        std::unique_ptr<queue_t>(new queue_t(5 * 1));

    auto idetect_thread = std::unique_ptr<IFrameDetectThread<YOLOv3>>(
        new IFrameDetectThread{iframe_queue.get(), idectect_to_track_queue.get(), vitis::ai::YOLOv3::create(model)}
    );

    auto itracking_thread = std::unique_ptr<IFrameTrackingThread>(
        new IFrameTrackingThread{idectect_to_track_queue.get(), sorting_queue.get(), iftopf_trackq.get(), sender.get()}
    );

    auto pframe_thread = std::unique_ptr<PFrameThread>(
        new PFrameThread{pframe_queue.get(), sorting_queue.get(), iftopf_trackq.get(), gop, sender.get()}
    );

    auto gui_thread = GuiThread::instance();
    auto gui_queue = gui_thread->getQueue();
    auto sorting_thread = std::unique_ptr<SortingThread>(
        new SortingThread(sorting_queue.get(), gui_queue, std::to_string(0)));
    // start everything

    std::cout << "starting threads..." << std::endl;
    MyThread::start_all();
    gui_thread->wait();
    MyThread::stop_all();
    MyThread::wait_all();
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "BYEBYE";
    return 0;
}
