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
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vart/runner_ext.hpp>
#include <vitis/ai/yolov3.hpp>
#include <vitis/ai/demo.hpp>
#include <vitis/ai/graph_runner.hpp>

#include "network_sender.h"
#include "src/feature/utils.h"
#include "src/matching/tracker.h"

#define args_nn_budget 100
#define args_max_cosine_distance 0.2
#define args_min_confidence 0.3
#define args_nms_max_overlap 1.0

#ifndef DEST_PORT
#define DEST_PORT 8000
#endif

#ifndef DEST_IP_ADDRESS
#define DEST_IP_ADDRESS  "140.134.38.236" 
#endif

using namespace vitis::ai;

using trackq_t = BoundedQueue<TrackInfo>;

vitis::ai::FrameInfo::operator TrackInfo() const {
    TrackInfo info;
    info.frame_id = this->frame_id;
    info.is_iframe = false;
    info.mat = this->mat;
    return info;
}

TrackInfo::operator vitis::ai::FrameInfo() const {
    return FrameInfo{0, frame_id, mat};
}

static cv::Scalar getColor(int label) {
  int c[3];
  for (int i = 1, j = 0; i <= 9; i *= 3, j++) {
    c[j] = ((label / i) % 3) * 127;
  }
  return cv::Scalar(c[2], c[1], c[0]);
}

static cv::Mat process_result(cv::Mat &image,
                              const TrackInfo &result,
                              float ratio = 1.0) {
  for (const auto& bbox: result.bboxes) {
    // std::cout << sizeof(result) << std::endl;
    int label = 0; // person
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

    auto tag = "Person " + std::to_string(bbox.box_id);
    int tbase = 0;
    auto scale = cv::getFontScaleFromHeight(0, 12.0);
    auto size = cv::getTextSize(tag, 0, scale, 1, &tbase);
    cv::putText(image, tag, cv::Point(xmin, ymin-size.height + tbase), 0, scale, getColor(label));

    //cv::putText id[i]
  }
  return image;
}

class FeatureTensor {

    std::unique_ptr<vart::RunnerExt> runner_;

    FeatureTensor() : runner_(nullptr) {}

    public:

    static FeatureTensor* getInstance() { static FeatureTensor o; return &o; }

    bool load_model(const std::string& g_xmodel_file) {
        auto graph = xir::Graph::deserialize(g_xmodel_file);
        auto attrs = xir::Attrs::create();
        auto runner =
            vitis::ai::GraphRunner::create_graph_runner(graph.get(), attrs.get());
        if(runner == nullptr) { return false; }
        runner_ = std::move(runner);
        return true;
    }

    bool getRectsFeature(const cv::Mat& img, DETECTIONS& d) {
        std::vector<cv::Mat> objects;
        for(DETECTION_ROW& dbox : d) {
            cv::Rect rc = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
                    int(dbox.tlwh(2)), int(dbox.tlwh(3)));
            rc.x -= (rc.height * 0.5 - rc.width) * 0.5;
            rc.width = rc.height * 0.5;
            rc.x = (rc.x >= 0 ? rc.x : 0);
            rc.y = (rc.y >= 0 ? rc.y : 0);
            rc.width = (rc.x + rc.width <= img.cols? rc.width: (img.cols-rc.x));
            rc.height = (rc.y + rc.height <= img.rows? rc.height:(img.rows - rc.y));

            cv::Mat mattmp = img(rc).clone();
            cv::resize(mattmp, mattmp, cv::Size(64, 128));
            objects.push_back(mattmp);
        }

        auto input_tensors = runner_->get_inputs();
        auto output_tensors = runner_->get_outputs();

        auto input_tensor = input_tensors[0]->get_tensor();
        auto batch = input_tensor->get_shape().at(0);
        auto height = input_tensor->get_shape().at(1);
        auto width = input_tensor->get_shape().at(2);

        for(int i = 0; i < objects.size(); i++) {
            fill_tensor_at(input_tensors[0], i % batch, objects.at(i));
            if((i % batch) == (batch - 1)) {
                runBatch(input_tensors, output_tensors);
                dump_tensor(output_tensors[0]->get_tensor());
            }
        }

        return true;
    }

    void fill_tensor_at(vart::TensorBuffer* tensorbuffer, int index, const cv::Mat& image) {
        cv::Mat resize_image;
        auto tensor = tensorbuffer->get_tensor();
        auto size = cv::Size(tensor->get_shape().at(1), tensor->get_shape().at(2));
        if (size != image.size()) {
            cv::resize(image, resize_image, size);
        } else {
            image.copyTo(resize_image);
        }

        auto idx = tensor->get_shape();
        std::fill(idx.begin(), idx.end(), 0);
        idx[0] = (int)index;

        uint64_t data_in = 0u;
        size_t size_in = 0u;
        // 還有點問題
        std::tie(data_in, size_in) = tensorbuffer->data(idx);

        for (int h = 0; h < image.rows; h++) {
            for (int w = 0; w < image.cols; w++) {
                for (int c = 0; c < 3; c++) {
                    ((float*)data_in)[h * image.cols * 3 + w * 3 + c] = (int)(image.at<cv::Vec3b>(h, w)[c]);
                }
            }
        }
    }

    void dump_tensor(const xir::Tensor* tensor) {
        std::stringstream ss;
        ss << "(";
        for(auto v: tensor->get_shape()) {
            ss << v << ", ";
        }
        ss << ")";
        std::cout << ss.str() << std::endl;
    }

    bool runBatch(
            std::vector<vart::TensorBuffer*>& input_tensor_buffers,
            std::vector<vart::TensorBuffer*>& output_tensor_buffers
        ) {
        //sync input tensor buffers
        for (auto& input : input_tensor_buffers) {
            input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                        input->get_tensor()->get_shape()[0]);
            }

        //run graph runner
        auto v = runner_->execute_async(input_tensor_buffers, output_tensor_buffers);
        auto status = runner_->wait((int)v.first, -1);
        CHECK_EQ(status, 0) << "failed to run the graph";

        //sync output tensor buffers
        for (auto output : output_tensor_buffers) {
            output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                        output->get_tensor()->get_shape()[0]);
            }
        return true;
    }
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

        if(frame.frame_id % gop_ == 1) {
            while(!iout_->push(TrackInfo{frame}, std::chrono::milliseconds(500))) {
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
    PFrameThread(trackq_t* in, queue_t* out, trackq_t* track, int gop)
        : MyThread{}
        , in_(in)
        , out_(out)
        , track_(track)
        , gop_(gop)
    {}

    ~PFrameThread() {}

    virtual int run() override {
        TrackInfo frame;
        TrackInfo ifinfo;
        if (!track_->top(ifinfo, std::chrono::milliseconds(500))) {
            return 0;
        }
        // std::cout<< "in p frame " << std::endl;

        // std::cout<< " gop_ is " << gop_ << endl;

        if (!in_->top(frame, std::chrono::milliseconds(500))) {
            return 0;
        }

        if(ifinfo.frame_id + gop_ - 1 < frame.frame_id) {
            track_->pop(ifinfo, std::chrono::milliseconds(500));
            return 0;
        }

        in_->pop(frame, std::chrono::milliseconds(500));

        frame.mat = process_result(frame.mat, ifinfo, frame.frame_id - ifinfo.frame_id);

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
	    , tracker_(args_max_cosine_distance, args_nn_budget)
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

        // deep sort start
        DETECTIONS detections;
        for(auto box: frame.bboxes) {
            DETECTION_ROW row;
            row.tlwh = DETECTBOX{box.bbox.x,box.bbox.y,box.bbox.width,box.bbox.height};
            row.confidence = 0.8;
            detections.emplace_back(row);
        }

        std::cout << "filtering dectections" << std::endl;

		utils::dataMoreConf(args_min_confidence, detections);
		utils::dataPreprocessing(args_nms_max_overlap, detections);

		//TENSORFLOW get rect's feature.
		if(FeatureTensor::getInstance()->getRectsFeature(frame.mat, detections) == false) {
			return false;
		}

        std::cout << "tracking boxes" << std::endl;

		tracker_.predict();
		tracker_.update(detections);
        // deep sort end

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
    tracker tracker_;
    NetworkSender* sender_;
};

using namespace cv;

// Entrance of single channel video demo
int main(int argc, char* argv[]) {
    signal(SIGINT, MyThread::signal_handler);
    auto device = argv[1];
    auto model = argv[2];
    auto gop = stoi(argv[3]);

    auto sender = std::unique_ptr<NetworkSender>{ new NetworkSender(DEST_IP_ADDRESS, DEST_PORT) };

    std::cout << "loading tracking model" << std::endl;
    FeatureTensor::getInstance()->load_model("/home/root/Vitis-AI/demo/Vitis-AI-Library/samples/graph_runner/resnet50_graph_runner/xir_c/Net.xmodel");
    std::cout << "loaded tracking model" << std::endl;

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
        new PFrameThread{pframe_queue.get(), sorting_queue.get(), iftopf_trackq.get(), gop}
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
