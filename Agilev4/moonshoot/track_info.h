#ifndef __TRACK_INFO_H__
#define __TRACK_INFO_H__

#include <vitis/ai/yolov3.hpp>
#include <vitis/ai/demo.hpp>
#include <opencv2/core.hpp>

namespace {
    using namespace vitis::ai;
    struct TrackBox_ {
        TrackBox_(const YOLOv3Result::BoundingBox& box)
        : has_track(false)
        , label(box.label)
        , bbox(box.x, box.y, box.width, box.height)
        , box_id(-1)
        , velocity() {}

        TrackBox_(const int& i, const int& l, cv::Rect2f b, cv::Rect2f v)
        : has_track(i >= 0)
        , label(l)
        , bbox(b)
        , box_id(i)
        , velocity(v) {}

        bool has_track;

        int label;
        cv::Rect2f bbox;

        // only valid when has_track is true
        int box_id;
        cv::Rect2f velocity;
    };

    struct TrackInfo_ {
        TrackInfo_() = default;
        TrackInfo_(const FrameInfo& info, bool isif = false)
        : frame_id(info.frame_id)
        , is_iframe(isif)
        , mat(info.mat) {}

        operator FrameInfo() const {
            return FrameInfo{0, frame_id, mat};
        }

        unsigned long frame_id;
        bool is_iframe;
        cv::Mat mat;
        std::vector<TrackBox_> bboxes;
    };
}

using TrackBox = TrackBox_;
using TrackInfo = TrackInfo_;
using trackq_t = vitis::ai::BoundedQueue<TrackInfo>;

#endif