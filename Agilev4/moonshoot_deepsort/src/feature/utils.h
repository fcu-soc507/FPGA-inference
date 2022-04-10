#ifndef __UTIL_H__
#define __UTIL_H__

#include "dataType.h"

namespace utils {
void dataMoreConf(float min_confidence, DETECTIONS &d);
void dataPreprocessing(float max_bbox_overlap, DETECTIONS &d);
void _Qsort(DETECTIONS d, std::vector<int>& a, int low, int high);
};

#endif
