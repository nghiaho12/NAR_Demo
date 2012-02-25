#ifndef __THREAD_JOB_H__
#define __THREAD_JOB_H__

#include <vector>
#include <opencv2/core/core.hpp>
#include "NAR_Sig.h"

struct ThreadJob
{
    // These variables processed along the threading pipeline
    cv::Mat img;
    cv::Mat grey;
    cv::Mat blurred;
    std::vector <cv::Point2f> keypoints;
    std::vector <NAR_Sig> sigs;
    float scale;
    unsigned int group_id;
    bool sub_pixel;

    // Everything below here is final result of the AR process, used for display
    int status; // return status of the AR process
    cv::Mat rotation, translation;
    std::vector <cv::Point2f> matches;
    bool use_search_region;
    cv::Point2i search_region_start, search_region_end;
    cv::Point2i corners[4]; // 4 corners of the AR object
    std::vector <cv::Point2f> optical_flow_tracks;
};

#endif
