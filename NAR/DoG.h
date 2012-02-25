#ifndef __DOG_H__
#define __DOG_H__

#include <vector>
#include <opencv2/core/core.hpp>

void DoGKeyPointExtraction(const cv::Mat &grey, bool sub_pixel, std::vector <cv::Point2f> &keypoints, cv::Mat &blur,
                           bool use_search_region, const cv::Point2i &start, const cv::Point2i &end);

#endif
