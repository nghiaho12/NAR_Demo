#ifndef __KEYPOINT_THREAD_H__
#define __KEYPOINT_THREAD_H__

#include <boost/thread/thread.hpp>
#include <opencv2/core/core.hpp>
#include <deque>
#include <vector>

#include "BaseThread.h"

class KeyPointThread : public BaseThread
{
public:
    KeyPointThread();
    ~KeyPointThread();

    void SetSearchRegion(const cv::Point2i &start, const cv::Point2i &end);
    void TurnOffSearchRegion();

private:
    virtual void DoWork();

    boost::mutex m_search_region_mutex;
    bool m_use_search_region;
    cv::Point2i m_start, m_end;
};

#endif
