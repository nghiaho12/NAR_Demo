#include <assert.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "KeyPointThread.h"
#include "DoG.h"
#include "NAR.h"

using namespace std;

KeyPointThread::KeyPointThread()
{
    m_next_thread = NULL;
    m_use_search_region = false;
}

KeyPointThread::~KeyPointThread()
{
    Done();
}

void KeyPointThread::DoWork()
{
    boost::posix_time::ptime t1, t2;

    assert(m_next_thread);

    m_done = false;

    unsigned int total_time = 0;
    unsigned int frame_count = 0;

    while(!m_done) {
		boost::mutex::scoped_lock lock(m_base_mutex);

        if(!m_jobs.empty()) {
            ThreadJob job = m_jobs.front();
            m_jobs.pop_front();
			lock.unlock();

            t1 = boost::posix_time::microsec_clock::local_time();
            DoGKeyPointExtraction(job.grey, job.sub_pixel, job.keypoints, job.blurred, m_use_search_region, m_start, m_end);
            t2 = boost::posix_time::microsec_clock::local_time();

            m_next_thread->AddJob(job);

            //cout << m_name << ": " << job.keypoints.size() << " keypoints in " << (t2-t1).total_milliseconds() << " ms " << endl;

            total_time += (unsigned int)(t2-t1).total_milliseconds();
            frame_count++;

            if(frame_count >= 8) {
                boost::mutex::scoped_lock lock2(m_fps_mutex);
                m_fps = frame_count*1000.0f / total_time;
                frame_count = 0;
                total_time = 0;
            }
        }
        else {
            boost::thread::yield();
        }
    }
}

void KeyPointThread::SetSearchRegion(const cv::Point2i &start, const cv::Point2i &end)
{
    boost::mutex::scoped_lock lock(m_search_region_mutex);
    m_use_search_region = true;
    m_start = start;
    m_end = end;
}

void KeyPointThread::TurnOffSearchRegion()
{
    boost::mutex::scoped_lock lock(m_search_region_mutex);
    m_use_search_region = false;
}
