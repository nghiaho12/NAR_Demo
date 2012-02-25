#ifndef __VIDEOTHREAD_H__
#define __VIDEOTHREAD_H__

#include <deque>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "NAR/NAR.h"

class VideoThread
{
public:
    ~VideoThread();
    VideoThread();

	void Run();
	void Done();
    float GetFPS();
    NAR& GetNAR();

private:
    void DoWork();

private:
    boost::thread m_thread;
    boost::mutex m_mutex;
    bool m_done;
    float m_fps;
    int m_frame_count;
    boost::posix_time::ptime m_last_time;
    int m_frame_buffer_length;

    NAR m_NAR;
};

#endif
