#include "VideoThread.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace boost::posix_time;

VideoThread::VideoThread()
{
    m_frame_buffer_length = 10;
}

VideoThread::~VideoThread()
{
    Done();
}

void VideoThread::Run()
{
    m_thread = boost::thread(boost::bind(&VideoThread::DoWork, this));
}

void VideoThread::DoWork()
{
    cv::VideoCapture cap(0); // open the default camera
    //cv::VideoCapture cap("/home/nghia/Projects/NAR_Demo/video/out.avi");
	cv::Mat grey;

    if(!cap.isOpened()) {
        cerr << "Failed to initialise webcam :(" << endl;
        exit(-1);
    }

	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    m_done = false;
    m_fps = 0.0;
    m_frame_count = 0;
    m_last_time =  boost::posix_time::microsec_clock::local_time();

    m_NAR.Run();

    while(!m_done) {
        cv::Mat frame;
        cap >> frame; // get a new frame from camera

        if(frame.rows == 0 || frame.cols == 0) {
            continue;
        }

        m_NAR.SetCameraCentre(frame.cols/2, frame.rows/2);
        m_NAR.AddNewJob(frame);

        m_frame_count++;

        if(m_frame_count >= 8) {
            boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();

            time_duration::tick_type t = (now - m_last_time).total_milliseconds();

            m_fps = m_frame_count*1000.0f/t;

            m_frame_count = 0;
            m_last_time = now;
        }
    }
}

void VideoThread::Done()
{
    m_done = true;
    m_thread.join();
}

float VideoThread::GetFPS()
{
    boost::mutex::scoped_lock lock(m_mutex);
    return m_fps;
}

NAR& VideoThread::GetNAR()
{
    return m_NAR;
}
