#include "BaseThread.h"
#include <iostream>

using namespace std;

BaseThread::BaseThread()
{
    m_buffer_limit = 5;
    m_fps = 0.0f;
}

void BaseThread::SetBufferLimit(unsigned int limit)
{
    m_buffer_limit = limit;
}

void BaseThread::Run()
{
    m_thread = boost::thread(boost::bind(&BaseThread::DoWork, this));
}

void BaseThread::Done()
{
    m_done = true;
    m_thread.join();
}

void BaseThread::AddJob(const ThreadJob &job)
{
    boost::mutex::scoped_lock lock(m_base_mutex);

    if(m_jobs.size() < m_buffer_limit) {
        m_jobs.push_back(job);
    }
    else {
        cout << m_name << ": buffer full" << endl;
    }
}

void BaseThread::SetNextThread(BaseThread *thread)
{
    m_next_thread = thread;
}

void BaseThread::SetName(const std::string &name)
{
    m_name = name;
}

float BaseThread::GetFPS()
{
    boost::mutex::scoped_lock lock(m_fps_mutex);
    return m_fps;
}
