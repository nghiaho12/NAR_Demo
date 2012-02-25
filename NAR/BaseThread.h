#ifndef __BASE_THREAD_H__
#define __BASE_THREAD_H__

#include <boost/thread/thread.hpp>
#include <deque>
#include <string>

#include "ThreadJob.h"

class BaseThread
{
public:
    BaseThread();

    void Run();
	void Done();
    void AddJob(const ThreadJob &job);

	void SetNextThread(BaseThread *thread);
    void SetName(const std::string &name);
    void SetBufferLimit(unsigned int limit);

    virtual float GetFPS();

protected:
    virtual void DoWork() = 0;

protected:
    boost::thread m_thread;
    boost::mutex m_base_mutex;
    boost::mutex m_fps_mutex;
    bool m_done;
    std::deque <ThreadJob> m_jobs;
    BaseThread *m_next_thread;
    std::string m_name;
    unsigned int m_buffer_limit;
    float m_fps;
};

#endif
