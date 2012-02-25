#include "ExtractFeatureThread.h"

#include <assert.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <smmintrin.h>

#include "KeyPointThread.h"
#include "DoG.h"

using namespace std;

ExtractFeatureThread::ExtractFeatureThread()
{
    for(int i=0; i < 360; i++) {
        m_sin[i] = (float)sin(TO_RAD(i));
        m_cos[i] = (float)cos(TO_RAD(i));
    }

    int k = 0;
    for(int dy=-NAR_PATCH_SIZE/2; dy < NAR_PATCH_SIZE/2; dy+=NAR_PATCH_SAMPLING) {
        for(int dx=-NAR_PATCH_SIZE/2; dx < NAR_PATCH_SIZE/2; dx+=NAR_PATCH_SAMPLING) {
            float a = atan2((float)dy,(float)dx);
            float r = sqrt((float)(dx*dx + dy*dy));

            m_angle_cache[k] = a;
            m_radius_cache[k] = r;

            k++;
        }
    }

    m_next_thread = NULL;
}

ExtractFeatureThread::~ExtractFeatureThread()
{
    Done();
}

void ExtractFeatureThread::DoWork()
{
    boost::posix_time::ptime t1, t2;

    assert(m_next_thread);

    unsigned int total_time = 0;
    unsigned int frame_count = 0;

    m_done = false;

    while(!m_done) {
		boost::mutex::scoped_lock lock(m_base_mutex);

        if(!m_jobs.empty()) {
            ThreadJob job = m_jobs.front();
            m_jobs.pop_front();
			lock.unlock();

            t1 = boost::posix_time::microsec_clock::local_time();

            vector <cv::Point2f> &kp = job.keypoints;

            for(size_t i=0; i < kp.size(); i++) {
                NAR_Sig new_feature;

                // t1 = boost::posix_time::microsec_clock::local_time();
                float orientation = CalcOrientation(job.blurred, (int)(kp[i].x+0.5f), (int)(kp[i].y+0.5)); // orientation of the FAST corner
                // t2 = boost::posix_time::microsec_clock::local_time();
                // cout << "CalcOrientation " << (t2-t1).total_microseconds() << endl;

                unsigned char patch[NAR_PATCH_SQ];

                if(GetRotatedPatch(job.blurred, (int)(kp[i].x+0.5), (int)(kp[i].y+0.5), orientation, patch)) {
                    GetPatchFeatureDescriptor(patch, new_feature.feature);

                    new_feature.x = kp[i].x;
                    new_feature.y = kp[i].y;
                    new_feature.orientation = orientation;
                    new_feature.scale = job.scale;

                    job.sigs.push_back(new_feature);
                }
            }

            if(job.scale != 1.0f) {
                float unscale = 1.0f/job.scale;

                for(size_t i=0; i < job.sigs.size(); i++) {
                    job.sigs[i].x *= unscale;
                    job.sigs[i].y *= unscale;
                }
            }

            m_next_thread->AddJob(job);

            t2 = boost::posix_time::microsec_clock::local_time();
            //cout << m_name << ": " << job.sigs.size() << " keypoints in " << (t2-t1).total_milliseconds() << " ms" << " " << endl;

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

float ExtractFeatureThread::CalcOrientation(const cv::Mat &grey, int cx, int cy)
{
    // assume a blurrd image
    const int patch_radius = NAR_PATCH_SIZE/2;
    const int radius_sq = patch_radius*patch_radius;

    int ix = 0;
    int iy = 0;
    int ixy = 0;
    int count = 0;

    for(int y=-patch_radius; y < patch_radius; y++) {
        for(int x=-patch_radius; x < patch_radius; x++) {
            int r2 = x*x + y*y;

            if(r2 < radius_sq) {
                int dx = grey.at<uchar>(cy+y, cx+x+1) - grey.at<uchar>(cy+y, cx+x-1);
                int dy = grey.at<uchar>(cy+y+1, cx+x) - grey.at<uchar>(cy+y-1, cx+x);

                ix += dx*dx;
                iy += dy*dy;
                ixy += dx*dy;
                count++;
            }
        }
    }

    // Prevents overflow
    ix /= (count<<2); // extra multiply by 4
    iy /= (count<<2);
    ixy /= (count<<2);

    int t = ix + iy;
    int d = ix*iy - ixy*ixy;

    // These 2 are slow
    float eig1 =  t*0.5f + sqrt(t*t*0.25f - (float)d);
    float angle = atan2f((float)ixy, eig1 - (float)iy);

/*
    if(isnan(angle)) {
        cout << "ix = " << ix << endl;
        cout << "iy = " << iy << endl;
        cout << "ixy = " << ixy << endl;
        cout << "d = " << d << endl;
        cout << "eig1 = " << eig1 << endl;
        cout << "t = " << t << endl;
        cout << "inside sqrt " << (t*t*0.25 - d) << endl;
        cout << "count = " << count << endl;

        exit(1);
    }
*/

    return (angle);
}

bool ExtractFeatureThread::GetRotatedPatch(const cv::Mat &grey, int x, int y, float orientation, unsigned char ret[NAR_PATCH_SQ])
{
    if(!(x >= BORDER && y >= BORDER && x < (grey.cols - BORDER) && y < (grey.rows - BORDER))) {
        return false;
    }

    for(int i=0; i < NAR_PATCH_SQ; i++) {
        float a = m_angle_cache[i] + orientation;
        float r = m_radius_cache[i];

        // This uses a lookup table
        int aidx = (int)TO_DEG(a);

        if(aidx < 0) aidx += 360;
        if(aidx >= 360) aidx = aidx - 360;

        float xf = x + r*m_cos[aidx];
        float yf = y + r*m_sin[aidx];

/*
        float xf = x + r*cos(a);
        float yf = y + r*sin(a);
*/
        float v = Bilinear(grey, xf, yf);

        ret[i] = (unsigned char)v;
    }

    return true;
}

void ExtractFeatureThread::GetPatchFeatureDescriptor(const unsigned char patch[NAR_PATCH_SQ], unsigned char ret[FEATURE_BYTES])
{
    int mean = 0;
    for(int i=0; i < NAR_PATCH_SQ; i++) {
        mean += patch[i];
    }

    mean /= NAR_PATCH_SQ;

    int k=0;
    for(int i=0; i < FEATURE_BYTES; i++) {
        int v = 0;

        for(int j=0; j < 8; j++) {
            if(patch[k] > mean) {
                v |= (1 << j);
            }

            k++;
        }

        ret[i] = v;
    }
}


float ExtractFeatureThread::Bilinear(const cv::Mat &grey, float x, float y)
{
    // Faster to do integer additions than float
    int x1 = (int)x;
    int x2 = x1 + 1;
    int y1 = (int)y;
    int y2 = y1 + 1;

    float w2 = x - x1;
    float w1 = 1.f - w2;

    float w4 = y - y1;
    float w3 = 1.f - w4;

    float grey1 = (float)grey.at<uchar>(y1,x1);
    float grey2 = (float)grey.at<uchar>(y1,x2);
    float grey3 = (float)grey.at<uchar>(y2,x1);
    float grey4 = (float)grey.at<uchar>(y2,x2);

    float result;

    // Build the weights
    __m128 a = _mm_set_ps(w1, w2, w1, w2);
    __m128 b = _mm_set_ps(w3, w3, w4, w4);
    __m128 w = _mm_mul_ps(a, b);

    __m128 g = _mm_set_ps(grey1, grey2, grey3, grey4);

    __m128 res = _mm_dp_ps(w, g, 0xff); // dot product

    _mm_store_ss(&result, res);

    return result;
}
/*
float ExtractFeatureThread::Bilinear(const cv::Mat &grey, float x, float y)
{
    int x1 = (int)x;
    int x2 = x1 + 1;
    int y1 = (int)y;
    int y2 = y1 + 1;

    float w2 = x-x1;
    float w1 = 1.f - w2;

    float w4 = y-y1;
    float w3 = 1.f - w4;

    float _w1 = w1*w3;
    float _w2 = w2*w3;
    float _w3 = w1*w4;
    float _w4 = w2*w4;

    unsigned char grey1 = grey.at<uchar>(y1,x1);
    unsigned char grey2 = grey.at<uchar>(y1,x2);
    unsigned char grey3 = grey.at<uchar>(y2,x1);
    unsigned char grey4 = grey.at<uchar>(y2,x2);

    return _w1*grey1 + _w2*grey2 + _w3*grey3 + _w4*grey4;
}
*/
