#ifndef __EXTRACTFEATURETHREAD_H__
#define __EXTRACTFEATURETHREAD_H__

#include <opencv2/core/core.hpp>
#include <boost/thread/thread.hpp>


#include "NAR_Sig.h"
#include "BaseThread.h"

class NAR;

class ExtractFeatureThread : public BaseThread
{
public:
    ~ExtractFeatureThread();
    ExtractFeatureThread();

    static float CalcOrientation(const cv::Mat &grey, int cx, int cy);
    static void GetPatchFeatureDescriptor(const unsigned char patch[NAR_PATCH_SQ], unsigned char ret[FEATURE_BYTES]);
    static float Bilinear(const cv::Mat &grey, float x, float y);
    bool GetRotatedPatch(const cv::Mat &grey, int x, int y, float orientation, unsigned char ret[NAR_PATCH_SQ]);

private:
    virtual void DoWork();

private:
    // Pre-computed values
    float m_sin[360];
    float m_cos[360];
    float m_angle_cache[NAR_PATCH_SQ];
    float m_radius_cache[NAR_PATCH_SQ];
};

#endif
