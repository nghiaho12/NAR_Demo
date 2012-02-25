#ifndef __NAR_H__
#define __NAR_H__

#include <vector>
#include <deque>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "NAR_Sig.h"
#include "NAR_Config.h"
#include "KTree.h"
#include "AlphaBetaTracker.h"
#include "KTree.h"
#include "ThreadJob.h"
#include "KeyPointThread.h"
#include "ExtractFeatureThread.h"

class NAR : public BaseThread
{
public:
    enum StatusCode {GOOD=0, PREDICTION, BAD}; // return status per frame

    NAR();
    ~NAR();

    // YOU MUST CALL THESE TWO FUNCTIONS
    void SetARObject(const cv::Mat &AR_object); // You must call this once
    void SetCameraCentre(double cx, double cy);

    // These functions get called every video frame
    void AddNewJob(const cv::Mat &img); // avoid naming conflict from BaseThread::AddJob(...)
    std::deque <ThreadJob>& GetJobsDone();

    // All settings below have default values
    void SetCameraFOV(double fov); // horizontal degrees
    void SetSearchDepth(int depth);
    void SetRASNACThreshold(double threshold);
    void SetMinInliers(int m);
    void SetSearchRegionPadding(int padding);
    void SetMaxSigDist(int max_sig_dist);
    void SetAlphaBeta(double alpha, double beta);
    void SetMaxFailedFrames(int n);
    void SetMaxOpticalFlowTracks(int n);

    // Parameters used to learn the AR object
    void SetAngleStep(int angle_step);
    void SetYawEnd(int yaw_end);
    void SetPitchEnd(int pitch_end);
    void SetScaleFactor(float scale_factor);
    void SetNumScales(int nscales);
    void SetMaxFeatureLabels(int max_feature_labels);

    double GetOpenGLFOV() const; // For OpenGL, vertical fov, instead of horizontal

    static cv::Mat CorrectRotationForOpenGL(const cv::Mat &rot); // assumes x is right, y is up, +ve z is out of the screen,
    static void GetYPR(const cv::Mat &rotation, double &yaw, double &pitch, double &roll); // decomposes 3x3 rotation matrix

    // Debug/visual feedback functions
    size_t GetARObjectSigSizeBytes(); // returns the size of AR object signature in bytes
    void DumpMatches(); // for debugging individual matches
    std::vector <cv::Point2f>& GetOFTracks() { return m_prev_optical_flow_pts; }

    boost::mutex m_job_mutex;

private:
    virtual void DoWork();

    // FindARObject processing pipeline
    int FindARObject(const cv::Mat &grey);
    bool FeatureMatching(ThreadJob &job, std::vector <NAR_Sig> &matches);
    bool FilterOrientation(std::vector <NAR_Sig> &input, std::vector <NAR_Sig> &output, int top = 0); // filter inconsistent oriented sigs
    bool Homography(ThreadJob &job, const std::vector <NAR_Sig> &matches, cv::Mat &H);
    bool PoseEstimation(const cv::Mat &H);
    void UpdateAlphaBetaTracker(ThreadJob &job);
    void UpdateSearchRegion(ThreadJob &job);
    void UpdateOpticalFlowTracks(ThreadJob &job, const cv::Mat &H, const cv::Mat &cur_grey);

    void LearnARObject(const cv::Mat &AR_object);
    void UpdateParameters(); // sets the 3x3 camera matrix
    int FindARObject(ThreadJob &job);

    float ShortestAngle(float a, float b); // shortest angle from a to be
    void ProjectModel(double x, double y, double z, double yaw, double pitch, double roll, cv::Point2i ret_corners[4]);
    StatusCode DetectionFailed(); // called when detection has failed
    void PredictSearchRegion(cv::Point &ret_start, cv::Point &ret_end);

    int BitCount(unsigned int u);
    int HammingDistance(const unsigned char *a, const unsigned char *b);

    static cv::Mat MakeRotation3x3(double yaw, double pitch, double roll); // compose yaw pitch roll to 3x3 rotation matrix
    static cv::Mat MakeRotation4x4(float x, float y, float z); // used by WarpImage()
    static void WarpImage(const cv::Mat &in, cv::Mat &out, float yaw /* degrees */, float pitch, float roll, cv::Mat &inv_affine);

private:
    // User parameters
    double m_fov; // horizontal field of view in degrees
    double m_cx, m_cy; // camera optical centre
    double m_RANSAC_threshold;
    int m_search_depth;
    int m_min_inliers;
    int m_search_region_padding;
    int m_max_sig_dist;
    // End parameters

    // Parameters used to learn the AR object
    int m_angle_step;
    int m_yaw_end;
    int m_pitch_end;
    float m_scale_factor;
    int m_nscales;
    int m_max_feature_labels;
    int m_max_optical_flow_tracks;
    // End parameters

    // Opengl specific
    double m_focal; // calculated from m_fov and model size
    double m_vfov; // for OpenGL, vertical field of view in degrees

    // Geometry specific to camera and AR object
    int m_AR_object_width, m_AR_object_height;
    cv::Mat m_camera_intrinsics; // 3x3 camera matrix
    cv::Mat m_inv_camera_intrinsics; // 3x3 camera matrix
    cv::Mat m_AR_object_image_pts; // 4 corners, image points, never changes
    cv::Mat m_AR_oject_worlds_pts; // 4 corners, world points, constantly changing

    // From pose estimation
    cv::Mat m_AR_object_rotation;
    cv::Mat m_AR_object_translation;

    // Searching
    KTree m_ktree;
    std::vector <NAR_Sig> m_AR_object_sigs;
    cv::Point2i m_region_start, m_region_end; // limits the search space in the image

    // For smoothing out the pose estimation
    AlphaBetaTracker m_tracker;
    int m_failed_frames; // number of consecutive failed detection
    int m_max_consecutive_fails;

    // Threading
    std::deque <ThreadJob> m_jobs_done;
    KeyPointThread m_keypoint_thread[KEYPOINT_LEVELS];
    ExtractFeatureThread m_extract_feature_thread[KEYPOINT_LEVELS];

    // Optical flow assist
    std::vector <cv::Point2f> m_prev_optical_flow_pts;
    std::vector <cv::Point2f> m_prev_AR_object_pts;
    int m_optical_flow_frame_count; // increments where this significant movement
    cv::Point2f m_avg_optical_flow;
    int m_last_optical_flow_size;
    cv::Mat m_prev_grey;
};

#endif
