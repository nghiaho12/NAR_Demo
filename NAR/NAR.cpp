#include <numeric>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <sstream>
#include <nmmintrin.h>

#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "NAR.h"
#include "RPP.h"
#include "DoG.h"
#include "KeyPointThread.h"
#include "ExtractFeatureThread.h"
#include "CpuID.h"

using namespace std;
using namespace boost::posix_time;

NAR::NAR()
{
#ifdef USE_SSE4
    CpuidFeatures features;

    GetCpuidFeatures(&features);

    if(!features.SSE41) {
        cerr << "Your CPU does not support SSE4.1. Edit NAR/NAR_Config.h and comment out #define USE_SSE4" << endl;
        exit(-1);
    }
#endif

    // Determine automatically on startup
    m_cx = -1;
    m_cy = -1;

    // Default settings
	{
		m_fov = 60.0;
		m_RANSAC_threshold = 4.0;
		m_search_depth = 6;
		m_max_sig_dist = FEATURE_LENGTH * 2/10; // used to threshold good/bad matches
		m_min_inliers = 10;
		m_search_region_padding = 0;
		m_failed_frames = 0;
		m_max_consecutive_fails = 3;
        m_max_optical_flow_tracks = 100;

		SetAlphaBeta(0.25, 0.25);
	}

	// Default AR object learning values
	{
        m_angle_step = 10;
        m_yaw_end = 60;
        m_pitch_end = 60;
        m_scale_factor = 0.5f;
        m_nscales = 3;
        m_max_feature_labels = 500;
	}

    SetName("NAR Thread");
    SetBufferLimit(30);

    for(int i=0; i < KEYPOINT_LEVELS; i++) {
		stringstream str;

		str << "KeyPoint thread " << i;
        m_keypoint_thread[i].SetName(str.str());
        m_keypoint_thread[i].SetNextThread(&m_extract_feature_thread[i]);

		str.str("");
		str << "ExtractFeature thread " << i;
        m_extract_feature_thread[i].SetName(str.str());
        m_extract_feature_thread[i].SetNextThread(this);

        m_keypoint_thread[i].Run();
        m_extract_feature_thread[i].Run();
    }

    m_optical_flow_frame_count = 0;
}

NAR::~NAR()
{
    Done();
}

void NAR::SetARObject(const cv::Mat &AR_object)
{
    m_AR_object_width = AR_object.cols;
    m_AR_object_height = AR_object.rows;

    // Find the largest dimension needed for creating temporary warped images
    int size = (int)sqrt(m_AR_object_width*m_AR_object_width*0.25 + m_AR_object_height*m_AR_object_height*0.25)*2;
    size = (size/4)*4 + 4;

    cv::Mat AR_object_grey;
    cv::cvtColor(AR_object, AR_object_grey, CV_BGR2GRAY);

    // Initialse other matrices
    m_AR_object_image_pts = cv::Mat::zeros(3,4,CV_64F);

    // 4 corner points of the model
    // top left
    m_AR_object_image_pts.at<double>(0,0) = 0;
    m_AR_object_image_pts.at<double>(1,0) = 0;
    m_AR_object_image_pts.at<double>(2,0) = 1;

    // top right
    m_AR_object_image_pts.at<double>(0,1) = m_AR_object_width;
    m_AR_object_image_pts.at<double>(1,1) = 0;
    m_AR_object_image_pts.at<double>(2,1) = 1;

    // bottom left
    m_AR_object_image_pts.at<double>(0,2) = m_AR_object_width;
    m_AR_object_image_pts.at<double>(1,2) = m_AR_object_height;
    m_AR_object_image_pts.at<double>(2,2) = 1;

    // bottom right
    m_AR_object_image_pts.at<double>(0,3) = 0;
    m_AR_object_image_pts.at<double>(1,3) = m_AR_object_height;
    m_AR_object_image_pts.at<double>(2,3) = 1;

    double scale = 1.0 / (double)m_AR_object_width;
    cv::Mat normalise_2D_mat = cv::Mat::eye(3,3,CV_64F);
    normalise_2D_mat.at<double>(0,0) = scale;
    normalise_2D_mat.at<double>(1,1) = scale;
    normalise_2D_mat.at<double>(0,2) = -scale*m_AR_object_width*0.5;
    normalise_2D_mat.at<double>(1,2) = -scale*m_AR_object_height*0.5;

    m_AR_oject_worlds_pts = normalise_2D_mat*m_AR_object_image_pts;

    for(int i=0; i < m_AR_oject_worlds_pts.cols; i++) {
        m_AR_oject_worlds_pts.at<double>(2,i) = 0.0; // z value
    }

	LearnARObject(AR_object_grey);
}

void NAR::DumpMatches()
{
    /*
    cv::Mat out = cv::Mat::zeros(cv::Size(600,600), CV_8U);

    for(unsigned int i=0; i < m_filtered_matches.size(); i++) {
        int patch_y = i/5;
        int patch_x = i - patch_y*5;
        int idx = m_filtered_matches[i].match_idx;

        assert(idx < m_AR_object_sigs.size());

        for(int y=0; y < NAR_PATCH_SIZE; y++) {
            for(int x=0; x < NAR_PATCH_SIZE; x++) {
                int xx = patch_x*(NAR_PATCH_SIZE*3) + x;
                int yy = patch_y*(NAR_PATCH_SIZE + 8) + y;

                out.at<uchar>(yy,xx) = m_filtered_matches[i].patch[y*NAR_PATCH_SIZE + x];
                out.at<uchar>(yy,xx + NAR_PATCH_SIZE + 8) = m_AR_object_sigs[idx].patch[y*NAR_PATCH_SIZE + x];
            }
        }
    }

    cv::imwrite("debug.png", out);
    */

}

bool NAR::FeatureMatching(ThreadJob &job, vector <NAR_Sig> &matches)
{
    boost::posix_time::ptime t1, t2;

    // first process called by FindARObject
    t1 = boost::posix_time::microsec_clock::local_time();

    vector <int> indexes; // index points to m_marker_sig
    vector <float> dists;

    // Using K-Tree
    indexes.resize(job.sigs.size());
    dists.resize(job.sigs.size());

    for(size_t i=0; i < job.sigs.size(); i++) {
        vector <int> ret_indexes;
        m_ktree.Search(job.sigs[i], ret_indexes);

        int best_dist = FEATURE_LENGTH;
        int best_idx = 0;

        for(size_t j=0; j < ret_indexes.size(); j++) {
            int idx = ret_indexes[j];

            int d = HammingDistance(job.sigs[i].feature, m_AR_object_sigs[idx].feature);

            if(d < best_dist) {
                best_dist = d;
                best_idx = ret_indexes[j];
            }
        }

        indexes[i] = best_idx;
        dists[i] = (float)best_dist;
    }

    // We'll get sigs with duplicate (x,y) (but diff pose)
    // Keep the best matching one only
    vector <float> score_map(m_AR_object_width*m_AR_object_height/4, FLT_MAX);
    vector <int> query_index_map(m_AR_object_width*m_AR_object_height/4, -1);
    vector <int> model_sig_index_map(m_AR_object_width*m_AR_object_height/4, -1); // using down size map

    int half_width = m_AR_object_width/2;
    int half_height = m_AR_object_height/2;

    for(size_t i=0; i < job.sigs.size(); i++) {
        int idx = indexes[i];

        int x = (int)(m_AR_object_sigs[idx].x/2 + 0.5f);
        int y = (int)(m_AR_object_sigs[idx].y/2 + 0.5f);

        x = min(x, half_width-1);
        y = min(y, half_height-1);

        x = max(x, 0);
        y = max(y, 0);

        if(dists[i] < score_map[y*half_width + x]) {
            score_map[y*half_width + x] = dists[i];
            query_index_map[y*half_width+ x] = (int)i;
            model_sig_index_map[y*half_width + x] = idx;
        }
    }

    for(size_t i=0; i < query_index_map.size(); i++) {
        int query_idx = query_index_map[i];

        if(query_idx != -1 && score_map[i] < m_max_sig_dist) {
            int model_sig_idx = model_sig_index_map[i];

            float a = job.sigs[query_idx].orientation;
            float b = m_AR_object_sigs[model_sig_idx].orientation;

            job.sigs[query_idx].orientation_diff = ShortestAngle(a,b);
            job.sigs[query_idx].match_x = m_AR_object_sigs[model_sig_idx].x;
            job.sigs[query_idx].match_y = m_AR_object_sigs[model_sig_idx].y;
            //job.sigs[query_idx].match_idx = model_sig_idx;
            //job.sigs[query_idx].score = score_map[i];

            matches.push_back(job.sigs[query_idx]);
        }
    }

    t2 = boost::posix_time::microsec_clock::local_time();

    cout << "Feature matching: " << ((t2-t1).total_milliseconds()) << " ms, total: " << job.sigs.size() << endl;

    if((int)matches.size() < m_min_inliers) {
        return false;
    }

    return true;
}

bool NAR::Homography(ThreadJob &job, const std::vector <NAR_Sig> &matches, cv::Mat &H)
{
    boost::posix_time::ptime t1, t2;
    vector <cv::Point2f> src2, dst2;
    const cv::Mat &cur_grey = job.blurred;

    t1 = boost::posix_time::microsec_clock::local_time();

    for(size_t i=0; i < matches.size(); i++) {
        src2.push_back(cv::Point2f(matches[i].match_x, matches[i].match_y));
        dst2.push_back(cv::Point2f(matches[i].x, matches[i].y));
    }

    // If we have optical flow vectors, attempt to track them and add them to the list of points for homography
    // The more points the better the homography!
    int OF_index_offset = (int)src2.size();

    if(!m_prev_optical_flow_pts.empty()) {
        vector <cv::Point2f> cur_pts, cur_pts2;
        vector <uchar> of_status;
        vector <float> of_err;

        cv::calcOpticalFlowPyrLK(m_prev_grey, cur_grey, m_prev_optical_flow_pts, cur_pts, of_status, of_err, cv::Size(21,21), 3);

        m_avg_optical_flow.x = 0;
        m_avg_optical_flow.y = 0;

        for(size_t i=0; i < cur_pts.size(); i++) {
            if(of_status[i]) {
                src2.push_back(m_prev_AR_object_pts[i]);
                dst2.push_back(cur_pts[i]);

                cur_pts2.push_back(cur_pts[i]);

                m_avg_optical_flow.x += (cur_pts[i].x - m_prev_optical_flow_pts[i].x);
                m_avg_optical_flow.y += (cur_pts[i].y - m_prev_optical_flow_pts[i].y);
            }
        }

        m_avg_optical_flow.x /= cur_pts2.size();
        m_avg_optical_flow.y /= cur_pts2.size();

        m_prev_optical_flow_pts = cur_pts2;
    }

    int best_inliers;
    vector <char> inlier_mask;
    vector <uchar> mask2;

    H = cv::findHomography(src2, dst2, CV_RANSAC, m_RANSAC_threshold, mask2);

    best_inliers = accumulate(mask2.begin(), mask2.end(), 0);
    t2 = boost::posix_time::microsec_clock::local_time();

    // Move back up later
    if(best_inliers < m_min_inliers) {
        cout << "Not enough inliers: " << best_inliers << endl;
        cout << "Homography: " << job.matches.size() << " - " <<  (t2-t1).total_milliseconds() << " ms" << endl;
        return false;
    }

    // Do least-square fit
    vector <cv::Point2f> src3, dst3;;
    vector <cv::Point2f> optical_flow_pts, AR_object_pts;

    vector <NAR_Sig> filtered;
    for(size_t i=0; i < mask2.size(); i++) {
        if(mask2[i] == 0) {
            continue;
        }

        src3.push_back(src2[i]);
        dst3.push_back(dst2[i]);

        if((int)i >= OF_index_offset) {
            optical_flow_pts.push_back(src2[i]);
            AR_object_pts.push_back(dst2[i]);
        }
    }

    H = cv::findHomography(src3, dst3, 0);

    job.matches = dst3;

    t2 = boost::posix_time::microsec_clock::local_time();

    cout << "Homography: " << job.matches.size() << " - " <<  (t2-t1).total_milliseconds() << " ms" << endl;

    return true;
}

bool NAR::PoseEstimation(const cv::Mat &H)
{
    boost::posix_time::ptime t1, t2;

    t1 = boost::posix_time::microsec_clock::local_time();

    // H * m_AR_object_image_pts = 4 corner points
    // I use the 4 corners instead of ALL the points.
    // Why do I do this? It's faster, but maybe less accurate?
    cv::Mat image_pts = m_inv_camera_intrinsics * H * m_AR_object_image_pts;

    double obj_err, img_err;
    int it;

    bool status = RPP::Rpp(m_AR_oject_worlds_pts, image_pts, m_AR_object_rotation, m_AR_object_translation, it, obj_err, img_err);

    if(!status) {
        return false;
    }

    t2 = boost::posix_time::microsec_clock::local_time();

    cout << "RPP: " << (t2-t1).total_milliseconds() << " ms" << endl;

    return true;
}

void NAR::UpdateAlphaBetaTracker(ThreadJob &job)
{
    double yaw = 0, pitch = 0, roll = 0;
    double x, y, z;

    x = m_AR_object_translation.at<double>(0,0);
    y = m_AR_object_translation.at<double>(1,0);
    z = m_AR_object_translation.at<double>(2,0);

    GetYPR(m_AR_object_rotation, yaw, pitch, roll);

    m_tracker.SetState(x, y, z, yaw, pitch, roll);

    if(m_tracker.Ready()) {
        m_tracker.GetCorrectedState(&x, &y, &z, &yaw, &pitch, &roll);

        m_AR_object_rotation = MakeRotation3x3(yaw, pitch, roll);
        m_AR_object_translation.at<double>(0,0) = x;
        m_AR_object_translation.at<double>(1,0) = y;
        m_AR_object_translation.at<double>(2,0) = z;

        ProjectModel(x, y, z, yaw, pitch, roll, job.corners);
    }

    job.rotation = m_AR_object_rotation;
    job.translation = m_AR_object_translation;
}

void NAR::UpdateSearchRegion(ThreadJob &job)
{
    cv::Point region_start;
    cv::Point region_end;

    if(m_tracker.Ready() && m_failed_frames < m_max_consecutive_fails) {
        job.use_search_region = true;

        PredictSearchRegion(region_start, region_end);

        job.search_region_start = region_start;
        job.search_region_end = region_end;

        for(int i=0; i < KEYPOINT_LEVELS; i++) {
            float scale = pow(KEYPOINT_SCALE_FACTOR, (float)i);

            m_keypoint_thread[i].SetSearchRegion(region_start*scale, region_end*scale);
        }
    }
    else {
        job.use_search_region = false;

        for(int i=0; i < KEYPOINT_LEVELS; i++) {
            m_keypoint_thread[i].TurnOffSearchRegion();
        }
    }
}

void NAR::UpdateOpticalFlowTracks(ThreadJob &job, const cv::Mat &H, const cv::Mat &cur_grey)
{
    if(m_optical_flow_frame_count >= 16) {
        m_prev_optical_flow_pts.clear();

        m_optical_flow_frame_count = 0;
        m_avg_optical_flow.x = 0;
        m_avg_optical_flow.y = 0;
        m_last_optical_flow_size = 0;
    }
    else if(m_prev_optical_flow_pts.empty()) {
        cv::Mat mask2 = cv::Mat::zeros(job.img.size(), CV_8U);
        cv::rectangle(mask2, job.search_region_start, job.search_region_end, cv::Scalar(255), CV_FILLED);
        cv::goodFeaturesToTrack(cur_grey, m_prev_optical_flow_pts, m_max_optical_flow_tracks, 0.04, 8.0, mask2);

        cout << "OF tracks " << m_prev_optical_flow_pts.size() << endl;

        // Reverse transform
        cv::Mat inv = H.inv();
        cv::Mat X(3,1,CV_64F);
        cv::Mat X2(3,1,CV_64F);
        X.at<double>(2,0) = 1.0;

        m_prev_AR_object_pts.resize(m_prev_optical_flow_pts.size());

        for(size_t i=0; i < m_prev_optical_flow_pts.size(); i++) {
            X.at<double>(0,0) = m_prev_optical_flow_pts[i].x;
            X.at<double>(1,0) = m_prev_optical_flow_pts[i].y;
            X.at<double>(2,0) = 1.0;

            X2 = inv*X;

            m_prev_AR_object_pts[i].x = (float)(X2.at<double>(0,0) / X2.at<double>(2,0));
            m_prev_AR_object_pts[i].y = (float)(X2.at<double>(1,0) / X2.at<double>(2,0));
        }

        m_optical_flow_frame_count = 0;
        m_avg_optical_flow.x = 0;
        m_avg_optical_flow.y = 0;
        m_last_optical_flow_size = (int)m_prev_optical_flow_pts.size();
    }

    if(fabs(m_avg_optical_flow.x) + fabs(m_avg_optical_flow.y) > 20) { // significant movement
        cout << "     Significant movement" << endl;
        m_optical_flow_frame_count++;
    }
    else if(m_prev_optical_flow_pts.size() < m_last_optical_flow_size*7/10) { // we've lost too many optical flow tracks
        cout << "     Lost too many optical flow tracks" << endl;
        m_optical_flow_frame_count++;
    }

    job.optical_flow_tracks = m_prev_optical_flow_pts;
}

int NAR::FindARObject(ThreadJob &job)
{
    if(m_AR_object_sigs.empty()) {
        cerr << "You forgot to call SetARObject()" << endl;
        assert(m_AR_object_sigs.empty());
    }

    if((int)job.sigs.size() < m_min_inliers) {
        return BAD;
    }

    if(m_cx == -1 || m_cy == -1) {
        cerr << "camera centre not set" << endl;
        assert(0);
    }

    boost::posix_time::ptime t1, t2;
    cv::Mat H(3,3,CV_64F);
    vector <NAR_Sig> matches;
    cv::Mat &cur_grey = job.blurred;

    cout << endl;

    // goto for the win! :)
    if(!FeatureMatching(job, matches)) goto fail;
    if(!FilterOrientation(matches, matches, 1)) goto fail;
    if(!Homography(job, matches, H)) goto fail;
    if(!PoseEstimation(H)) goto fail;
    UpdateAlphaBetaTracker(job);
    UpdateSearchRegion(job);
    UpdateOpticalFlowTracks(job, H, cur_grey);

    m_prev_grey = cur_grey;
    m_failed_frames = 0;
    cout << "Found match!" << endl;

    return NAR::GOOD;

fail:
    int status = DetectionFailed();
    return status;
}

NAR::StatusCode NAR::DetectionFailed()
{
    m_failed_frames++;
    m_prev_optical_flow_pts.clear();
    m_prev_AR_object_pts.clear();

    // For now we'll just return the same pose when the frame has failed to detect the model
    // Prediction doesn't work very well in the presence of motion blur
    if(m_failed_frames < m_max_consecutive_fails) {
        /*
        double yaw, pitch, roll;
        double x, y, z;

        m_tracker.GetCorrectedState(&x, &y, &z, &yaw, &pitch, &roll);

        m_AR_object_rotation = YPR(yaw, pitch, roll);
        m_AR_object_translation.at<double>(0,0) = x;
        m_AR_object_translation.at<double>(1,0) = y;
        m_AR_object_translation.at<double>(2,0) = z;
        */

        return PREDICTION;
    }

    // reset search regions
    for(int i=0; i < KEYPOINT_LEVELS; i++) {
        m_keypoint_thread[i].TurnOffSearchRegion();
    }

    m_tracker.Reset();

    return BAD;
}

void NAR::PredictSearchRegion(cv::Point &ret_start, cv::Point &ret_end)
{
    cv::Mat X = m_AR_object_rotation*m_AR_oject_worlds_pts; // 3x3 * 3x4

    for(int i=0; i < X.cols; i++) {
        X.at<double>(0,i) += m_AR_object_translation.at<double>(0,0);
        X.at<double>(1,i) += m_AR_object_translation.at<double>(1,0);
        X.at<double>(2,i) += m_AR_object_translation.at<double>(2,0);
    }

    X = m_camera_intrinsics*X;

    double _min[2] = {DBL_MAX, DBL_MAX};
    double _max[2] = {-DBL_MAX, -DBL_MAX};

    for(int i=0; i < X.cols; i++) {
        X.at<double>(0,i) /= X.at<double>(2,i);
        X.at<double>(1,i) /= X.at<double>(2,i);

        _min[0] = min(_min[0], X.at<double>(0,i));
        _min[1] = min(_min[1], X.at<double>(1,i));

        _max[0] = max(_max[0], X.at<double>(0,i));
        _max[1] = max(_max[1], X.at<double>(1,i));
    }

    ret_start.x = (int)_min[0];
    ret_start.y = (int)_min[1];
    ret_end.x = (int)_max[0];
    ret_end.y = (int)_max[1];

    ret_start.x -= m_search_region_padding;
    ret_start.y -= m_search_region_padding;

    ret_end.x += m_search_region_padding;
    ret_end.y += m_search_region_padding;

    m_region_start = ret_start;
    m_region_end = ret_end;
}

bool NAR::FilterOrientation(vector <NAR_Sig> &input, vector <NAR_Sig> &output, int top)
{
    assert(top > 0);

	if(input.size() == 0) {
		return false;
	}

    // Modifies the orientation_bin in input

    const int angle_per_bin = 10;
    const int total_bins = 360 / angle_per_bin;

    vector <int> orientation_hist(total_bins,0);
    vector <NAR_Sig> filtered;

    // Find the most dominant orientation
    for(size_t i=0; i < input.size(); i++) {
        double d = input[i].orientation_diff;

        int bin = (int)(TO_DEG(d) / angle_per_bin + 0.5);

		if(bin >= total_bins) {
			bin = total_bins - 1;
		}

        input[i].orientation_bin = bin;

        orientation_hist[bin]++;
    }

    vector < pair<int,int> > angle_count; // first = count, second = bin

    for(int i=0; i < total_bins; i++) {
        angle_count.push_back(pair<int,int>(orientation_hist[i],i));
    }

    sort(angle_count.begin(), angle_count.end(), greater< pair<int,int> >());

    for(size_t i=0; i < input.size(); i++) {
        for(int j=0; j < top; j++) {
            if(input[i].orientation_bin == angle_count[j].second) {
                filtered.push_back(input[i]);
                break;
            }
        }
    }

    filtered.swap(output);

    cout << "FilterOrientation: " <<  output.size() << " matches" << endl;

    if((int)output.size() < m_min_inliers) {
        return false;
    }

    return true;
}

float NAR::ShortestAngle(float a, float b)
{
    float d = b - a;

    if(d < 0) {
        d += (float)(2.0f*M_PI);
    }

    if(d > 2*M_PI) {
        d -= (float)(2.0f*M_PI);
    }

    return d;
}

void NAR::GetYPR(const cv::Mat &rot, double &yaw, double &pitch, double &roll)
{
    // http://planning.cs.uiuc.edu/node103.html
    yaw = atan2(rot.at<double>(1,0), rot.at<double>(0,0));
    pitch = atan2(-rot.at<double>(2,0),sqrt(rot.at<double>(2,1)*rot.at<double>(2,1) + rot.at<double>(2,2)*rot.at<double>(2,2)));
    roll = atan2(rot.at<double>(2,1), rot.at<double>(2,2));
}

cv::Mat NAR::MakeRotation3x3(double yaw, double pitch, double roll)
{
    cv::Mat rot(3,3,CV_64F);

    double sina = sin(yaw);
    double cosa = cos(yaw);

    double sinb = sin(pitch);
    double cosb = cos(pitch);

    double sinc = sin(roll);
    double cosc = cos(roll);

    rot.at<double>(0,0) = cosa*cosb;
    rot.at<double>(0,1) = cosa*sinb*sinc - sina*cosc;
    rot.at<double>(0,2) = cosa*sinb*cosc + sina*sinc;

    rot.at<double>(1,0) = sina*cosb;
    rot.at<double>(1,1) = sina*sinb*sinc + cosa*cosc;
    rot.at<double>(1,2) = sina*sinb*cosc -cosa*sinc;

    rot.at<double>(2,0) = -sinb;
    rot.at<double>(2,1) = cosb*sinc;
    rot.at<double>(2,2) = cosb*cosc;

    return rot;
}

cv::Mat NAR::CorrectRotationForOpenGL(const cv::Mat &rotation)
{
    double yaw, pitch, roll;

    GetYPR(rotation, yaw, pitch, roll);

    return MakeRotation3x3(yaw, -pitch, -roll);
}

void NAR::ProjectModel(double x, double y, double z, double yaw, double pitch, double roll, cv::Point2i ret_corners[4])
{
    cv::Mat rot = MakeRotation3x3(yaw, pitch, roll);

    cv::Mat model_pts = rot*m_AR_oject_worlds_pts;

    for(int i=0; i < 4; i++) {
        model_pts.at<double>(0,i) += x;
        model_pts.at<double>(1,i) += y;
        model_pts.at<double>(2,i) += z;
    }

    cv::Mat pts = m_camera_intrinsics*model_pts;

    for(int i=0; i < 4; i++) {
        ret_corners[i].x = (int)(pts.at<double>(0,i) / pts.at<double>(2,i) + 0.5);
        ret_corners[i].y = (int)(pts.at<double>(1,i) / pts.at<double>(2,i) + 0.5);
    }
}

void NAR::UpdateParameters()
{
    m_focal = m_cx/tan(m_fov*0.5*M_PI/180.0);

    m_camera_intrinsics = cv::Mat::eye(3,3,CV_64F);

    m_camera_intrinsics.at<double>(0,0) = m_focal;
    m_camera_intrinsics.at<double>(1,1) = m_focal;
    m_camera_intrinsics.at<double>(0,2) = m_cx;
    m_camera_intrinsics.at<double>(1,2) = m_cy;

    m_inv_camera_intrinsics = m_camera_intrinsics.inv();

    m_vfov = atan(m_cy/m_focal)*180.0/M_PI*2.0;
}

double NAR::GetOpenGLFOV() const
{
    return m_vfov;
}

void NAR::SetSearchDepth(int num)
{
    m_search_depth = num;
}

void NAR::SetRASNACThreshold(double threshold)
{
    m_RANSAC_threshold = threshold;
}

void NAR::SetCameraFOV(double fov)
{
    m_fov = fov;
}

void NAR::SetCameraCentre(double cx, double cy)
{
    m_cx = cx;
    m_cy = cy;
    UpdateParameters();
}

void NAR::SetSearchRegionPadding(int padding)
{
    m_search_region_padding = padding;
}

void NAR::SetMinInliers(int m)
{
    m_min_inliers = m;
}

void NAR::SetMaxSigDist(int max_sig_dist)
{
    m_max_sig_dist = max_sig_dist;
}

void NAR::SetAlphaBeta(double alpha, double beta)
{
    m_tracker.SetAlphaBeta(alpha, beta);
}

void NAR::SetMaxFailedFrames(int n)
{
    m_failed_frames = n;
}

void NAR::SetMaxOpticalFlowTracks(int n)
{
    m_max_optical_flow_tracks = n;
}

void NAR::SetAngleStep(int angle_step)
{
    m_angle_step = angle_step;
}

void NAR::SetYawEnd(int yaw_end)
{
    m_yaw_end = yaw_end;
}

void NAR::SetPitchEnd(int pitch_end)
{
    m_pitch_end = pitch_end;
}

void NAR::SetScaleFactor(float scale_factor)
{
    m_scale_factor = scale_factor;
}

void NAR::SetNumScales(int nscales)
{
    m_nscales = nscales;
}

void NAR::SetMaxFeatureLabels(int max_feature_labels)
{
    m_max_feature_labels = max_feature_labels;
}

int NAR::BitCount(unsigned int u)
{
    // http://tekpool.wordpress.com/2006/09/25/bit-count-parallel-counting-mit-hakmem/
    unsigned int uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
    return((uCount + (uCount >> 3)) & 030707070707) % 63;
}

int NAR::HammingDistance(const unsigned char *a, const unsigned char *b)
{
#ifdef USE_SSE4
    uint64_t *aa = (uint64_t *)a;
    uint64_t *bb = (uint64_t *)b;
    //return  _mm_popcnt_u64((*aa) ^ (*bb));

    uint64_t ret = 0;

    for(unsigned int i=0; i < FEATURE_LENGTH/(sizeof(uint64_t)*8); i++) {
        ret += _mm_popcnt_u64(aa[i] ^ bb[i]);
    }

    return (int)ret;
#else
    int dist = 0;

    int *aa = (int*)a;
    int *bb = (int*)b;

    dist += BitCount(aa[0] ^ bb[0]);
    dist += BitCount(aa[1] ^ bb[1]);

    return dist;
#endif
}

void NAR::WarpImage(const cv::Mat &in, cv::Mat &out, float yaw, float pitch, float roll, cv::Mat &inv_affine)
{
    cv::Mat to_centre = cv::Mat::eye(4,4,CV_32F);
    cv::Mat to_origin = cv::Mat::eye(4,4,CV_32F);

    to_origin.at<float>(0,3) = (float)(-in.cols/2);
    to_origin.at<float>(1,3) = (float)(-in.rows/2);

    to_centre.at<float>(0,3) = (float)(in.cols/2);
    to_centre.at<float>(1,3) = (float)(in.rows/2);

    cv::Mat R = MakeRotation4x4((float)TO_RAD(pitch), (float)TO_RAD(yaw), (float)TO_RAD(roll));
    cv::Mat T = to_centre*R*to_origin;

    // We let the 3D point to have z=0
    cv::Mat affine(2,3,CV_32F);

    affine.at<float>(0,0) = T.at<float>(0,0);
    affine.at<float>(0,1) = T.at<float>(0,1);
    affine.at<float>(0,2) = T.at<float>(0,3);
    affine.at<float>(1,0) = T.at<float>(1,0);
    affine.at<float>(1,1) = T.at<float>(1,1);
    affine.at<float>(1,2) = T.at<float>(1,3);

    inv_affine = cv::Mat::eye(3,3,CV_32F);
    cv::Mat sub = inv_affine(cv::Rect(0,0,3,2));
    affine.copyTo(sub);

    inv_affine = inv_affine.inv();
    inv_affine = inv_affine(cv::Rect(0,0,3,2));

    cv::warpAffine(in, out, affine, cv::Size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(128));
}

void NAR::LearnARObject(const cv::Mat &AR_object)
{
    assert(AR_object.type() == CV_8U);

    const int grid_size = 2;

    int total_poses = m_nscales * (m_yaw_end/m_angle_step+1) * (m_pitch_end/m_angle_step+1);

    cv::Mat X(3,1,CV_32F);
    cv::Mat X2(2,1,CV_32F);
    cv::Mat resized, warped, inv_affine;
    cv::Mat blurred;
    unsigned char patch[NAR_PATCH_SQ];

    X.at<float>(2,0) = 1.0f;

    // Increase AR_object borders
    cv::Mat big = cv::Mat(AR_object.size()*2, CV_8U);
    cv::rectangle(big, cv::Point(0,0), cv::Point(big.cols-1, big.rows-1), CV_RGB(128,128,128), CV_FILLED);

    int off_x = (big.cols - AR_object.cols) / 2;
    int off_y = (big.rows - AR_object.rows) / 2;

    cv::Mat sub = big(cv::Rect(off_x, off_y, AR_object.cols, AR_object.rows));
    AR_object.copyTo(sub);

    int grid_width = AR_object.cols / grid_size;
    int grid_height = AR_object.rows / grid_size;

    vector <int> feature_count(grid_width*grid_height, 0);
    vector <int> feature_label(grid_width*grid_height, -1);
    vector < vector<NAR_Sig> > label_features;

    int current_feature_label = 0;
    int pose_count = 0;

    cout << "Learning AR object ..." << endl;

    for(int scale=0; scale < m_nscales; scale++) {
        float s = powf(m_scale_factor, (float)scale);

        int w = (int)(big.cols*s);
        int h = (int)(big.rows*s);

        cout << "Temporary image size: " << w << "x" << h << endl;

        cv::resize(big, resized, cv::Size(w,h));

        for(int yaw=0; yaw <= m_yaw_end; yaw += m_angle_step) {
            for(int pitch=0; pitch <= m_pitch_end; pitch += m_angle_step) {
                pose_count++;
                cout << "Learning pose " << pose_count << "/" << total_poses << endl;

                WarpImage(resized, warped, (float)yaw, (float)pitch, 0.0f, inv_affine);

                vector <cv::Point2f> keypoints;
                DoGKeyPointExtraction(warped, true, keypoints, blurred, false, cv::Point2i(), cv::Point2i());

                for(size_t i=0; i < keypoints.size(); i++) {
                    // Do sub-pixel using cornerScore function somewhere here
                    float x = keypoints[i].x;
                    float y = keypoints[i].y;

                    X.at<float>(0,0) = x;
                    X.at<float>(1,0) = y;

                    X2 = inv_affine*X;

                    int orig_x = (int)(X2.at<float>(0,0)*s - off_x + 0.5f);
                    int orig_y = (int)(X2.at<float>(1,0)*s - off_y + 0.5f);

                    if(orig_x < 0 || orig_x >= AR_object.cols || orig_y < 0 || orig_y >= AR_object.rows) {
                        continue;
                    }

                    int idx = (orig_y/grid_size)*grid_width + orig_x/grid_size;
                    int label = feature_label[idx];

                    if(label == -1) {
                        label = current_feature_label;
                        feature_label[idx] = label;

                        current_feature_label++;

                        label_features.resize(current_feature_label);
                    }

                    feature_count[idx]++;

                    float orientation = ExtractFeatureThread::CalcOrientation(blurred, (int)(x+0.5f), (int)(y+0.5f));

                    // 2 possible orientation
                    for(int o=0; o < 2; o++) {
                        if(o == 1) {
                            orientation += (float)M_PI;
                        }

                        if(orientation > 2.0*M_PI) {
                            orientation -= (float)(2.0*M_PI);
                        }

                        if(!m_extract_feature_thread[0].GetRotatedPatch(blurred, (int)(x+0.5f), (int)(y+0.5f), orientation, patch)) {
                            continue;
                        }

                        NAR_Sig new_feature;

                        new_feature.x = (float)orig_x;
                        new_feature.y = (float)orig_y;
                        new_feature.orientation = orientation;
                        //memcpy(new_feature.patch, patch, NAR_PATCH_SQ);

                        ExtractFeatureThread::GetPatchFeatureDescriptor(patch, new_feature.feature);

                        label_features[label].push_back(new_feature);
                    }
                }
            }
        }
    }

    vector < pair<int,int> > count_label;

    for(size_t i=0; i < feature_count.size(); i++) {
        if(feature_label[i] == -1) {
            continue;
        }

        count_label.push_back(pair<int,int>(feature_count[i], feature_label[i]));
    }

    sort(count_label.begin(), count_label.end(), greater <pair<int,int> >());

    m_AR_object_sigs.clear();
    int i=0;
    for(i=0; i < m_max_feature_labels && i < (int)count_label.size(); i++) {
        int label = count_label[i].second;

        //cout << i << " features " << label << " " << " seen " << count_label[i].first << " times" << endl;
        m_AR_object_sigs.insert(m_AR_object_sigs.end(), label_features[label].begin(), label_features[label].end());
    }

    cout << "Total feature labels " << count_label.size() << endl;
    cout << "Keeping feature labels " << i << endl;
    cout << "Total features kept " << m_AR_object_sigs.size() << endl;

    // Build the KTree
    m_ktree.Create(m_AR_object_sigs, m_search_depth);
}

cv::Mat NAR::MakeRotation4x4(float x, float y, float z)
{
    cv::Mat X = cv::Mat::eye(4,4,CV_32F);
    cv::Mat Y = cv::Mat::eye(4,4,CV_32F);
    cv::Mat Z = cv::Mat::eye(4,4,CV_32F);

    float sinx = sinf(x);
    float siny = sinf(y);
    float sinz = sinf(z);

    float cosx = cosf(x);
    float cosy = cosf(y);
    float cosz = cosf(z);

    X.at<float>(1,1) = cosx;
    X.at<float>(1,2) = -sinx;
    X.at<float>(2,1) = sinx;
    X.at<float>(2,2) = cosx;

    Y.at<float>(0,0) = cosy;
    Y.at<float>(0,2) = siny;
    Y.at<float>(2,0) = -siny;
    Y.at<float>(2,2) = cosy;

    Z.at<float>(0,0) = cosz;
    Z.at<float>(0,1) = -sinz;
    Z.at<float>(1,0) = sinz;
    Z.at<float>(1,1) = cosz;

	cv::Mat R = Z*Y*X;

	return R;
}

void NAR::AddNewJob(const cv::Mat &img)
{
    static unsigned int group_id = 0;

    cv::Mat grey;

    if(img.channels() == 3) {
        cv::cvtColor(img, grey, CV_BGR2GRAY);
    }
    else if(img.channels() == 1) {
        grey = img;
    }
    else {
        cerr << "Invalid image channel: " << img.channels() << endl;
        exit(-1);
    }

    ThreadJob new_job;

    new_job.img = img;
    new_job.group_id = group_id;
    group_id++;

    for(int i=0; i < KEYPOINT_LEVELS; i++) {
        float scale = 1.0f;

        if(i == 0) {
            new_job.grey = grey;
            new_job.sub_pixel = false;
        }
        else {
            scale = pow(KEYPOINT_SCALE_FACTOR, (float)i);
            cv::resize(grey, new_job.grey, cv::Size((int)(grey.rows*scale), (int)(grey.cols*scale)));
            new_job.sub_pixel = true;
        }

        new_job.scale = scale;

        m_keypoint_thread[i].AddJob(new_job);
    }
}

void NAR::DoWork()
{
    boost::posix_time::ptime t1, t2;
    map<int, vector<ThreadJob> > buffer;
    map<int, vector<ThreadJob> >::iterator iter;

    unsigned int total_time = 0;
    unsigned int frame_count = 0;
    m_done = false;

    while(!m_done) {
        boost::mutex::scoped_lock lock(m_base_mutex);

        while(!m_jobs.empty()) {
            ThreadJob &job = m_jobs.front();
            buffer[job.group_id].push_back(job);
            m_jobs.pop_front();
        }

        lock.unlock();

        if(!buffer.empty()) {
            vector <int> to_erase;
            unsigned int current_group_id = 0;
            bool group_id_processed = false;

            for(iter = buffer.begin(); iter != buffer.end(); ++iter) {
                vector <ThreadJob> &jobs = iter->second;

                // Got all the necessary jobs
                if((int)jobs.size() == KEYPOINT_LEVELS) {
                    t1 = boost::posix_time::microsec_clock::local_time();

                    ThreadJob job_done;

                    job_done.img = jobs[0].img;

                    current_group_id = jobs[0].group_id;
                    group_id_processed = true;

                    for(size_t i=0; i < jobs.size(); i++) {
                        job_done.sigs.insert(job_done.sigs.end(), jobs[i].sigs.begin(), jobs[i].sigs.end());

                        // Pass the full size blurred grey image
                        if(jobs[i].blurred.size() == jobs[i].img.size()) {
                            job_done.blurred = jobs[i].blurred;
                        }
                    }

                    // Find the object
                    {
                        boost::posix_time::ptime start, end;
                        start = boost::posix_time::microsec_clock::local_time();

                        job_done.status = FindARObject(job_done);

                        end = boost::posix_time::microsec_clock::local_time();

                        cout << "FindARObject: " << (end-start).total_milliseconds() << " ms" << endl;
                    }

                    boost::mutex::scoped_lock lock2(m_job_mutex);
                    m_jobs_done.push_back(job_done);
                    lock2.unlock();

                    t2 = boost::posix_time::microsec_clock::local_time();

                    total_time += (unsigned int)((t2-t1).total_milliseconds());
                    frame_count++;

                    if(frame_count >= 8) {
                        boost::mutex::scoped_lock lock3(m_fps_mutex);
                        m_fps = frame_count*1000.0f / total_time;

                        // The effective fps is the fps of the slowest thread
                        float min_fps = m_fps;

                        for(int i=0; i < KEYPOINT_LEVELS; i++) {
                            if(m_keypoint_thread[i].GetFPS() < min_fps) {
                                min_fps = m_keypoint_thread[i].GetFPS();
                            }

                            if(m_extract_feature_thread[i].GetFPS() < min_fps) {
                                min_fps = m_extract_feature_thread[i].GetFPS();
                            }
                        }

                        m_fps = min_fps;

                        lock3.unlock();
                        total_time = 0;
                        frame_count = 0;
                    }

                    to_erase.push_back(jobs[0].group_id);
                }
            }

            // Delete older group_id than current, if any.
            // This can happen if the buffer is full in any of the threads.
            // Resulting in *zombie* group_id that will never get prcoessed.
            if(group_id_processed) {
                for(iter = buffer.begin(); iter != buffer.end(); ++iter) {
                    unsigned int id = iter->first;
                    if(id < current_group_id) {
                       // cout << "ZOMBIE ID FOUND! " << id << endl;
                        to_erase.push_back(id);
                    }
                }
            }

            for(size_t i=0; i < to_erase.size(); i++) {
                buffer.erase(to_erase[i]);
            }
        }
        else {
            boost::thread::yield();
        }
    }
}

std::deque <ThreadJob>& NAR::GetJobsDone()
{
    return m_jobs_done;
}

size_t NAR::GetARObjectSigSizeBytes()
{
    return m_AR_object_sigs.size() * sizeof(NAR_Sig);
}
