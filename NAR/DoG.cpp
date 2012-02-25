#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include "NAR_Config.h"
#include "DoG.h"

using namespace std;

static bool KeyPointLocalisation(const cv::Mat &grey, int x, int y, float &ret_x, float &ret_y);
static float SubPixel(const cv::Mat &grey, float x, float y);

void DoGKeyPointExtraction(const cv::Mat &grey, bool sub_pixel, std::vector <cv::Point2f> &keypoints, cv::Mat &blur,
                           bool use_search_region, const cv::Point2i &start, const cv::Point2i &end)
{
    const double sigma1 = 1.6;
    const double sigma2 = 1.6*1.6;
    const float low_contrast = 7.0f;
    const float edge_threshold = 12.1f;

    cv::Mat grey2, blur1, blur2, dog, maxima;

    grey.convertTo(grey2, CV_32F);

    cv::GaussianBlur(grey2, blur1, cv::Size(), sigma1);
    cv::GaussianBlur(grey2, blur2, cv::Size(), sigma2);
    cv::subtract(blur1, blur2, dog);

    blur1.convertTo(blur, CV_8U);

    int stride = (int)grey.step1();
    const int border = BORDER;

    for(int y=border; y < grey.rows - border; y++) {
        for(int x=border; x < grey.cols - border; x++) {
            if(use_search_region) {
                if(x < start.x) continue;
                if(y < start.y) continue;
                if(x > end.x) continue;
                if(y > end.y) continue;
            }

            float *dog_ptr = (float*)(dog.data) + y*stride + x;

            float cur = *dog_ptr;

            if(fabs(cur) < low_contrast) {
                continue;
            }

            bool extrema = false;

            // determine whether to search for minima or maxima
            if(cur < *(dog_ptr - stride - 1)) {
                if(cur < *(dog_ptr - stride)) {
                    if(cur < *(dog_ptr - stride + 1)) {
                        if(cur < *(dog_ptr - 1)) {
                            if(cur < *(dog_ptr + 1)) {
                                if(cur < *(dog_ptr + stride - 1)) {
                                    if(cur < *(dog_ptr + stride)) {
                                        if(cur < *(dog_ptr + stride + 1)) {
                                            extrema = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else {
                if(cur > *(dog_ptr - stride)) {
                    if(cur > *(dog_ptr - stride + 1)) {
                        if(cur > *(dog_ptr - 1)) {
                            if(cur > *(dog_ptr + 1)) {
                                if(cur > *(dog_ptr + stride - 1)) {
                                    if(cur > *(dog_ptr + stride)) {
                                        if(cur > *(dog_ptr + stride + 1)) {
                                            extrema = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if(extrema) {
                // eliminate edges
                float aa = *(dog_ptr) + *(dog_ptr);
                float Dxx = *(dog_ptr - 1) - aa + *(dog_ptr + 1);
                float Dyy = *(dog_ptr - stride) - aa + *(dog_ptr + stride);

                float Dx1 = *(dog_ptr - stride + 1) - *(dog_ptr - stride - 1);
                float Dx2 = *(dog_ptr + stride + 1) - *(dog_ptr + stride - 1);
                float Dxy = (Dx2 - Dx1)*0.25f;//*0.5f;

                float tr = Dxx + Dyy;
                float det = Dxx*Dyy - Dxy*Dxy;

                if(det <= 0.0f) {
                    continue;
                }

                float ratio = tr*tr / det;

                if(ratio > edge_threshold) {
                    continue;
                }

                if(sub_pixel) {
                    float ret_x, ret_y;
                    if(KeyPointLocalisation(dog, x, y, ret_x, ret_y)) {
                        keypoints.push_back(cv::Point2f(ret_x, ret_y));
                    }
                }
                else {
                    keypoints.push_back(cv::Point2f((float)x, (float)y));
                }
            }
        }
    }
}

static bool KeyPointLocalisation(const cv::Mat &grey, int x, int y, float &ret_x, float &ret_y)
{
    // keypoint localisation - I use gradient descent instead of Taylor expansion, less tedious to implement
    // the gradient descent is minimising the cost function defined as
    // cost = dx*dx + dy*dy, sum squares of the derivatives

    float cur_x = (float)x;
    float cur_y = (float)y;

    // this is for undo
    float last_x = (float)x;
    float last_y = (float)y;

    float last_cost = FLT_MAX; // resposnse
    float alpha = 0.5; // learning rate

    int iter = 0;

    while(true) {
        float dx, dy; // derivatives

        float dx1 = SubPixel(grey, cur_x - 0.1f, cur_y);
        float dx2 = SubPixel(grey, cur_x + 0.1f, cur_y);

        float dy1 = SubPixel(grey, cur_x, cur_y - 0.1f);
        float dy2 = SubPixel(grey, cur_x, cur_y + 0.1f);

        dx = dx2 - dx1;
        dy = dy2 - dy1;

        cur_x = cur_x - alpha*dx;
        cur_y = cur_y - alpha*dy;

        float cost = dx*dx + dy*dy;

        //printf("iter %d - %f %f %f, %f %f\n", iter, cost, dx, dy, cur_x, cur_y);
        // lack of significant improvement
        if(cost + 0.001 > last_cost) {
            cur_x = last_x;
            cur_y = last_y;

           // printf("last iter: %d - response = %f\n", iter, cost);
            break;
        }

        last_x = cur_x;
        last_y = cur_y;
        last_cost = cost;

        iter++;
    }

    ret_x = cur_x;
    ret_y = cur_y;

    return true;
}

static float SubPixel(const cv::Mat &grey, float x, float y)
{
    int x1 = (int)x;
    int x2 = x1 + 1;

    int y1 = (int)y;
    int y2 = y1 + 1;

    x1 = max(0,x1);
    y1 = max(0,y1);
    x2 = min(grey.cols-1,x2);
    y2 = min(grey.rows-1,y2);

    float wx2 = x - x1;
    float wx1 = 1.0f - wx2;

    float wy2 = y - y1;
    float wy1 = 1.0f - wy2;

    float a = wx1*grey.at<float>(y1,x1) + wx2*grey.at<float>(y1,x2);
    float b = wx1*grey.at<float>(y2,x1) + wx2*grey.at<float>(y2,x2);
    float ret = wy1*a + wy2*b;

    return ret;
}
