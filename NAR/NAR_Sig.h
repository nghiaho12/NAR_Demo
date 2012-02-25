#ifndef __NAR_SIG_H__
#define __NAR_SIG_H__

#include "NAR_Config.h"

#include <iostream>
#include <cstdlib>
#include <vector>
#include <opencv2/core/core.hpp>

// NOTE: x,y needs to be float for optical flow to work (assumes floats), else you get drifts
class NAR_Sig
{
public:
    int Get(const int idx) const;

public:
    float x, y;
    float orientation; // sig's orientation
    float scale; // fraction of original image size this feature was detected at, valid range (0,1)

    // TODO: get rid of this
    float orientation_diff; // orientation diff between sig and match
    int orientation_bin; // temporary working variable

    float match_x, match_y;
    unsigned char feature[FEATURE_BYTES];

/*
    // Debug
    unsigned char patch[NAR_PATCH_SQ];
    int match_idx;
    int score;
*/
};


#endif
