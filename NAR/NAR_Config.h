#ifndef __NAR_CONFIG__
#define __NAR_CONFIG__

#define USE_SSE4 // use SSE4.1 instruction. You need to have a 64bit compiler. if you don't have it comment this out.

const int NAR_PATCH_SIZE  = 32; // NxN pixel patch
const int NAR_PATCH_SAMPLING = 4; // sample within the patch by skipping every I pixels

// Calc from above
const int NAR_PATCH_SQ = (NAR_PATCH_SIZE*NAR_PATCH_SIZE)/(NAR_PATCH_SAMPLING*NAR_PATCH_SAMPLING); // effectice patch size after sampling
const int FEATURE_LENGTH = NAR_PATCH_SQ; // max length in RandomPairs.h
const int FEATURE_BYTES = FEATURE_LENGTH/8; // only supports multiples of 64 to work correctly!!

const float KEYPOINT_SCALE_FACTOR = 0.75f;
const int KEYPOINT_LEVELS = 3; // pyramid levels -> (image_size / pow(KEYPOINT_SCALE_FACTOR, KEYPOINT_LEVELS))

const int BORDER = NAR_PATCH_SIZE;

// Misc stuff
#define TO_RAD(x) (x*0.0174532925199433)
#define TO_DEG(x) (x*57.2957795130823)

#endif
