#ifndef __KNODE__
#define __KNODE__

/*
A simplified Hierarchical K-Mean tree.
Searches only goes down the tree, never back up.
This works in practice provided the tree is not too deep (around 6 level or less)

TODO: change code to use new C++ syntax
*/

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>

#include "NAR_Sig.h"
#include "NAR_Config.h"

// Node in K-Tree
class KNode
{
public:
    KNode()
    {
        level = -1;

        left = NULL;
        right = NULL;

        _left = -1;
        _right = -1;
    }

    float centre[FEATURE_LENGTH];
    int level;
    std::vector <int> indexes;
    std::vector <int> tmpIndexes;

    KNode *left;
    KNode *right;

    int id; // for GPU
    int _left, _right; // for GPU
};

// A point in the K-Tree node
struct KTreePoint
{
    float coords[FEATURE_LENGTH];
    unsigned char sig[FEATURE_BYTES];
};
class KTree
{
public:
    KTree();
    ~KTree();

    void Create(const std::vector <NAR_Sig> &sigs, int levels, int min_pts_per_node = 50);
    void Search(const NAR_Sig &sig, std::vector <int> &indexes) const;
    KNode* GetRoot() const { return root; }
    int GetNumNodes() const { return m_id; }

private:
    void Free();
    void Split(const std::vector <NAR_Sig> &sigs, const std::vector <int> &indexes, KNode *left, KNode *right);

    inline float DistanceSq(const float a[FEATURE_LENGTH], const float b[FEATURE_LENGTH]) const;

private:
    KNode *root;
    int m_id;
};

inline float KTree::DistanceSq(const float a[FEATURE_LENGTH], const float b[FEATURE_LENGTH]) const
{
    __m128 sum = _mm_setzero_ps(); // zero out the memory

    // process 4 floats at a time
    for(int i=0; i < FEATURE_LENGTH; i+=4) {
        __m128 aa = _mm_loadu_ps(&a[i]); // load
        __m128 bb = _mm_loadu_ps(&b[i]); // load
        __m128 cc = _mm_sub_ps(aa, bb); // subtraction
        cc = _mm_mul_ps(cc, cc); // square
        sum = _mm_add_ps(sum, cc); // accumulation
    }

    float res[4];

    _mm_storeu_ps(res, sum);

    return res[0] + res[1] + res[2] + res[3];

/*
    float ret = 0;
    for(int i=0; i < SIG_SIZE; i++) {
        ret += (a[i] - b[i])*(a[i] - b[i]);
    }


    return ret;
*/
}

#endif
