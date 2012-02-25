#include "KTree.h"
#include <iostream>

using namespace std;

KTree::KTree()
{
    root = NULL;
    m_id = 0;
}

KTree::~KTree()
{
    Free();
}

void KTree::Create(const vector<NAR_Sig> &sigs, int levels, int min_pts_per_node)
{
    root = new KNode();
    root->level = 0;
    root->indexes.resize(sigs.size());
    root->id = m_id++;

    for(unsigned int i=0; i < sigs.size(); i++) {
        root->indexes[i] = i;
    }

    // Non-recursive way of building trees
    vector <KNode*> toVisit;

    toVisit.push_back(root);

    while(!toVisit.empty()) {
        vector <KNode*> nextSearch;

        while(!toVisit.empty()) {
            KNode *node = toVisit.back();
            toVisit.pop_back();

            node->left = NULL;
            node->right = NULL;

            if(node->level < levels) {
                if((int)node->indexes.size() > min_pts_per_node) {
                    KNode *left = new KNode();
                    KNode *right = new KNode();

                    left->level = node->level+1;
                    right->level = node->level+1;
                    left->id = m_id++;
                    right->id = m_id++;

                    Split(sigs, node->indexes, left, right);

                    // Clear current indexes
                    {
                        vector <int> dummy;
                        node->indexes.swap(dummy);
                    }

                    node->left = left;
                    node->right = right;
                    node->_left = left->id;
                    node->_right = right->id;

                    nextSearch.push_back(left);
                    nextSearch.push_back(right);
                }
            }
        }

        toVisit = nextSearch;
    }

	cout << "Done" << endl;
}

void KTree::Split(const vector <NAR_Sig> &sigs, const vector <int> &indexes, KNode *left, KNode *right)
{
    if(indexes.size() == 0) {
        left = NULL;
        right = NULL;
    }

    left->indexes.clear();
    right->indexes.clear();

    CvMat *samples = cvCreateMat((int)indexes.size(), FEATURE_LENGTH, CV_32F);
    CvMat *labels = cvCreateMat((int)indexes.size(), 1, CV_32S);

    for(unsigned int i=0; i < indexes.size(); i++) {
        int idx = indexes[i];

        const NAR_Sig &sig = sigs[idx];

        for(int j=0; j < FEATURE_LENGTH; j++) {
            samples->data.fl[i*FEATURE_LENGTH + j] = (float)sig.Get(j);
        }
    }

    cvKMeans2(samples, 2, labels, cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 100, 0.1));

    // Calc the centres
    float centres[2][FEATURE_LENGTH];
    int count[2];

    memset(centres[0], 0, sizeof(float)*FEATURE_LENGTH);
    memset(centres[1], 0, sizeof(float)*FEATURE_LENGTH);

    count[0] = 0;
    count[1] = 0;

    KNode *nodes[2];
    nodes[0] = left;
    nodes[1] = right;

    for(unsigned int i=0; i < indexes.size(); i++) {
        int idx = labels->data.i[i];

        for(int j=0; j < FEATURE_LENGTH; j++) {
           centres[idx][j] += samples->data.fl[i*FEATURE_LENGTH + j];
        }

        nodes[idx]->indexes.push_back(indexes[i]);
        count[idx]++;
    }

    for(int j=0; j < FEATURE_LENGTH; j++) {
        centres[0][j] /= count[0];
        centres[1][j] /= count[1];
    }

    memcpy(left->centre, centres[0], sizeof(float)*FEATURE_LENGTH);
    memcpy(right->centre, centres[1], sizeof(float)*FEATURE_LENGTH);

    cvReleaseMat(&samples);
    cvReleaseMat(&labels);
}

void KTree::Search(const NAR_Sig &sig, vector <int> &indexes) const
{
    float sigF[FEATURE_LENGTH];

    for(int i=0; i < FEATURE_LENGTH; i++) {
        sigF[i] = (float)sig.Get(i);
    }

    indexes.clear();

    vector <KNode*> toVisit;
    toVisit.push_back(root);

    while(!toVisit.empty()) {
        vector <KNode*> nextSearch;

        while(!toVisit.empty()) {
            KNode *node = toVisit.back();
            toVisit.pop_back();

            // Check if this is the end node
            if(node->left != NULL) {
                float right_dist = DistanceSq(sigF, node->right->centre);
                float left_dist = DistanceSq(sigF, node->left->centre);

                if(left_dist < right_dist)
                    nextSearch.push_back(node->left);
                else
                    nextSearch.push_back(node->right);
            }
            else {
                // Must have reached the end, append indexes to output
                indexes.insert(indexes.end(), node->indexes.begin(), node->indexes.end());
            }
        }

        toVisit = nextSearch;
    }
}

void KTree::Free()
{
    if(!root)
        return;

    // Non-recursive way
    vector <KNode*> toVisit;
    vector <KNode*> toDelete;

    toVisit.push_back(root);

    while(!toVisit.empty()) {

        vector <KNode*> nextSearch;

        while(!toVisit.empty()) {
            KNode *node = toVisit.back();
            toVisit.pop_back();

            if(node->left != NULL) {
                nextSearch.push_back(node->left);
                nextSearch.push_back(node->right);
            }
        }

        toVisit = nextSearch;
        toDelete.insert(toDelete.end(), nextSearch.begin(), nextSearch.end());
    }

    for(unsigned int i=0; i < toDelete.size(); i++) {
        delete toDelete[i];
    }

    delete root;
    root = NULL;
}
