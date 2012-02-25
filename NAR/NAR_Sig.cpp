#include "NAR_Sig.h"
#include <string.h> // for memset
#include <assert.h>

int NAR_Sig::Get(const int idx) const
{
    assert(idx < FEATURE_LENGTH);
    return (feature[idx >> 3] >> (idx % 8)) & 1;
}
