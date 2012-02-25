#include "CpuID.h"

#ifdef _WIN32
	#include<intrin.h>
#elif __linux__
	#define __cpuid(out, infoType)\
	asm("cpuid": "=a" (out[0]), "=b" (out[1]), "=c" (out[2]), "=d" (out[3]): "a" (infoType));
#else
    #error "Platform not supported or tested"
#endif

void GetCpuidString(CpuidString *stringStruct)
{
	int info[4];
	__cpuid(info, String);
	stringStruct->CPUInfo[0] = info[0];
	stringStruct->CPUInfo[1] = info[1];
	stringStruct->CPUInfo[2] = info[3];
	stringStruct->CPUInfo[3] = info[2];
}

void GetCpuidFeatures(CpuidFeatures *featureStruct)
{
	__cpuid(featureStruct->CPUInfo, FeatureSupport);
}
