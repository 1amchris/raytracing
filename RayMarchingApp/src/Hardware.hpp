#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef __CUDACC__
#define __TARGET_CPU__ __host__
#else
#define __TARGET_CPU__
#endif

#ifdef __CUDACC__
#define __TARGET_GPU__ __device__
#else
#define __TARGET_GPU__ 
#endif

#define __TARGET_ALL__ __TARGET_CPU__ __TARGET_GPU__

// This may or may not work. Have to make sure it gets compiled independantly for the kernel calls
#ifdef __CUDA_ARCH__
#define __TARGET_IS_GPU__ true
#else
#define __TARGET_IS_GPU__ false
#endif

#ifdef __CUDA_ARCH__
#define __TARGET_IS_CPU__ false
#else
#define __TARGET_IS_CPU__ true
#endif