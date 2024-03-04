#pragma once

#include <cuda_runtime.h>

__global__ void vectorAddition(float* A, float* B, float* C, int size);

void add_vectors_and_print_them(int size);