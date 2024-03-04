#include <iostream>
#include <cstdlib>

#include "helper.h"

#include <glm/glm.hpp>


__global__ void vectorAddition(glm::vec2* A, glm::vec2* B, glm::vec2* C, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        C[index] = A[index] + B[index];
    }
}

void add_vectors_and_print_them(int size) {
    // Allocate memory for vectors on the host
    glm::vec2* h_A = (glm::vec2*)malloc(size * sizeof(glm::vec2));
    glm::vec2* h_B = (glm::vec2*)malloc(size * sizeof(glm::vec2));
    glm::vec2* h_C = (glm::vec2*)malloc(size * sizeof(glm::vec2));

    // Initialize vectors on the host
    for (int i = 0; i < size; ++i) {
        h_A[i] = glm::vec2(static_cast<float>(i), static_cast<float>(i));
        h_B[i] = glm::vec2(static_cast<float>(2 * i), static_cast<float>(2 * i));
    }

    // Allocate memory for vectors on the device
    glm::vec2* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, size * sizeof(glm::vec2));
    cudaMalloc(&d_B, size * sizeof(glm::vec2));
    cudaMalloc(&d_C, size * sizeof(glm::vec2));

    // Copy vectors from host to device
    cudaMemcpy(d_A, h_A, size * sizeof(glm::vec2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(glm::vec2), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    vectorAddition<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size * sizeof(glm::vec2), cudaMemcpyDeviceToHost);

    // Free allocated memory on the device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Display the result
    for (int i = 0; i < 10; ++i) {
        std::cout << "(" << h_C[i].x << ", " << h_C[i].y << ") ";
    }
    std::cout << "...\n";

    // Free allocated memory on the host
    free(h_A);
    free(h_B);
    free(h_C);
}
