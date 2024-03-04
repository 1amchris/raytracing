#pragma once

#include <glm/glm.hpp>

__global__ void vectorAddition(glm::vec2* A, glm::vec2* B, glm::vec2* C, int size);

void add_vectors_and_print_them(int size);