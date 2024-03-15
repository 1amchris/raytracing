#pragma once

#include <cstring>
#include <memory>
#include <glm/glm.hpp>

namespace Utils
{
    class PCG
    {
    public:
        __host__ __device__ static float Float(float& seed)
        {
            uint32_t state = glm::floatBitsToUint(seed) * 747796405u + 2891336453u;
            uint32_t word = ((state >> 13u) ^ state) * 62189911u;
            uint32_t randInt = (word >> ((state >> 28u) & 3u)) | (word << (32 - ((state >> 28u) & 3u)));
            return randInt / (float)std::numeric_limits<uint32_t>::max();
        }

        __host__ __device__ static glm::vec3 Vec3(glm::vec3 seed)
        {
            return glm::vec3(Float(seed.x), Float(seed.y), Float(seed.z));
        }

        __host__ __device__ static glm::vec3 InUnitSphere(glm::vec3 seed)
        {
            return glm::normalize(Vec3(seed) * 2.0f - 1.0f);
        }
    };
}
