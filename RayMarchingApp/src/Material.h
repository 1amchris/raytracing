#pragma once

#include <glm/glm.hpp>


#pragma pack(1)
struct Material
{
	glm::vec3 Albedo{ 1.0f };
	float Roughness = 0.0f;
	float RefractivityIndex = 0.0f;
	float Opacity = 1.0f;
};