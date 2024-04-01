#pragma once

#include <glm/glm.hpp>


#pragma pack(1)
struct Material
{
	glm::vec3 Albedo{ 1.0f };
	float Roughness = 0.0f;
	float RefractiveIndex = -1.0f;
	// float Metallic = 0.0f;

	// glm::vec3 EmissionColor{ 0.0f };
	// float EmissionPower = 0.0f;

	// glm::vec3 GetEmission() const { return EmissionColor * EmissionPower; }
};