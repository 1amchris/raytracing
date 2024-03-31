#pragma once

#include <vector>
#include <glm/glm.hpp>

#include "Sphere.h"
#include "Material.h"


struct Scene
{
	std::vector<Shapes::Sphere*> Shapes {};
	std::vector<Material*> Materials{};

	void ResetIndices() {
		for (size_t i = 0; i < Shapes.size(); i++)
			Shapes[i]->ShapeIndex = (int)i;
	}

	void Dispose() {
		for (Shapes::Sphere* shape : Shapes)
			delete shape;
	}
};