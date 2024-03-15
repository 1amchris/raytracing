#pragma once

#include <vector>
#include <glm/glm.hpp>

#include "Sphere.h"


struct Scene
{
	std::vector<Shapes::Sphere*> Shapes {};

	void ResetShapeIndices() {
		for (size_t i = 0; i < Shapes.size(); i++)
			Shapes[i]->ShapeIndex = (int)i;
	}

	void Dispose() {
		for (Shapes::Sphere* shape : Shapes)
			delete shape;
	}
};