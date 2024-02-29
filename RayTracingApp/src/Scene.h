#pragma once

#include <vector>

#include "Material.h"
#include "Shapes/Shape.h"

#include "glm/glm.hpp"

struct Scene
{
	std::vector<Shapes::Shape*> Shapes{};
	std::vector<Material*> Materials{};

	void ResetShapeIndices() {
		for (size_t i = 0; i < Shapes.size(); i++)
			Shapes[i]->ShapeIndex = (int)i;
	}

	void Dispose() {
		for (Shapes::Shape* shape : Shapes)
			delete shape;

		for (Material* material : Materials)
			delete material;
	}
};