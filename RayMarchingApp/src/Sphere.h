#pragma once

#include "Ray.h"

namespace Shapes {
	class Sphere
	{
	public:
		int ShapeIndex;
		glm::vec3 Position;
		float Radius;
	};
}