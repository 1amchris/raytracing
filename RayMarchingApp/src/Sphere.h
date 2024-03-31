#pragma once

#include "Ray.h"

namespace Shapes {
	#pragma pack(1)
	class Sphere
	{
	public:
		int ShapeIndex;
		int MaterialIndex;
		glm::vec3 Position;
		float Radius;
	};
}