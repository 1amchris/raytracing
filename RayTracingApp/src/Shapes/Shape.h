#pragma once

#include <optional>

#include "../Ray.h"

namespace Shapes {
	struct Intersection {
		glm::vec3 SurfaceNormal;
		glm::vec3 Position;
		int ObjectIndex;
		float Distance;
	};

	class Shape
	{
	public:

		virtual std::optional<const Intersection> Intersect(const Ray& ray) const = 0;

	public:
		int ShapeIndex;
		int MaterialIndex;
	};
}