#pragma once

#include "Shape.h"

namespace Shapes {
	class Plane : public Shape
	{
	public:
		Plane(glm::vec3 dir1, glm::vec3 dir2, float offset) :
			Normal(glm::normalize(glm::cross(dir1, dir2))), Offset(offset)
		{
		}

		Plane(glm::vec3 normal, float offset) :
			Normal(normal), Offset(offset)
		{
		}

		virtual std::optional<const Intersection> Intersect(const Ray& ray) const override;

	public:
		glm::vec3 Normal;
		float Offset;

	private:
		const Intersection CreatePayload(const Ray& ray, const float hitDistance) const;

	};
}