#pragma once

#include "Shape.h"

#include "../Ray.h"

namespace Shapes {
	class Sphere : public Shape
	{
	public:
		virtual std::optional<const Intersection> Intersect(const Ray& ray) const override;

	public:
		glm::vec3 Position;
		float Radius;

	private:
		const Intersection CreatePayload(const Ray& ray, const float hitDistance) const;
	};
}