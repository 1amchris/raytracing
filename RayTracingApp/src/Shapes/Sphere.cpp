#include "Sphere.h"

#include "glm/glm.hpp"

#include "../AxisAlignedBounds.h"


std::optional<const Shapes::Intersection> Shapes::Sphere::Intersect(const Ray& ray) const
{
	// idk here. It may or may not be helpful
	if (!AxisAlignedBounds(Position - Radius, Position + Radius).Intersect(ray))
		return std::nullopt;

	glm::vec3 ObjSpaceRay = ray.Origin - Position;
	float a = glm::dot(ray.Direction, ray.Direction);
	float b = 2.0f * glm::dot(ObjSpaceRay, ray.Direction);
	float c = glm::dot(ObjSpaceRay, ObjSpaceRay) - Radius * Radius;

	float discriminant = b * b - 4 * a * c;

	if (discriminant < 0)
		return std::nullopt;

	float sqrtDiscrim = glm::sqrt(discriminant);
	float denom = 2 * a;
	float t0 = (-b - sqrtDiscrim) / denom;

	if (t0 > 1e-4)
		return CreatePayload(ray, t0);

	float t1 = (-b + sqrtDiscrim) / denom;
	if (t1 > 1e-4)
		return CreatePayload(ray, t1);

	return std::nullopt;

}

const Shapes::Intersection Shapes::Sphere::CreatePayload(const Ray& ray, const float hitDistance) const
{
	Intersection intersection{};
	intersection.ObjectIndex = ShapeIndex;
	intersection.Distance = hitDistance;
	intersection.Position = ray.Origin + ray.Direction * hitDistance;
	intersection.SurfaceNormal = glm::normalize(intersection.Position - Position);
	return intersection;
}
