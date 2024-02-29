#include "Plane.h"

std::optional<const Shapes::Intersection> Shapes::Plane::Intersect(const Ray& ray) const
{
	float relativeDirections = glm::dot(Normal, ray.Direction);

	// The ray and the plane are parallel. There is no intersection
	if (relativeDirections == 0)
		return std::nullopt;

	float hitDistance = (Offset - glm::dot(Normal, ray.Origin)) / relativeDirections;

	if (hitDistance < 1e-4f)
		return std::nullopt;

	return CreatePayload(ray, hitDistance);
}

const Shapes::Intersection Shapes::Plane::CreatePayload(const Ray& ray, const float hitDistance) const
{
	Shapes::Intersection intersection{};
	intersection.ObjectIndex = ShapeIndex;
	intersection.Distance = hitDistance;
	intersection.Position = ray.Origin + ray.Direction * hitDistance;
	intersection.SurfaceNormal = Normal;
	return intersection;
}
