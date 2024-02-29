#include "TriangleMesh.h"

#include <iostream>

Shapes::TriangleMesh::TriangleMesh(std::vector<glm::vec3> vertices, std::vector<glm::vec3> normals, std::vector<size_t> indices):
	m_Vertices(vertices), m_Normals(normals), m_Indices(indices)
{
	// Could possibly also add a couple of checks, such as:
	//   1. m_Indices.size() > 0 [It's not a mesh without at least one triangle]
	//   2. m_Indices.size() % 3 == 0 [all triangles should have three associated vertices]
	//   3. m_Vertices.size() == m_Normals.size() [vertices and normals should have the same amount of entries]
	
	glm::vec3 min{ std::numeric_limits<float>::max() },
			  max{ std::numeric_limits<float>::min() };

	for (glm::vec3& vertex : m_Vertices)
	{
		min.x = glm::min(min.x, vertex.x);
		min.y = glm::min(min.y, vertex.y);
		min.z = glm::min(min.z, vertex.z);

		max.x = glm::max(max.x, vertex.x);
		max.y = glm::max(max.y, vertex.y);
		max.z = glm::max(max.z, vertex.z);
	}

	m_Bounds = std::shared_ptr<Bounds>{ new AxisAlignedBounds(min, max) };
}

std::optional<const Shapes::Intersection> Shapes::TriangleMesh::Intersect(const Ray& ray) const
{
	if (!m_Bounds->Intersect(ray)) {
		return std::nullopt;
	}

	return CreatePayload(ray, 10, ray.Origin + 10.0f * ray.Direction, -glm::normalize(ray.Direction));

	std::optional<Intersection> closestIntersection;
	for (size_t i = 0; i < m_Indices.size(); i += 3)
	{
		size_t indexA = m_Indices[i],
			   indexB = m_Indices[i + 1],
			   indexC = m_Indices[i + 2];

		glm::vec3 vertexA = m_Vertices[indexA],
				  vertexB = m_Vertices[indexB], 
				  vertexC = m_Vertices[indexC];

		// Note: This is a very naive way of doing it.
		// There's room for a lot of trivial optimizations.

		/* Find the plane & the intersection Q */
		glm::vec3 AB = vertexB - vertexA,
				  BC = vertexC - vertexB,
				  CA = vertexA - vertexC;
		glm::vec3 ABC = glm::cross(AB, -CA);
		glm::vec3 planeNormal = glm::normalize(ABC);
		float planeOffset = glm::dot(planeNormal, vertexA);
		float planeRayRelativeDirection = glm::dot(planeNormal, ray.Direction);

		if (planeRayRelativeDirection == 0)
			continue;

		float hitDistance = (planeOffset - glm::dot(planeNormal, ray.Origin)) / planeRayRelativeDirection;

		if (hitDistance < 1e-4f 
			|| (closestIntersection.has_value() && hitDistance >= closestIntersection->Distance))
			continue;

		glm::vec3 Q = ray.Origin + hitDistance * ray.Direction;
		glm::vec3 AQ = Q - vertexA,
				  BQ = Q - vertexB,
				  CQ = Q - vertexC;

		/* Is Q inside or outside the triangle? */
		glm::vec3 QBC = glm::cross(BC, BQ),
				  AQC = glm::cross(CA, CQ),
				  ABQ = glm::cross(AB, AQ);
		if (glm::dot(QBC, planeNormal) < 0 
			|| glm::dot(AQC, planeNormal) < 0 
			|| glm::dot(ABQ, planeNormal) < 0)
			continue;

		/* Find the barycentric coordinates */
		float TwoAreaABC = glm::length(ABC);
		float alpha = glm::length(QBC) / TwoAreaABC,
  			  beta =  glm::length(AQC) / TwoAreaABC,
			  gamma = glm::length(ABQ) / TwoAreaABC;

		glm::vec3 normal = glm::normalize(alpha * m_Normals[indexA]
										+ beta  * m_Normals[indexB]
										+ gamma * m_Normals[indexC]);

		closestIntersection = CreatePayload(ray, hitDistance, Q, normal);
	}

	return closestIntersection;
}

const Shapes::Intersection Shapes::TriangleMesh::CreatePayload(const Ray& ray, const float& hitDistance, const glm::vec3& Position, glm::vec3& Normal) const
{
	Shapes::Intersection intersection{};
	intersection.ObjectIndex = ShapeIndex;
	intersection.Distance = hitDistance;
	intersection.Position = Position;
	intersection.SurfaceNormal = Normal;
	return intersection;
}
