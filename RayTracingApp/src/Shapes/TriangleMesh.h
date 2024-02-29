#pragma once

#include <vector>
#include <memory>


#include "Shape.h"

#include "../Bounds.h"
#include "../AxisAlignedBounds.h"

#include "../Ray.h"

namespace Shapes {
	class TriangleMesh : public Shape
	{
		// TODO: Add a way to move/rotate/scale the thing around

	public:
		/**
		 * vertices: A vector of positions which, together, make the whole mesh.
		 * normals: A vector of normals, where the vertices[i]'s normal is normals[i].
		 * indices: A vector of indices where each triangle of the mesh is stored.
		 *   ABC => A: (vertices/normals)[indices[0]], B: (vertices/normals)[indices[1]], C: (vertices/normals)[indices[2]]
		 *   XYZ (the i'th triangle) => X: (vertices/normals)[indices[3 * i]], Y: (vertices/normals)[indices[3 * i + 1]], Z: (vertices/normals)[indices[3 * i + 2]]
		 */
		TriangleMesh(std::vector<glm::vec3> vertices, std::vector<glm::vec3> normals, std::vector<size_t> indices);

		virtual std::optional<const Intersection> Intersect(const Ray& ray) const override;

	private:
		const Intersection CreatePayload(const Ray& ray, const float& hitDistance, const glm::vec3& Position, glm::vec3& Normal) const;

	private:
		std::vector<glm::vec3> m_Vertices;
		std::vector<glm::vec3> m_Normals;
		std::vector<size_t> m_Indices;
		std::shared_ptr<Bounds> m_Bounds;
	};
}