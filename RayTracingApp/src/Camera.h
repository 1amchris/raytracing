#pragma once

#include <vector>
#include <stdint.h>

#include "glm/glm.hpp"

class Camera {
public:
	Camera(float verticalFOV, float nearPlane, float farPlane);

	bool OnUpdate(float timeStep); /* timeStep in ms */
	void OnResize(uint32_t width, uint32_t height);

	const glm::mat4& GetProjection() const { return m_Projection; }
	const glm::mat4& GetInverseProjection() const { return m_InverseProjection; }
	const glm::mat4& GetView() const { return m_View; }
	const glm::mat4& GetInverseView() const { return m_InverseView; }

	const glm::vec3& GetPosition() const { return m_Position; }
	const glm::vec3& GetDirection() const { return m_Direction; }

	const std::vector<glm::vec3>& GetRayDirections() const { return m_RayDirections; }

	const float GetMovementSpeed() const { return 4.0f; }
	const float GetRotationSpeed() const { return 0.8f; }

private:
	void RecalculateProjection();
	void RecalculateView();
	void RecalculateRayDirections();

private:
	glm::mat4 m_Projection{ 1.0f };
	glm::mat4 m_InverseProjection{ 1.0f };
	glm::mat4 m_View{ 1.0f };
	glm::mat4 m_InverseView{ 1.0f };

	float m_VerticalFOV = 45.0f;
	float m_NearPlane = 0.1f;
	float m_FarPlane = 100.0f;

	glm::vec3 m_Position{ 0.1f, 0.1f, 10.0f };
	glm::vec3 m_Direction = { 0.0f, 0.0f, -1.0f };

	std::vector<glm::vec3> m_RayDirections;

	glm::vec2 m_LastMousePosition{ 0.0f, 0.0f };

	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;
};