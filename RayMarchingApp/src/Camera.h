#pragma once

#include <vector>
#include <stdint.h>
#include <glm/glm.hpp>

class Camera {
public:
	Camera(float verticalFOV, float nearPlane, float farPlane);

	bool OnUpdate(float timeStep); /* timeStep in ms */
	void OnResize(uint32_t width, uint32_t height);
	void OnProjectionChange();

	const glm::mat4& GetProjection() const { return m_Projection; }
	const glm::mat4& GetInverseProjection() const { return m_InverseProjection; }
	const glm::mat4& GetView() const { return m_View; }
	const glm::mat4& GetInverseView() const { return m_InverseView; }

	const glm::vec3& GetPosition() const { return m_Position; }
	const glm::vec3& GetDirection() const { return m_Direction; }

	const std::vector<glm::vec3>& GetRayDirections() const { return m_RayDirections; }
	const std::vector<std::tuple<glm::vec3, glm::vec3>>& GetRayDifferentials() const { return m_RayDifferentials; }

	float GetMovementSpeed() const { return 4.0f; }
	float GetRotationSpeed() const { return 0.8f; }

public:
	float VerticalFOV = 90.0f;
	float NearPlane = 0.1f;
	float FarPlane = 100.0f;

private:
	void RecalculateProjection();
	void RecalculateView();
	void RecalculateRayDirections();

private:
	const glm::vec3 m_UpDirection { 0.0f, 1.0f, 0.0f };

	glm::mat4 m_Projection{ 1.0f };
	glm::mat4 m_InverseProjection{ 1.0f };
	glm::mat4 m_View{ 1.0f };
	glm::mat4 m_InverseView{ 1.0f };

	glm::vec3 m_Position{ 0.0f, 1.0f, 4.0f };
	glm::vec3 m_Direction = { 0.0f, 0.0f, -1.0f };

	std::vector<glm::vec3> m_RayDirections;
	std::vector<std::tuple<glm::vec3, glm::vec3>> m_RayDifferentials;
	std::vector<glm::vec3> m_RayDirectionDvs;

	glm::vec2 m_LastMousePosition{ 0.0f, 0.0f };

	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;
};