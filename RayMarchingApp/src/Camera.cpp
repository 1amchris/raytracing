#include "Camera.h"

#include <Walnut/Input/Input.h>
#include <glm/gtx/quaternion.hpp>

Camera::Camera(float verticalFOV, float nearPlane, float farPlane)
	: m_VerticalFOV(verticalFOV), m_NearPlane(nearPlane), m_FarPlane(farPlane)
{
}

bool Camera::OnUpdate(float timeStep)
{
	glm::vec2 mousePos = Walnut::Input::GetMousePosition();
	glm::vec2 delta = (mousePos - m_LastMousePosition) * 0.002f;
	m_LastMousePosition = mousePos;

	if (!Walnut::Input::IsMouseButtonDown(Walnut::MouseButton::Right))
	{
		Walnut::Input::SetCursorMode(Walnut::CursorMode::Normal);
		return false;
	}

	Walnut::Input::SetCursorMode(Walnut::CursorMode::Locked);

	bool hasMoved = false;

	constexpr glm::vec3 upDirection(0.0f, 1.0f, 0.0f);
	glm::vec3 rightDirection = glm::cross(m_Direction, upDirection);

	float movementSpeed = GetMovementSpeed();

	// Movement
	if (Walnut::Input::IsKeyDown(Walnut::KeyCode::W))
	{
		glm::vec3 distance = m_Direction * movementSpeed * timeStep;
		if (Walnut::Input::IsKeyDown(Walnut::KeyCode::LeftControl))
			distance *= 25.0f;
		m_Position += distance;
		hasMoved = true;
	}
	if (Walnut::Input::IsKeyDown(Walnut::KeyCode::S))
	{
		glm::vec3 distance = m_Direction * movementSpeed * timeStep;
		if (Walnut::Input::IsKeyDown(Walnut::KeyCode::LeftControl))
			distance *= 25.0f;
		m_Position -= distance;
		hasMoved = true;
	}
	if (Walnut::Input::IsKeyDown(Walnut::KeyCode::D))
	{
		m_Position += rightDirection * movementSpeed * timeStep;
		hasMoved = true;
	}
	if (Walnut::Input::IsKeyDown(Walnut::KeyCode::A))
	{
		m_Position -= rightDirection * movementSpeed * timeStep;
		hasMoved = true;
	}
	if (Walnut::Input::IsKeyDown(Walnut::KeyCode::LeftShift))
	{
		m_Position -= upDirection * movementSpeed * timeStep;
		hasMoved = true;
	}
	if (Walnut::Input::IsKeyDown(Walnut::KeyCode::Space))
	{
		m_Position += upDirection * movementSpeed * timeStep;
		hasMoved = true;
	}

	// Rotation
	if (delta != glm::vec2(0.0f))
	{
		float rotationSpeed = GetRotationSpeed();
		float pitchDelta = delta.y * rotationSpeed;
		float yawDelta = delta.x * rotationSpeed;

		glm::quat q = glm::normalize(glm::cross(
			glm::angleAxis(-pitchDelta, rightDirection),
			glm::angleAxis(-yawDelta, upDirection)
		));
		m_Direction = glm::rotate(q, m_Direction);

		hasMoved = true;
	}

	if (hasMoved)
	{
		RecalculateView();
		RecalculateRayDirections();
	}

	return hasMoved;
}

void Camera::OnResize(uint32_t width, uint32_t height)
{
	if (width == m_ViewportWidth && height == m_ViewportHeight)
		return;

	m_ViewportWidth = width;
	m_ViewportHeight = height;

	RecalculateProjection();
	RecalculateRayDirections();
}

void Camera::RecalculateProjection()
{
	m_Projection = glm::perspectiveFov(
		glm::radians(m_VerticalFOV),
		(float)m_ViewportWidth,
		(float)m_ViewportHeight,
		m_NearPlane,
		m_FarPlane);
	m_InverseProjection = glm::inverse(m_Projection);
}

void Camera::RecalculateView()
{
	m_View = glm::lookAt(m_Position, m_Position + m_Direction, glm::vec3{ 0.0f, 1.0f, 0.0f });
	m_InverseView = glm::inverse(m_View);
}

void Camera::RecalculateRayDirections()
{
	m_RayDirections.resize(m_ViewportWidth * m_ViewportHeight);

	for (uint32_t y = 0; y < m_ViewportHeight; y++)
	{
		for (uint32_t x = 0; x < m_ViewportWidth; x++)
		{
			glm::vec2 coord { (float)x / (float)m_ViewportWidth, (float)y / (float)m_ViewportHeight };
			coord = coord * 2.0f - 1.0f; // [-1.0, 1.0]

			glm::vec4 target = m_InverseProjection * glm::vec4{ coord, 1.0f, 1.0f };
			glm::vec3 rayDirection = glm::vec3(m_InverseView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0.0f));
			m_RayDirections[x + y * m_ViewportWidth] = rayDirection;
		}
	}
}
