#include "Renderer.h"

#include <iostream>
#include <limits>
#include <execution>

#include <Walnut/Random.h>
#include "Utils/Random.hpp"
#include "Utils/Colors.hpp"

#define USE_FAST_RANDOM 1

void Renderer::OnResize(uint32_t width, uint32_t height)
{
	if (m_FinalImage)
	{
		if (m_FinalImage->GetWidth() == width && m_FinalImage->GetHeight() == height) 
			return;

		m_FinalImage->Resize(width, height);
	}
	else
	{
		m_FinalImage = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA);
	}

	m_AspectRatio = (float)width / (float)height;

	delete[] m_ImageData;
	m_ImageData = new uint32_t[width * height];

	delete[] m_AccumulationData;
	m_AccumulationData = new glm::vec4[width * height];
	ResetFrameIndex();

	m_ImageHorizontalIter.resize(width);
	for (uint32_t i = 0; i < width; i++)
		m_ImageHorizontalIter[i] = i;

	m_ImageVerticalIter.resize(height);
	for (uint32_t i = 0; i < height; i++)
		m_ImageVerticalIter[i] = i;
}

void Renderer::Render(const Scene& scene, const Camera& camera)
{
	m_ActiveCamera = &camera;
	m_ActiveScene = &scene;

	if (m_FrameIndex == 1)
		memset(m_AccumulationData, 0, 
			m_FinalImage->GetWidth() 
			* m_FinalImage->GetHeight() 
			* sizeof(glm::vec4));

	std::for_each(std::execution::par, m_ImageVerticalIter.begin(), m_ImageVerticalIter.end(),
		[this](uint32_t y)
		{
			for (uint32_t x = 0; x < m_FinalImage->GetWidth(); x++)
			{
				glm::vec4 color = RayGen(x, y);
				const int inlineCoord = x + y * m_FinalImage->GetWidth();
				m_AccumulationData[inlineCoord] += color;

				glm::vec4 averageColor = m_AccumulationData[inlineCoord] / (float)m_FrameIndex;
				averageColor = glm::clamp(averageColor, { 0.0f }, { 1.0f });
				m_ImageData[inlineCoord] = Utils::ConvertToRGBA(averageColor);
			}
		});

	m_FinalImage->SetData(m_ImageData);

	if (m_Settings.Accumulate)
		m_FrameIndex++;
	else
		ResetFrameIndex();
}

glm::vec4 Renderer::RayGen(const uint32_t x, const uint32_t y)
{
	uint32_t inlineCoord = x + y * m_FinalImage->GetWidth();
	Ray ray{};
	ray.Origin = m_ActiveCamera->GetPosition();
	ray.Direction = m_ActiveCamera->GetRayDirections()[inlineCoord];

	glm::vec3 light{ 0.0f };
	glm::vec3 contribution{ 1.0f };
	constexpr int MAXIMUM_BOUNCES = 5;
	constexpr glm::vec3 SKY_COLOR{ 0.6f, 0.7f, 1.0f };
	for (int i = 0; i < MAXIMUM_BOUNCES; i++)
	{
		std::optional<Shapes::Intersection> intersection = TraceRay(ray);

		if (!intersection.has_value()) 
		{
			light += SKY_COLOR * contribution;
			break;
		}

		const Shapes::Shape* closestShape = m_ActiveScene->Shapes[intersection->ObjectIndex];
		const Material* material = m_ActiveScene->Materials[closestShape->MaterialIndex];

		contribution *= material->Albedo;
		light += material->GetEmission() * contribution;

#if USE_FAST_RANDOM
		glm::vec3 seed = (light + intersection->Position) * glm::uintBitsToFloat(inlineCoord * m_FrameIndex);
		glm::vec3 normalRoughness = material->Roughness * Utils::PCG::InUnitSphere(seed);
#else
		glm::vec3 normalRoughness = material->Roughness * Walnut::Random::InUnitSphere();
#endif
		glm::vec3 normal = glm::normalize(intersection->SurfaceNormal + normalRoughness);
		ray.Direction = glm::reflect(ray.Direction, normal);
		ray.Origin = intersection->Position + intersection->SurfaceNormal * 1e-4f;
	}

	return { light, 1.0f };
}

std::optional<const Shapes::Intersection> Renderer::TraceRay(const Ray& ray) const
{
	std::optional<Shapes::Intersection> closestIntersection;

	for (size_t i = 0; i < m_ActiveScene->Shapes.size(); i++)
	{
		const Shapes::Shape* shape = m_ActiveScene->Shapes[i];
		std::optional<Shapes::Intersection> intersection = shape->Intersect(ray);

		if (intersection.has_value() 
			&& (!closestIntersection.has_value() 
				|| intersection->Distance < closestIntersection->Distance)) {
			closestIntersection = intersection;
		}
	}

	return closestIntersection;
}
