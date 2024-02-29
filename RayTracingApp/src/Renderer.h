#pragma once
 
#include <memory>

#include "glm/glm.hpp"
#include "Walnut/Image.h"

#include "Camera.h"
#include "Ray.h"
#include "Scene.h"

class Renderer {
public:
	struct Settings
	{
		bool Accumulate = true;
	};

	void OnResize(uint32_t width, uint32_t height);
	void Render(const Scene& scene, const Camera& camera);

	void ResetFrameIndex() { m_FrameIndex = 1; }

	std::shared_ptr<Walnut::Image> GetFinalImage() const { return m_FinalImage; }
	Settings& GetSettings() { return m_Settings; }

private:
	glm::vec4 RayGen(const uint32_t x, const uint32_t y);

	std::optional<const Shapes::Intersection> TraceRay(const Ray& ray) const;

private:
	std::shared_ptr<Walnut::Image> m_FinalImage;
	Settings m_Settings;

	std::vector<uint32_t> m_ImageHorizontalIter, m_ImageVerticalIter;

	uint32_t* m_ImageData = nullptr;
	glm::vec4* m_AccumulationData = nullptr;
	
	const Camera* m_ActiveCamera = nullptr;
	const Scene* m_ActiveScene = nullptr;

	uint32_t m_FrameIndex = 1;
	float m_AspectRatio = 1;
};