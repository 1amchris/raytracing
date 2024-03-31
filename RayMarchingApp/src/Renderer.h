#pragma once
 
#include <memory>

#include <glm/glm.hpp>
#include <Walnut/Image.h>

#include "Hardware.hpp"

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
	void Render(const Scene* scene, const Camera* camera);

	void ResetFrameIndex() { m_FrameIndex = 1; }

	std::shared_ptr<Walnut::Image> GetFinalImage() const { return m_FinalImage; }
	Settings& GetSettings() { return m_Settings; }

	~Renderer();

private:
	std::shared_ptr<Walnut::Image> m_FinalImage;
	Settings m_Settings;

	glm::vec4* m_AccumulationData = nullptr;
	
	const Camera* m_ActiveCamera = nullptr;
	const Scene* m_ActiveScene = nullptr;

	uint32_t m_FrameIndex = 1;
	float m_AspectRatio = 1;
};
