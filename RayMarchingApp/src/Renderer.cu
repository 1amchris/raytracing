#include "Renderer.h"

#include <iostream>
#include <limits>
#include <execution>
#include <optional>

#include <Walnut/Random.h>

#include <cuda/std/limits>

#include "Utils/Random.hpp"
#include "Utils/Colors.hpp"


Renderer::~Renderer() 
{
	cudaFree(m_AccumulationData);
}

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

	cudaFree(m_AccumulationData);
	cudaMallocManaged(&m_AccumulationData, width * height * sizeof(glm::vec4));
	ResetFrameIndex();
}

void Renderer::Render(const Scene& scene, const Camera& camera)
{
	m_ActiveCamera = &camera;
	m_ActiveScene = &scene;

	const uint32_t imageWidth = m_FinalImage->GetWidth();
	const uint32_t imageHeight = m_FinalImage->GetHeight();
	const size_t imageSize = imageWidth * imageHeight;

	if (m_FrameIndex == 1)
		memset(m_AccumulationData, 0, imageSize * sizeof(glm::vec4));

	uint32_t* p_resultingImage;
	cudaMallocManaged(&p_resultingImage, imageSize * sizeof(uint32_t));

	std::vector<Ray> rays;
	rays.reserve(imageSize);
	for (const glm::vec3& rayDirection : m_ActiveCamera->GetRayDirections()) {
		Ray ray{};
		ray.Origin = m_ActiveCamera->GetPosition();
		ray.Direction = rayDirection;
		rays.push_back(ray);
	}
	Ray* p_rays;
	cudaMallocManaged(&p_rays, rays.size() * sizeof(Ray));
	cudaMemcpy(p_rays, rays.data(), rays.size() * sizeof(Ray), cudaMemcpyHostToDevice);

	std::vector<Shapes::Sphere> shapes;
	shapes.reserve(m_ActiveScene->Shapes.size());
	for (const auto* shape : m_ActiveScene->Shapes) {
		shapes.push_back(*shape);
	}
	Shapes::Sphere* p_shapes;
	cudaMallocManaged(&p_shapes, shapes.size() * sizeof(Shapes::Sphere));
	cudaMemcpy(p_shapes, shapes.data(), shapes.size() * sizeof(Shapes::Sphere), cudaMemcpyHostToDevice);

	dim3 blockDim(16, 16);
	dim3 gridDim((imageWidth + blockDim.x - 1) / blockDim.x,
		(imageHeight + blockDim.y - 1) / blockDim.y);

	RayGen<<<gridDim, blockDim>>>(p_resultingImage, m_AccumulationData, m_FrameIndex, p_rays, imageWidth, imageHeight, p_shapes, shapes.size());
	cudaDeviceSynchronize();

	m_FinalImage->SetData(p_resultingImage);

	cudaFree(p_resultingImage);
	cudaFree(p_rays);
	cudaFree(p_shapes);

	if (m_Settings.Accumulate)
		++m_FrameIndex;
	else
		ResetFrameIndex();
}

__global__ 
void RayGen(uint32_t* resultingImage, glm::vec4* imageAccumulator, int frameIndex, const Ray* rays, 
						const uint32_t imageWidth, const uint32_t imageHeight,
						const Shapes::Sphere* shapes, const size_t shapesCount)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= imageWidth || y >= imageHeight) return;

	const uint32_t inlineCoord = x + y * imageWidth;

	//Ray ray = rays[inlineCoord];

	//float distanceMarched = 0.0f;
	//for (int i = 0; i < 80; ++i) 
	//{
	//	glm::vec3 position = ray.Origin + ray.Direction * distanceMarched;

	//	float distanceToClosestShape = ::cuda::std::numeric_limits<float>::max();
	//	for (int shapeIndex = 0; shapeIndex < shapesCount; ++shapeIndex)
	//	{
	//		Shapes::Sphere shape = shapes[shapeIndex];
	//		float distanceToShape = (shape.Position - position).length() - shape.Radius;

	//		if (distanceToShape < distanceToClosestShape)
	//			distanceToClosestShape = distanceToShape;
	//	}

	//	distanceMarched += distanceToClosestShape;

	//	if (distanceToClosestShape < 1e-3f || 1e3f < distanceMarched)
	//		break;
	//}

	//glm::vec3 light { distanceMarched / 1e3f };

	glm::vec3 light{ (float)x / imageWidth, (float)y / imageHeight, 0.0f };

	glm::vec4 color { light, 1.0f };
	imageAccumulator[inlineCoord] += color;
	glm::vec4 averageColor = imageAccumulator[inlineCoord] / (float)frameIndex;
	averageColor = glm::clamp(averageColor, { 0.0f }, { 1.0f });
	resultingImage[inlineCoord] = Utils::ConvertToRGBA(averageColor);
}
