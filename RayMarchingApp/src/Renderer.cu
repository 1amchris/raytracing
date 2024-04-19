#include "Renderer.h"

#include <iostream>
#include <limits>
#include <execution>
#include <optional>
#include <functional>

#define _USE_MATH_DEFINES
#include <cmath>

#include <Walnut/Random.h>

#include <cuda/std/limits>
#include <curand.h>
#include <curand_kernel.h>

#include "Random.hpp"
#include "Colors.hpp"
#include "Interval.hpp"
#include "Hardware.hpp"


#define MAX_STEP_COUNT 2500
#define MAX_BOUNCE_COUNT 10
#define MIN_DISTANCE_TO_COLLIDE 1e-4f

class ClosestEntity {
public:
	float Distance;
	float AbsoluteDistance;
	int ShapeIndex;
};

class RayIntersectionPayload {
public:
	float IntersectionDistance;
	glm::vec3 IntersectionPosition;
	glm::vec3 SurfaceNormal;
	int ShapeIndex;
	int StepCount;
	bool InsideShape;
};

__TARGET_GPU__ 
static ClosestEntity FindClosestShape(
	const glm::vec3 point, const Interval<float>& interval, 
	const Shapes::Sphere* shapes, const size_t shapesCount)
{
	ClosestEntity result{};
	result.Distance = interval.Max;
	result.AbsoluteDistance = glm::abs(interval.Max);
	result.ShapeIndex = -1;

	for (int shapeIndex = 0; shapeIndex < shapesCount; ++shapeIndex)
	{
		Shapes::Sphere shape = shapes[shapeIndex];
		float distanceToShape = glm::length(point - shape.Position) - shape.Radius;
		float absoluteDistance = glm::abs(distanceToShape);

		if (absoluteDistance < result.Distance)
		{
			result.ShapeIndex = shapeIndex;
			result.Distance = distanceToShape;
			result.AbsoluteDistance = absoluteDistance;
		}
	}

	return result;
}

__TARGET_GPU__ 
static RayIntersectionPayload MarchRay(
	const glm::vec3& sourcePosition, const Ray& ray, const Interval<float>& interval, 
	const Shapes::Sphere* shapes, const size_t shapesCount) 
{
	float distanceMarched = 0.0f;
	int step = 0;
	for (;step < MAX_STEP_COUNT; ++step)
	{
		glm::vec3 point = ray.Origin + ray.Direction * distanceMarched;
		if (interval.Max <= glm::distance(sourcePosition, point))
			break;

		ClosestEntity closestShape = FindClosestShape(point, interval, shapes, shapesCount);
		distanceMarched += closestShape.AbsoluteDistance;

		if (closestShape.Distance < MIN_DISTANCE_TO_COLLIDE)
		{
			RayIntersectionPayload payload{};
			payload.IntersectionDistance = distanceMarched;
			payload.IntersectionPosition = ray.Origin + ray.Direction * distanceMarched;
			payload.StepCount = step;
			payload.ShapeIndex = closestShape.ShapeIndex;
			payload.SurfaceNormal = glm::normalize(payload.IntersectionPosition - shapes[closestShape.ShapeIndex].Position);
			payload.InsideShape = closestShape.Distance <= -MIN_DISTANCE_TO_COLLIDE;
			return payload;
		}
	}

	RayIntersectionPayload payload{};
	payload.StepCount = step;
	payload.ShapeIndex = -1;
	return payload;
}

__TARGET_GPU__ glm::vec3 RandomPointInUnitSphere(curandState* state) {
	float u = curand_uniform(state); // Random value for radius
	float v = curand_uniform(state); // Random value for azimuthal angle
	float w = curand_uniform(state); // Random value for polar angle

	float radius = cbrt(u);
	float phi = 2.0f * M_PI * v; // Azimuthal angle
	float theta = acos(2.0f * w - 1.0f); // Polar angle

	return glm::vec3{ sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta) };
}

__TARGET_GPU__ 
static float Reflectance(float cosine, float refractionIndicesRatio) 
{
	// Schlick's approximation
	auto r0 = (1.0f - refractionIndicesRatio) / (1.0f + refractionIndicesRatio);
	r0 *= r0;
	return r0 + (1.0f - r0) * glm::pow((1.0f - cosine), 5.0f);
}

__global__
static void RayGen(
	curandState* globalState, uint32_t* resultingImage, glm::vec4* imageAccumulator, int frameIndex,
	const Ray* rays, const uint32_t imageWidth, const uint32_t imageHeight, const Interval<float> interval,
	const Shapes::Sphere* shapes, const size_t shapesCount,
	const Material* materials, const size_t materialsCount)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= imageWidth || y >= imageHeight) return;

	const uint32_t inlineCoord = x + y * imageWidth;
	curandState localState = globalState[inlineCoord];
	curand_init(inlineCoord * frameIndex, inlineCoord, 0, &localState);

	Ray ray = rays[inlineCoord];
	
	glm::vec3 light = glm::mix({ 0.5f, 0.7f, 1.0f }, glm::vec3{ 1.0f }, 0.5f * (ray.Direction.y + 1.0f));
	
	glm::vec3 sourcePosition = ray.Origin;
	for (int bounceCount = 0; bounceCount < MAX_BOUNCE_COUNT; ++bounceCount)
	{
		RayIntersectionPayload payload = MarchRay(sourcePosition, ray, interval, shapes, shapesCount);

		if (payload.ShapeIndex == -1)
			break;

		const Shapes::Sphere shape = shapes[payload.ShapeIndex];
		const Material material = materials[shape.MaterialIndex];
		const glm::vec3 normalAgainstRay = payload.InsideShape ? -payload.SurfaceNormal : payload.SurfaceNormal;

		if (payload.InsideShape)
			light *= glm::min(glm::vec3{ 1.0f }, material.Albedo * (1.0f - material.Opacity) / payload.IntersectionDistance);
		else
			light *= material.Albedo;


		/* Material is refractive */
		if (material.RefractivityIndex > 1.0f)
		{
			const float eta = payload.InsideShape ? material.RefractivityIndex : (/* Void */ 1.0f / material.RefractivityIndex);

			const float cosTheta = glm::dot(-ray.Direction, normalAgainstRay);
			const float sinThetaSquared = 1.0f - cosTheta * cosTheta;
			const bool canRefract = eta * eta * sinThetaSquared <= 1.0f;

			if (canRefract) 
			{
				ray.Origin = payload.IntersectionPosition - normalAgainstRay * MIN_DISTANCE_TO_COLLIDE * 10.0f;
				ray.Direction = glm::refract(ray.Direction, normalAgainstRay, eta);
				continue;
			}
		}

		/* Material is not transparent, or total reflection */
		{	
			ray.Origin = payload.IntersectionPosition + normalAgainstRay * MIN_DISTANCE_TO_COLLIDE * 10.0f;
			glm::vec3 reflectedRayDirection = glm::reflect(ray.Direction, normalAgainstRay);
			ray.Direction = glm::normalize(reflectedRayDirection + material.Roughness * RandomPointInUnitSphere(&localState));
			
			continue;
		}
	}

	//glm::vec4 color{ Utils::LinearToGammaTransform(light), 1.0f };
	glm::vec4 color{ light, 1.0f };
	imageAccumulator[inlineCoord] += color;
	glm::vec4 averageColor = imageAccumulator[inlineCoord] / (float)frameIndex;
	averageColor = glm::clamp(averageColor, { 0.0f }, { 1.0f });
	resultingImage[inlineCoord] = Utils::ConvertToRGBA(averageColor);
	globalState[inlineCoord] = localState;
}

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

void Renderer::Render(const Scene* scene, const Camera* camera)
{
	m_ActiveCamera = camera;
	m_ActiveScene = scene;

	const uint32_t imageWidth = m_FinalImage->GetWidth();
	const uint32_t imageHeight = m_FinalImage->GetHeight();
	const size_t imageSize = imageWidth * imageHeight;

	if (m_FrameIndex == 1)
		cudaMemset(m_AccumulationData, 0, imageSize * sizeof(glm::vec4));

	// I should try to parallellize this/save and reuse. It's the most of my render time.
	uint32_t* p_resultingImage;
	cudaMallocManaged(&p_resultingImage, imageSize * sizeof(uint32_t));
	
	std::vector<Ray> rays;
	std::transform(m_ActiveCamera->GetRayDirections().begin(), m_ActiveCamera->GetRayDirections().end(), std::back_inserter(rays),
		[this](const glm::vec3 rayDirection) { return Ray{ m_ActiveCamera->GetPosition(), rayDirection }; });
	Ray* p_rays;
	cudaMallocManaged(&p_rays, rays.size() * sizeof(Ray));
	cudaMemcpy(p_rays, rays.data(), rays.size() * sizeof(Ray), cudaMemcpyHostToDevice);

	std::vector<Shapes::Sphere> shapes;
	std::transform(m_ActiveScene->Shapes.begin(), m_ActiveScene->Shapes.end(), std::back_inserter(shapes), 
		[](const Shapes::Sphere* shape) { return *shape; });
	Shapes::Sphere* p_shapes;
	cudaMallocManaged(&p_shapes, shapes.size() * sizeof(Shapes::Sphere));
	cudaMemcpy(p_shapes, shapes.data(), shapes.size() * sizeof(Shapes::Sphere), cudaMemcpyHostToDevice);

	std::vector<Material> materials;
	std::transform(m_ActiveScene->Materials.begin(), m_ActiveScene->Materials.end(), std::back_inserter(materials),
		[](const Material* material) { return *material; });
	Material* p_materials;
	cudaMallocManaged(&p_materials, materials.size() * sizeof(Material));
	cudaMemcpy(p_materials, materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	Interval<float> renderInterval{ m_ActiveCamera->NearPlane, m_ActiveCamera->FarPlane };

	dim3 blockDim(32, 32);
	dim3 gridDim((imageWidth + blockDim.x - 1) / blockDim.x, (imageHeight + blockDim.y - 1) / blockDim.y);

	curandState* d_states;
	cudaMalloc(&d_states, imageSize * sizeof(curandState));

	RayGen <<<gridDim, blockDim>>> (
		d_states, p_resultingImage, m_AccumulationData, m_FrameIndex,
		p_rays, imageWidth, imageHeight, renderInterval, 
		p_shapes, shapes.size(), 
		p_materials, materials.size()
	);
	cudaDeviceSynchronize();


	m_FinalImage->SetData(p_resultingImage);

	cudaFree(d_states);
	cudaFree(p_resultingImage);
	cudaFree(p_rays);
	cudaFree(p_shapes);

	if (m_Settings.Accumulate)
		++m_FrameIndex;
	else
		ResetFrameIndex();
}
