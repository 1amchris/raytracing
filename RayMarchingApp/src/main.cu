#include <string>
#include <limits>
#include <memory>
#include <glm/gtc/type_ptr.hpp>

#include <Walnut/Application.h>
#include <Walnut/EntryPoint.h>
#include <Walnut/Image.h>
#include <Walnut/Random.h>
#include <Walnut/Timer.h>

#include "Camera.h"
#include "Renderer.h"
#include "Sphere.h"
#include "Material.h"


class DefaultLayer : public Walnut::Layer
{
public:
	DefaultLayer():
		m_ViewportWidth(0),
		m_ViewportHeight(0)
	{
		{ /* Material 1 */
			Material* material = new Material();
			material->Albedo = { 0.161f, 0.565f, 0.710f }; // Neptune average color
			material->Roughness = 0.4f;
			m_ActiveScene->Materials.push_back(material);
		}
		{ /* Material 2 */
			Material* material = new Material();
			material->Albedo = { 0.1f, 0.2f, 0.8f };
			material->Roughness = 0.01f;
			m_ActiveScene->Materials.push_back(material);
		}
		{ /* Material 3 */
			Material* material = new Material();
			material->Albedo = { 8.0f, 0.55f, 0.1f };
			material->Roughness = 0.2f;
			m_ActiveScene->Materials.push_back(material);
		}
		{ /* Material 4 */
			Material* material = new Material();
			material->Albedo = { 0.95f, 0.975f, 1.0f };
			material->RefractiveIndex = 1.458f;
			material->Roughness = 0.1f;
			m_ActiveScene->Materials.push_back(material);
		}

		{ /* Sphere 1 */
			Shapes::Sphere* sphere = new Shapes::Sphere();
			sphere->Radius = 1.25f;
			sphere->Position = glm::vec3{ 0.0f, 0.0f, 0.0f };
			sphere->MaterialIndex = 0;
			m_ActiveScene->Shapes.push_back(sphere);
		}
		{ /* Sphere 2 */
			Shapes::Sphere* sphere = new Shapes::Sphere();
			sphere->Radius = 1.1f;
			sphere->Position = glm::vec3{ 1.85f, 2.3f, 0.0f };
			sphere->MaterialIndex = 3;
			m_ActiveScene->Shapes.push_back(sphere);
		}
		{ /* Sphere 3 */
			Shapes::Sphere* sphere = new Shapes::Sphere();
			sphere->Radius = 1.25f;
			sphere->Position = glm::vec3{ 4.0f, 1.0f, 1.0f };
			sphere->MaterialIndex = 0;
			m_ActiveScene->Shapes.push_back(sphere);
		}
		{ /* Sphere 4 */
			Shapes::Sphere* sphere = new Shapes::Sphere();
			sphere->Radius = 1.25f;
			sphere->Position = glm::vec3{ 1.9f, 0.0f, -1.0f };
			sphere->MaterialIndex = 1;
			m_ActiveScene->Shapes.push_back(sphere);
		}
		{ /* Sphere 5 */
			Shapes::Sphere* sphere = new Shapes::Sphere();
			sphere->Radius = 1.25f;
			sphere->Position = glm::vec3{ 2.0f, 0.0f, 2.0f };
			sphere->MaterialIndex = 2;
			m_ActiveScene->Shapes.push_back(sphere);
		}

		{ /* Sphere 6 */
			Shapes::Sphere* sphere = new Shapes::Sphere();
			sphere->Radius = 1.25f;
			sphere->Position = glm::vec3{ 0.0f, 4.0f, 0.0f };
			sphere->MaterialIndex = 0;
			m_ActiveScene->Shapes.push_back(sphere);
		}
		{ /* Sphere 7 */
			Shapes::Sphere* sphere = new Shapes::Sphere();
			sphere->Radius = 1.25f;
			sphere->Position = glm::vec3{ 4.0f, 5.0f, 1.0f };
			sphere->MaterialIndex = 0;
			m_ActiveScene->Shapes.push_back(sphere);
		}
		{ /* Sphere 8 */
			Shapes::Sphere* sphere = new Shapes::Sphere();
			sphere->Radius = 1.25f;
			sphere->Position = glm::vec3{ 1.9f, 4.0f, -1.0f };
			sphere->MaterialIndex = 1;
			m_ActiveScene->Shapes.push_back(sphere);
		}
		{ /* Sphere 9 */
			Shapes::Sphere* sphere = new Shapes::Sphere();
			sphere->Radius = 1.25f;
			sphere->Position = glm::vec3{ 2.0f, 4.0f, 2.0f };
			sphere->MaterialIndex = 2;
			m_ActiveScene->Shapes.push_back(sphere);
		}

		m_ActiveScene->ResetIndices();
	}

	~DefaultLayer() {
		m_ActiveScene->Dispose();
	}

	virtual void OnUpdate(float timeStep) override 
	{
		if (m_ActiveCamera->OnUpdate(timeStep))
		{
			m_Renderer->ResetFrameIndex();
			m_ShouldRedraw = false;
		}
	}

	virtual void OnUIRender() override 
	{
		RenderConsole();
		RenderShapeEditor();
		RenderViewport();

		if (m_ShouldRedraw)
		{
			m_Renderer->ResetFrameIndex();
			m_ShouldRedraw = false;
		}

		Render();
	};

	void Render()
	{
		Walnut::Timer timer;

		m_Renderer->OnResize(m_ViewportWidth, m_ViewportHeight);
		m_ActiveCamera->OnResize(m_ViewportWidth, m_ViewportHeight);
		
		m_Renderer->Render(m_ActiveScene.get(), m_ActiveCamera.get());

		m_LastRenderTime = timer.ElapsedMillis();
	}

private:
	void RenderConsole()
	{
		bool recalculateProjection = false;
		ImVec4 ImGui_Header1 = ImVec4(0, 1, 1, 1);

		ImGui::Begin("Console");
		
		{
			ImGui::TextColored(ImGui_Header1, "Render Settings");
			
			ImGui::Text("Last Render: %.3fms", m_LastRenderTime);
			ImGui::Checkbox("Accumulate", &m_Renderer->GetSettings().Accumulate);
		}

		ImGui::Separator();

		{
			ImGui::TextColored(ImGui_Header1, "Camera Settings");
		
			recalculateProjection |= ImGui::DragFloat("FOV", &m_ActiveCamera->VerticalFOV, 0.1f, 0.0f, 179.9f);
			recalculateProjection |= ImGui::DragFloat("Near Plane", &m_ActiveCamera->NearPlane, 0.0f, m_ActiveCamera->FarPlane);
			recalculateProjection |= ImGui::DragFloat("Far Plane", &m_ActiveCamera->FarPlane, m_ActiveCamera->NearPlane, std::numeric_limits<float>::max());
		}

		ImGui::End();

		if (recalculateProjection)
		{
			m_ShouldRedraw |= true;
			m_ActiveCamera->OnProjectionChange();
		}
	}

	void RenderShapeEditor() 
	{
		ImGui::Begin("Spheres Editor");

		for (Shapes::Sphere* shape : m_ActiveScene->Shapes)
		{
			ImGui::PushID(shape->ShapeIndex);
			
			ImGui::Text("Sphere %d", shape->ShapeIndex + 1);
			m_ShouldRedraw |= ImGui::DragFloat3("Position", glm::value_ptr(shape->Position), 0.1f);
			m_ShouldRedraw |= ImGui::DragFloat("Radius", &shape->Radius, 0.1f, 0.0f, std::numeric_limits<float>::max());

			ImGui::Separator();
			ImGui::PopID();
		}

		ImGui::End();
	}

	void RenderViewport()
	{
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		ImGui::Begin("Viewport");

		m_ViewportWidth = (int)ImGui::GetContentRegionAvail().x;
		m_ViewportHeight = (int)ImGui::GetContentRegionAvail().y;

		auto image = m_Renderer->GetFinalImage();
		if (image)
			ImGui::Image(
				(ImTextureID)image->GetDescriptorSet(),
				{ (float)image->GetWidth(), (float)image->GetHeight() },
				ImVec2(0, 1), ImVec2(1, 0)
			);

		ImGui::End();
		ImGui::PopStyleVar();
	}

	void OnSceneUpdate()
	{
		m_Renderer->ResetFrameIndex();
	}

private:
	std::shared_ptr<Camera> m_ActiveCamera = std::make_shared<Camera>(90.0f, 0.1f, 100.0f);
	std::shared_ptr<Scene> m_ActiveScene = std::make_shared<Scene>();
	std::shared_ptr<Renderer> m_Renderer = std::make_shared<Renderer>();
	
	uint32_t m_ViewportHeight, m_ViewportWidth;
	float m_LastRenderTime = 0.0f;
	bool m_ShouldRedraw = false;
};

Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{
	Walnut::ApplicationSpecification spec;
	spec.Name = "Check me out on Github: https://github.com/1amchris/raytracing";
	spec.Width = 1280;
	spec.Height = 720;

	Walnut::Application* app = new Walnut::Application(spec);
	app->PushLayer<DefaultLayer>();
	app->SetMenubarCallback([app]()
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Exit"))
			{
				app->Close();
			}
			ImGui::EndMenu();
		}
	});

	return app;
}