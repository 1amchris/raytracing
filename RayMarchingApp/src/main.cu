#include <string>
#include <limits>
#include <glm/gtc/type_ptr.hpp>

#include <Walnut/Application.h>
#include <Walnut/EntryPoint.h>
#include <Walnut/Image.h>
#include <Walnut/Random.h>
#include <Walnut/Timer.h>

#include "Camera.h"
#include "Renderer.h"
#include "Sphere.h"


class DefaultLayer : public Walnut::Layer
{
public:
	DefaultLayer():
		m_ActiveCamera(45.0f, 0.1f, 100.0f),
		m_ViewportWidth(0),
		m_ViewportHeight(0)
	{
		{ /* Sphere 1 */
			Shapes::Sphere* sphere = new Shapes::Sphere();
			sphere->Radius = 1.0f;
			sphere->Position = glm::vec3{ 0.0f, 0.0f, 2.0f };
			m_ActiveScene.Shapes.push_back(sphere);
		}

		m_ActiveScene.ResetShapeIndices();
	}

	~DefaultLayer() {
		m_ActiveScene.Dispose();
	}

	virtual void OnUpdate(float timeStep) override 
	{
		if (m_ActiveCamera.OnUpdate(timeStep))
		{
			m_Renderer.ResetFrameIndex();
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
			m_Renderer.ResetFrameIndex();
			m_ShouldRedraw = false;
		}

		Render();
	};

	void Render()
	{
		Walnut::Timer timer;

		m_Renderer.OnResize(m_ViewportWidth, m_ViewportHeight);
		m_ActiveCamera.OnResize(m_ViewportWidth, m_ViewportHeight);
		
		m_Renderer.Render(m_ActiveScene, m_ActiveCamera);

		m_LastRenderTime = timer.ElapsedMillis();
	}

private:
	void RenderConsole()
	{
		ImGui::Begin("Console");
		ImGui::Text("Last Render: %.3fms", m_LastRenderTime);
		ImGui::Separator();
		ImGui::Checkbox("Accumulate", &m_Renderer.GetSettings().Accumulate);
		m_ShouldRedraw |= ImGui::Button("Redraw Scene");
		ImGui::End();
	}

	void RenderShapeEditor() 
	{
		ImGui::Begin("Spheres Editor");

		for (Shapes::Sphere* shape : m_ActiveScene.Shapes)
		{
			ImGui::PushID(shape->ShapeIndex);
			
			ImGui::Text("Sphere %d", shape->ShapeIndex);
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

		auto image = m_Renderer.GetFinalImage();
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
		m_Renderer.ResetFrameIndex();
	}

private:
	Renderer m_Renderer;
	Camera m_ActiveCamera;
	Scene m_ActiveScene;
	
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