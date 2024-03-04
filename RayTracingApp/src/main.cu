#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "Walnut/Image.h"
#include "Walnut/Random.h"
#include "Walnut/Timer.h"

#include "Camera.h"
#include "Renderer.h"

#include "Shapes/Shape.h"
#include "Shapes/Sphere.h"
#include "Shapes/Plane.h"
#include "Shapes/TriangleMesh.h"

#include "glm/gtc/type_ptr.hpp"

#include <sstream>
#include <string>
#include <iomanip>
#include <limits>

#include "cuda/test.h"

class DefaultLayer : public Walnut::Layer
{
public:
	DefaultLayer():
		m_ActiveCamera(45.0f, 0.1f, 100.0f),
		m_ViewportWidth(0),
		m_ViewportHeight(0)
	{
		{ /* Material 1 */
			Material* material = new Material();
			material->Albedo = { 0.161f, 0.565f, 0.710f }; // Neptune average color
			material->Roughness = 0.4f;
			m_ActiveScene.Materials.push_back(material);
		}
		{ /* Material 2 */
			Material* material = new Material();
			material->Albedo = { 0.1f, 0.2f, 0.8f };
			material->Roughness = 0.1f;
			m_ActiveScene.Materials.push_back(material);
		}
		{ /* Material 3 */
			Material* material = new Material();
			material->Albedo = { 1.0f, 0.25f, 1.0f };
			material->Roughness = 0.2f;
			m_ActiveScene.Materials.push_back(material);
		}
		{ /* Material 4 */
			Material* material = new Material();
			material->Albedo = { 1.0f, 0.7f, 0.25f };
			material->Roughness = 0.2f;
			material->EmissionColor = { 1.0f, 0.7f, 0.25f };
			material->EmissionPower = 10.0f;
			m_ActiveScene.Materials.push_back(material);
		}
		{ /* Material 5 */
			Material* material = new Material();
			material->Albedo = { 1.0f, 1.0f, 1.0f };
			material->Roughness = 0.3f;
			m_ActiveScene.Materials.push_back(material);
		}

		{ /* Sphere 1 */
			Shapes::Sphere* sphere = new Shapes::Sphere();
			sphere->MaterialIndex = 0;
			sphere->Radius = 100.0f;
			sphere->Position = glm::vec3{ 0.0f, -101.0f, 0.0f };
			m_ActiveScene.Shapes.push_back(sphere);
		}
		{ /* Sphere 2 */
			Shapes::Sphere* sphere = new Shapes::Sphere();
			sphere->MaterialIndex = 2;
			sphere->Radius = 0.5f;
			sphere->Position = glm::vec3{ -1.0f, 0.5f, 10.0f };
			m_ActiveScene.Shapes.push_back(sphere);
		}
		{ /* Sphere 3 */
			Shapes::Sphere* sphere = new Shapes::Sphere();
			sphere->MaterialIndex = 3;
			sphere->Radius = 1.0f;
			sphere->Position = glm::vec3{ 1.0f, 0.0f, 0.0f };
			m_ActiveScene.Shapes.push_back(sphere);
		}
		{ /* Sphere 4 */
			Shapes::Sphere* sphere = new Shapes::Sphere();
			sphere->MaterialIndex = 2;
			sphere->Radius = 10.0f;
			sphere->Position = glm::vec3{ 6.0f, 2.5f, -15.0f };
			m_ActiveScene.Shapes.push_back(sphere);
		}
		{ /* Plane */
			Shapes::Plane* plane = new Shapes::Plane(glm::normalize(glm::vec3(1, 1, 0)), -2.0f);
			plane->MaterialIndex = 1;
			m_ActiveScene.Shapes.push_back(plane);
		}
		{ /* Cube */
			std::vector<glm::vec3> vertices = {
				{ 0.0f, 0.0f, 0.0f },
				{ 0.0f, 0.0f, 1.0f },
				{ 0.0f, 1.0f, 0.0f },
				{ 0.0f, 1.0f, 1.0f },
				{ 1.0f, 0.0f, 0.0f },
				{ 1.0f, 0.0f, 1.0f },
				{ 1.0f, 1.0f, 0.0f },
				{ 1.0f, 1.0f, 1.0f },
			};
			std::vector<glm::vec3> normals = {
				glm::normalize(glm::vec3 { -1.0f, -1.0f, -1.0f }),
				glm::normalize(glm::vec3 { -1.0f, -1.0f,  1.0f }),
				glm::normalize(glm::vec3 { -1.0f,  1.0f, -1.0f }),
				glm::normalize(glm::vec3 { -1.0f,  1.0f,  1.0f }),
				glm::normalize(glm::vec3 {  1.0f, -1.0f, -1.0f }),
				glm::normalize(glm::vec3 {  1.0f, -1.0f,  1.0f }),
				glm::normalize(glm::vec3 {  1.0f,  1.0f, -1.0f }),
				glm::normalize(glm::vec3 {  1.0f,  1.0f,  1.0f }),
			};
			std::vector<size_t> indices = { 
				// front
				1, 5, 3,
				5, 7, 3,

				// back
				0, 4, 2,
				4, 6, 2,

				// top
				3, 7, 2,
				7, 6, 2,

				// bottom
				0, 4, 1,
				4, 5, 1,

				// left
				0, 1, 2,
				1, 3, 2,

				// right
				5, 4, 7,
				4, 6, 7,
			};
			Shapes::TriangleMesh* mesh = new Shapes::TriangleMesh(vertices, normals, indices);
			mesh->MaterialIndex = 2;
			m_ActiveScene.Shapes.push_back(mesh);
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
		RenderMaterialEditor();
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

		for (Shapes::Shape* shape : m_ActiveScene.Shapes)
		{
			ImGui::PushID(shape->ShapeIndex);
			
			if (Shapes::Sphere* sphere = dynamic_cast<Shapes::Sphere*>(shape))
			{
				ImGui::Text("Sphere %d", sphere->ShapeIndex);
				m_ShouldRedraw |= ImGui::DragFloat3("Position", glm::value_ptr(sphere->Position), 0.1f);
				m_ShouldRedraw |= ImGui::DragFloat("Radius", &sphere->Radius, 0.1f, 0.0f, std::numeric_limits<float>::max());
			}
			else if (Shapes::Plane* plane = dynamic_cast<Shapes::Plane*>(shape))
			{
				ImGui::Text("Plane %d", plane->ShapeIndex);
				
				m_ShouldRedraw |= ImGui::DragFloat3("Normal", glm::value_ptr(plane->Normal), 0.1f);
				if (ImGui::Button("Normalize Normal", ImVec2(-1, 0)))
				{
					m_ShouldRedraw = true;
					plane->Normal = glm::normalize(plane->Normal);
				}
				ImGui::Spacing();

				m_ShouldRedraw |= ImGui::DragFloat("Offset", &plane->Offset, 0.1f);
			}
			else if (Shapes::TriangleMesh* mesh = dynamic_cast<Shapes::TriangleMesh*>(shape))
			{
				ImGui::Text("Mesh %d", mesh->ShapeIndex);
			}
			else
			{
				ImGui::Text("Shape %d", shape->ShapeIndex);
			}

			m_ShouldRedraw |= ImGui::DragInt("Material", &shape->MaterialIndex, 1, 0, (int)m_ActiveScene.Materials.size() - 1);

			ImGui::Separator();
			ImGui::PopID();
		}

		ImGui::End();
	}

	void RenderMaterialEditor()
	{
		ImGui::Begin("Materials Editor");

		for (size_t i = 0; i < m_ActiveScene.Materials.size(); i++)
		{
			ImGui::PushID((int)i);

			ImGui::Text("Material %d", i);

			Material* material = m_ActiveScene.Materials[i];
			m_ShouldRedraw |= ImGui::ColorEdit3("Albedo", glm::value_ptr(material->Albedo));
			m_ShouldRedraw |= ImGui::DragFloat("Roughness", &material->Roughness, 0.01f, 0.0f, 1.0f);
			m_ShouldRedraw |= ImGui::DragFloat("Metallic", &material->Metallic, 0.01f, 0.0f, 1.0f);
			m_ShouldRedraw |= ImGui::ColorEdit3("Emission Color", glm::value_ptr(material->EmissionColor));
			m_ShouldRedraw |= ImGui::DragFloat("Emission Power", &material->EmissionPower, 0.01f, 0.0f, std::numeric_limits<float>::max());

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
			ImGui::Image((ImTextureID)image->GetDescriptorSet(), {
				(float)image->GetWidth(), (float)image->GetHeight()
				}, ImVec2(0, 1), ImVec2(1, 0));

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
	spec.Width = 960;
	spec.Height = 540;

	// This is a CUDA test, to make sure it is possible to compute using the GPU
	add_vectors_and_print_them(1000);

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