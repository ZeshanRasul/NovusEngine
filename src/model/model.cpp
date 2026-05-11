#include "model.h"
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>

#include <iostream>

void Model::loadModel(std::string modelFilename, RenderableComponent& gameObj)
{
	tinygltf::Model    model;
	tinygltf::TinyGLTF loader;
	std::string        err;
	std::string        warn;

	bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, modelFilename);

	if (!warn.empty())
		std::cout << "glTF warning: " << warn << std::endl;
	if (!err.empty())
		std::cout << "glTF error: " << err << std::endl;
	if (!ret)
		throw std::runtime_error("Failed to load glTF model");

	gameObj.vertices.clear();
	gameObj.indices.clear();
	gameObj.meshes.clear();
	gameObj.materials.clear();
	gameObj.materialTextures.clear();
	gameObj.materialDescriptorSets.clear();

	std::string baseDir = modelFilename.substr(0, modelFilename.find_last_of("/\\") + 1);

	for (const auto& mat : model.materials)
	{
		Material material(mat.name.empty() ? "Material" : mat.name);

		if (mat.pbrMetallicRoughness.baseColorFactor.size() == 4)
		{
			material.baseColorFactor = glm::vec4(
				mat.pbrMetallicRoughness.baseColorFactor[0],
				mat.pbrMetallicRoughness.baseColorFactor[1],
				mat.pbrMetallicRoughness.baseColorFactor[2],
				mat.pbrMetallicRoughness.baseColorFactor[3]);
		}

		material.metallicFactor = static_cast<float>(mat.pbrMetallicRoughness.metallicFactor);
		material.roughnessFactor = static_cast<float>(mat.pbrMetallicRoughness.roughnessFactor);

		auto resolveTexture = [&](int texIndex, std::string& outPath, float& outIndex) {
			if (texIndex >= 0 && texIndex < static_cast<int>(model.textures.size()))
			{
				int imageIndex = model.textures[texIndex].source;
				if (imageIndex >= 0 && imageIndex < static_cast<int>(model.images.size()))
				{
					outPath = baseDir + model.images[imageIndex].uri;
					outIndex = 0.0f;
				}
			}
			};

		resolveTexture(mat.pbrMetallicRoughness.baseColorTexture.index, material.albedoTexturePath, material.baseColorTextureIndex);
		resolveTexture(mat.pbrMetallicRoughness.metallicRoughnessTexture.index, material.metallicRoughnessTexturePath, material.metallicRoughnessTextureIndex);
		resolveTexture(mat.normalTexture.index, material.normalTexturePath, material.normalTextureIndex);
		resolveTexture(mat.occlusionTexture.index, material.occlusionTexturePath, material.occlusionTextureIndex);
		resolveTexture(mat.emissiveTexture.index, material.emissiveTexturePath, material.emissiveTextureIndex);

		std::cout << "Loaded material: " << material.GetName() << std::endl;
		std::cout << "  Base color texture: " << material.albedoTexturePath << std::endl;
		std::cout << "  Metallic-roughness texture: " << material.metallicRoughnessTexturePath << std::endl;
		std::cout << "  Normal texture: " << material.normalTexturePath << std::endl;

		gameObj.materials.push_back(std::move(material));
		gameObj.materialTextures.emplace_back();
	}

	if (gameObj.materials.empty())
	{
		gameObj.materials.emplace_back("DefaultMaterial");
		gameObj.materialTextures.emplace_back();
	}

	for (const auto& mesh : model.meshes)
	{
		for (const auto& primitive : mesh.primitives)
		{
			Mesh meshInfo;
			meshInfo.firstIndex = static_cast<uint32_t>(gameObj.indices.size());
			meshInfo.materialIndex = primitive.material >= 0 ? primitive.material : 0;

			const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
			const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
			const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];

			const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.at("POSITION")];
			const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
			const tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];

			bool hasNormals = primitive.attributes.find("NORMAL") != primitive.attributes.end();
			bool hasTexCoords = primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end();
			bool hasTangents = primitive.attributes.find("TANGENT") != primitive.attributes.end();

			const tinygltf::Accessor* texCoordAccessor = nullptr;
			const tinygltf::BufferView* texCoordBufferView = nullptr;
			const tinygltf::Buffer* texCoordBuffer = nullptr;

			const tinygltf::Accessor* normalAccessor = nullptr;
			const tinygltf::BufferView* normalBufferView = nullptr;
			const tinygltf::Buffer* normalBuffer = nullptr;

			const tinygltf::Accessor* tangentAccessor = nullptr;
			const tinygltf::BufferView* tangentBufferView = nullptr;
			const tinygltf::Buffer* tangentBuffer = nullptr;

			if (hasTexCoords)
			{
				texCoordAccessor = &model.accessors[primitive.attributes.at("TEXCOORD_0")];
				texCoordBufferView = &model.bufferViews[texCoordAccessor->bufferView];
				texCoordBuffer = &model.buffers[texCoordBufferView->buffer];
			}

			if (hasNormals)
			{
				normalAccessor = &model.accessors[primitive.attributes.at("NORMAL")];
				normalBufferView = &model.bufferViews[normalAccessor->bufferView];
				normalBuffer = &model.buffers[normalBufferView->buffer];
			}

			if (hasTangents)
			{
				tangentAccessor = &model.accessors[primitive.attributes.at("TANGENT")];
				tangentBufferView = &model.bufferViews[tangentAccessor->bufferView];
				tangentBuffer = &model.buffers[tangentBufferView->buffer];
			}

			uint32_t baseVertex = static_cast<uint32_t>(gameObj.vertices.size());

			for (size_t i = 0; i < posAccessor.count; i++)
			{
				Vertex vertex{};

				const float* pos = reinterpret_cast<const float*>(&posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset + i * 12]);
				vertex.pos = { pos[0], -pos[1], pos[2] };

				if (hasTexCoords)
				{
					const float* texCoord = reinterpret_cast<const float*>(&texCoordBuffer->data[texCoordBufferView->byteOffset + texCoordAccessor->byteOffset + i * 8]);
					vertex.texCoord = { texCoord[0], texCoord[1] };
				}
				else
				{
					vertex.texCoord = { 0.0f, 0.0f };
				}

				if (hasNormals)
				{
					const float* normal = reinterpret_cast<const float*>(&normalBuffer->data[normalBufferView->byteOffset + normalAccessor->byteOffset + i * 12]);
					vertex.normal = { normal[0], -normal[1], normal[2] };
				}
				else
				{
					vertex.normal = { 0.0f, 0.0f, 0.0f };
				}

				if (hasTangents)
				{
					const float* tangent = reinterpret_cast<const float*>(&tangentBuffer->data[tangentBufferView->byteOffset + tangentAccessor->byteOffset + i * 16]);
					vertex.tangent = { tangent[0], -tangent[1], tangent[2], tangent[3] };
				}
				else
				{
					vertex.tangent = { 0.0f, 0.0f, 0.0f, 1.0f };
				}

				gameObj.vertices.push_back(vertex);
			}

			const unsigned char* indexData = &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset];
			size_t               indexCount = indexAccessor.count;
			size_t               indexStride = 0;

			if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
				indexStride = sizeof(uint16_t);
			else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
				indexStride = sizeof(uint32_t);
			else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
				indexStride = sizeof(uint8_t);
			else
				throw std::runtime_error("Unsupported index component type");

			gameObj.indices.reserve(gameObj.indices.size() + indexCount);

			for (size_t i = 0; i < indexCount; i++)
			{
				uint32_t index = 0;

				if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
					index = *reinterpret_cast<const uint16_t*>(indexData + i * indexStride);
				else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
					index = *reinterpret_cast<const uint32_t*>(indexData + i * indexStride);
				else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
					index = *reinterpret_cast<const uint8_t*>(indexData + i * indexStride);

				gameObj.indices.push_back(baseVertex + index);
			}

			meshInfo.indexCount = static_cast<uint32_t>(gameObj.indices.size() - meshInfo.firstIndex);
			gameObj.meshes.push_back(meshInfo);
		}
	}
}
