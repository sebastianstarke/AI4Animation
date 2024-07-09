/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Licensed under the Oculus SDK License Agreement (the "License");
 * you may not use the Oculus SDK except in compliance with the License,
 * which is provided at the time of installation or download, or which
 * otherwise accompanies this software in either electronic or hard copy form.
 *
 * You may obtain a copy of the License at
 *
 * https://developer.oculus.com/licenses/oculussdk/
 *
 * Unless required by applicable law or agreed to in writing, the Oculus SDK
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using UnityEngine;
using OVRSimpleJSON;

using System.Threading.Tasks;

/// <summary>
/// This is a lightweight glTF model loader that is guaranteed to work with models loaded from the Oculus runtime
/// using OVRPlugin.LoadRenderModel. It is not recommended to be used as a general purpose glTF loader.
/// </summary>

public enum OVRChunkType
{
	JSON = 0x4E4F534A,
	BIN = 0x004E4942,
}

public enum OVRTextureFormat
{
	NONE,
	KTX2,
	PNG,
	JPEG,
}

/// <summary>
/// This enum represents a simplified representation on how Texture Filter quality is implemented in Unity.
/// The values set in this enum are NOT random and are directly used by ApplyTextureQuality() and DetectTextureQuality()
/// to get/set the correspondent setup in Unity.
/// </summary>
public enum OVRTextureQualityFiltering
{
	None = -1,
	Bilinear = 0,
	Trilinear = 1,
	Aniso2x = 2,
	Aniso4x = 3,
	Aniso8x = 4,
	Aniso16x = 5,
}

public struct OVRBinaryChunk
{
	public Stream chunkStream;
	public uint chunkLength;
	public long chunkStart;
}

public struct OVRMeshData
{
	public Mesh mesh;
	public Material material;
}

public struct OVRMaterialData
{
	public Shader shader;
	public int textureId;
	public OVRTextureData texture;
}

public struct OVRGLTFScene
{
	public GameObject root;
	public List<GameObject> nodes;
	public Dictionary<OVRGLTFInputNode, OVRGLTFAnimatinonNode> animationNodes;
}

public struct OVRTextureData
{
	public byte[] data;
	public int width;
	public int height;
	public OVRTextureFormat format;
	public TextureFormat transcodedFormat;
}

public class OVRGLTFLoader
{
	private JSONNode m_jsonData;
	private Stream m_glbStream;
	private OVRBinaryChunk m_binaryChunk;

	private List<GameObject> m_Nodes;
	private Dictionary<OVRGLTFInputNode, OVRGLTFAnimatinonNode> m_AnimationNodes;

	private Shader m_Shader = null;
	private Shader m_AlphaBlendShader = null;
	private OVRTextureQualityFiltering m_TextureQuality = OVRTextureQualityFiltering.Bilinear;  // = Unity default
	private float m_TextureMipmapBias = 0.0f;	// = shader default

	public static readonly Vector3 GLTFToUnitySpace = new Vector3(-1, 1, 1);
	public static readonly Vector3 GLTFToUnityTangent = new Vector4(-1, 1, 1, -1);
	public static readonly Vector4 GLTFToUnitySpace_Rotation = new Vector4(1, -1, -1, 1);

	private static Dictionary<string, OVRGLTFInputNode> InputNodeNameMap = new Dictionary<string, OVRGLTFInputNode>{
		{"button_a", OVRGLTFInputNode.Button_A_X },
		{"button_x", OVRGLTFInputNode.Button_A_X },
		{"button_b", OVRGLTFInputNode.Button_B_Y },
		{"button_y", OVRGLTFInputNode.Button_B_Y },
		{"button_oculus", OVRGLTFInputNode.Button_Oculus_Menu },
		{"trigger_front", OVRGLTFInputNode.Trigger_Front },
		{"trigger_grip", OVRGLTFInputNode.Trigger_Grip},
		{"thumbstick", OVRGLTFInputNode.ThumbStick },
	};

	public OVRGLTFLoader(string fileName)
	{
		m_glbStream = File.Open(fileName, FileMode.Open);
	}

	public OVRGLTFLoader(byte[] data)
	{
		m_glbStream = new MemoryStream(data, 0, data.Length, false, true);
	}

	public OVRGLTFScene LoadGLB(bool supportAnimation, bool loadMips = true)
	{
		OVRGLTFScene scene = new OVRGLTFScene();
		m_Nodes = new List<GameObject>();
		m_AnimationNodes = new Dictionary<OVRGLTFInputNode, OVRGLTFAnimatinonNode>();

        int rootNodeId = 0;
		if (ValidateGLB(m_glbStream))
		{
			byte[] jsonChunkData = ReadChunk(m_glbStream, OVRChunkType.JSON);
			if (jsonChunkData != null)
			{
				string json = System.Text.Encoding.ASCII.GetString(jsonChunkData);
				m_jsonData = JSON.Parse(json);
			}

			uint binChunkLength = 0;
			bool validBinChunk = ValidateChunk(m_glbStream, OVRChunkType.BIN, out binChunkLength);
			if (validBinChunk && m_jsonData != null)
			{
				m_binaryChunk.chunkLength = binChunkLength;
				m_binaryChunk.chunkStart = m_glbStream.Position;
				m_binaryChunk.chunkStream = m_glbStream;

				if (m_Shader == null)
				{
					Debug.LogWarning("A shader was not set before loading the model. Using default mobile shader.");
					m_Shader = Shader.Find("Legacy Shaders/Diffuse");
				}
				if (m_AlphaBlendShader == null)
				{
					Debug.LogWarning("An alpha blend shader was not set before loading the model. Using default transparent shader.");
					m_AlphaBlendShader = Shader.Find("Unlit/Transparent");
				}

				rootNodeId = LoadGLTF(supportAnimation, loadMips);
				if (rootNodeId < 0)
				{
					m_glbStream.Close();
					return scene;
				}
			}
		}
		m_glbStream.Close();

		scene.nodes = m_Nodes;
		scene.root = new GameObject("GLB Scene Root");
		scene.animationNodes = m_AnimationNodes;
        foreach (GameObject node in m_Nodes)
		{
			if (node.transform.parent == null)
			{
				node.transform.SetParent(scene.root.transform);
			}
		}

		scene.root.transform.Rotate(Vector3.up, 180.0f);

		return scene;
	}

	public void SetModelShader(Shader shader)
	{
		m_Shader = shader;
	}

	public void SetModelAlphaBlendShader(Shader shader)
	{
		m_AlphaBlendShader = shader;
	}

	/// <summary>
	/// All textures in the glb will be loaded with the following setting. The default is Bilinear.
	/// Once loaded, textures will be read-only on GPU memory.
	/// </summary>
	/// <param name="loadedTexturesQuality">The quality setting.</param>
	public void SetTextureQualityFiltering(OVRTextureQualityFiltering loadedTexturesQuality)
	{
		m_TextureQuality = loadedTexturesQuality;
	}

	/// <summary>
	/// All textures in the glb will be preset with this MipMap value. The default is 0.
	/// Only supported when MipMaps are loaded and the provided shader has a property named "_MainTexMMBias"
	/// </summary>
	/// <param name="loadedTexturesMipmapBiasing">The value for bias. Value is clamped between [-1,1]</param>
	public void SetMipMapBias(float loadedTexturesMipmapBiasing)
	{
		m_TextureMipmapBias = Mathf.Clamp(loadedTexturesMipmapBiasing, -1.0f, 1.0f);
	}

	/// <summary>
	/// Decodes the Texture Quality setting from the input Texture2D properties' values.
	/// </summary>
	/// <param name="srcTexture">The input Texture2D</param>
	/// <returns>The enum TextureQualityFiltering representing the quality.</returns>
	public static OVRTextureQualityFiltering DetectTextureQuality(in Texture2D srcTexture)
	{
		OVRTextureQualityFiltering quality = OVRTextureQualityFiltering.None;
		switch (srcTexture.filterMode)
		{
			case FilterMode.Point:
				quality = OVRTextureQualityFiltering.None;
				break;
			case FilterMode.Bilinear:
				goto default;
			case FilterMode.Trilinear:
				if (srcTexture.anisoLevel <= 1)
					quality = OVRTextureQualityFiltering.Trilinear;
				// In theory, aniso supports values between 2-16x, but in reality GPUs and gfx APIs implement powers of 2 (values in between have no change)
				else if (srcTexture.anisoLevel < 4)
					quality = OVRTextureQualityFiltering.Aniso2x;
				else if (srcTexture.anisoLevel < 8)
					quality = OVRTextureQualityFiltering.Aniso4x;
				else if (srcTexture.anisoLevel < 16)
					quality = OVRTextureQualityFiltering.Aniso8x;
				else
					quality = OVRTextureQualityFiltering.Aniso16x;
				break;
			default:
				quality = OVRTextureQualityFiltering.Bilinear;
				break;
		}
		return quality;
	}

	/// <summary>
	/// Applies the input Texture Quality setting into the ref Texture2D provided as input. Texture2D must not be readonly.
	/// </summary>
	/// <param name="qualityLevel">The quality level to apply</param>
	/// <param name="destTexture">The destination Texture2D to apply quality setting to</param>
	public static void ApplyTextureQuality(OVRTextureQualityFiltering qualityLevel, ref Texture2D destTexture)
	{
		if (destTexture == null)
			return;

		switch (qualityLevel)
		{
			case OVRTextureQualityFiltering.None:
				destTexture.filterMode = FilterMode.Point;
				destTexture.anisoLevel = 0;
				break;
			case OVRTextureQualityFiltering.Bilinear:
				destTexture.filterMode = FilterMode.Bilinear;
				destTexture.anisoLevel = 0;
				break;
			case OVRTextureQualityFiltering.Trilinear:
				destTexture.filterMode = FilterMode.Trilinear;
				destTexture.anisoLevel = 0;
				break;
			default:    // for higher values
				destTexture.filterMode = FilterMode.Trilinear;
				// In theory, aniso supports values between 2-16x, but in reality GPUs and gfx APIs implement powers of 2 (values in between have no change)
				destTexture.anisoLevel = Mathf.FloorToInt(Mathf.Pow(2.0f, (int)qualityLevel - 1));   // given the enum value, this gives aniso x2 x4 x8 x16
				break;
		}
	}

	private bool ValidateGLB(Stream glbStream)
	{
		// Read the magic entry and ensure value matches the glTF value
		int uint32Size = sizeof(uint);
		byte[] buffer = new byte[uint32Size];
		glbStream.Read(buffer, 0, uint32Size);
		uint magic = BitConverter.ToUInt32(buffer, 0);

		if (magic != 0x46546C67)
		{
			Debug.LogError("Data stream was not a valid glTF format");
			return false;
		}

		// Read glTF version
		glbStream.Read(buffer, 0, uint32Size);
		uint version = BitConverter.ToUInt32(buffer, 0);

		if (version != 2)
		{
			Debug.LogError("Only glTF 2.0 is supported");
			return false;
		}

		// Read glTF file size
		glbStream.Read(buffer, 0, uint32Size);
		uint length = BitConverter.ToUInt32(buffer, 0);
		if (length != glbStream.Length)
		{
			Debug.LogError("glTF header length does not match file length");
			return false;
		}
		return true;
	}

	private byte[] ReadChunk(Stream glbStream, OVRChunkType type)
	{
		uint chunkLength;
		if (ValidateChunk(glbStream, type, out chunkLength))
		{
			byte[] chunkBuffer = new byte[chunkLength];
			glbStream.Read(chunkBuffer, 0, (int)chunkLength);
			return chunkBuffer;
		}
		return null;
	}

	private bool ValidateChunk(Stream glbStream, OVRChunkType type, out uint chunkLength)
	{
		int uint32Size = sizeof(uint);
		byte[] buffer = new byte[uint32Size];
		glbStream.Read(buffer, 0, uint32Size);
		chunkLength = BitConverter.ToUInt32(buffer, 0);

		glbStream.Read(buffer, 0, uint32Size);
		uint chunkType = BitConverter.ToUInt32(buffer, 0);

		if (chunkType != (uint)type)
		{
			Debug.LogError("Read chunk does not match type.");
			return false;
		}
		return true;
	}

	private int LoadGLTF(bool supportAnimation, bool loadMips)
	{
		if (m_jsonData == null)
		{
			Debug.LogError("m_jsonData was null");
			return -1;
		}

		var scenes = m_jsonData["scenes"];
		if (scenes.Count == 0)
		{
			Debug.LogError("No valid scenes in this glTF.");
			return -1;
		}

		// Create GameObjects for each node in the model so that they can be referenced during processing
		var nodes = m_jsonData["nodes"].AsArray;
		for (int i = 0; i < nodes.Count; i++)
		{
			var jsonNode = m_jsonData["nodes"][i];
			GameObject go = new GameObject(jsonNode["name"]);
			m_Nodes.Add(go);
		}

		// Limit loading to just the first scene in the glTF
		var mainScene = scenes[0];
		var rootNodes = mainScene["nodes"].AsArray;

		// Load all nodes (some models like e.g. laptops use multiple nodes)
		foreach (JSONNode rootNode in rootNodes)
		{
			int rootNodeId = rootNode.AsInt;
			ProcessNode(m_jsonData["nodes"][rootNodeId], rootNodeId, loadMips);
		}

		if(supportAnimation)
			ProcessAnimations();

        return rootNodes[0].AsInt;
	}

	private void ProcessNode(JSONNode node, int nodeId, bool loadMips)
	{
		// Process the child nodes first
		var childNodes = node["children"];
		if (childNodes.Count > 0)
		{
			for (int i = 0; i < childNodes.Count; i++)
			{
				int childId = childNodes[i].AsInt;
				m_Nodes[childId].transform.SetParent(m_Nodes[nodeId].transform);
                ProcessNode(m_jsonData["nodes"][childId], childId, loadMips);
			}
		}

		string nodeName = node["name"].ToString();
		if (nodeName.Contains("batteryIndicator"))
		{
			GameObject.Destroy(m_Nodes[nodeId]);
			return;
		}

		if (node["mesh"] != null)
		{
			var meshId = node["mesh"].AsInt;
			OVRMeshData meshData = ProcessMesh(m_jsonData["meshes"][meshId], loadMips);

			if (node["skin"] != null)
			{
				var renderer = m_Nodes[nodeId].AddComponent<SkinnedMeshRenderer>();
				renderer.sharedMesh = meshData.mesh;
				renderer.sharedMaterial = meshData.material;

				var skinId = node["skin"].AsInt;
				ProcessSkin(m_jsonData["skins"][skinId], renderer);
			}
			else
			{
				var filter = m_Nodes[nodeId].AddComponent<MeshFilter>();
				filter.sharedMesh = meshData.mesh;
				var renderer = m_Nodes[nodeId].AddComponent<MeshRenderer>();
				renderer.sharedMaterial = meshData.material;
			}
		}

		var translation = node["translation"].AsArray;
		var rotation = node["rotation"].AsArray;
		var scale = node["scale"].AsArray;

		if (translation.Count > 0)
		{
			Vector3 position = new Vector3(
				translation[0] * GLTFToUnitySpace.x,
				translation[1] * GLTFToUnitySpace.y,
				translation[2] * GLTFToUnitySpace.z);
			m_Nodes[nodeId].transform.position = position;
		}

		if (rotation.Count > 0)
		{
			Vector3 rotationAxis = new Vector3(
				rotation[0] * GLTFToUnitySpace.x,
				rotation[1] * GLTFToUnitySpace.y,
				rotation[2] * GLTFToUnitySpace.z);
			rotationAxis *= -1.0f;
			m_Nodes[nodeId].transform.rotation = new Quaternion(rotationAxis.x, rotationAxis.y, rotationAxis.z, rotation[3]);
		}

		if (scale.Count > 0)
		{
			Vector3 scaleVec = new Vector3(scale[0], scale[1], scale[2]);
			m_Nodes[nodeId].transform.localScale = scaleVec;
		}
	}

	private OVRMeshData ProcessMesh(JSONNode meshNode, bool loadMips)
	{
		OVRMeshData meshData = new OVRMeshData();

		int totalVertexCount = 0;
		var primitives = meshNode["primitives"];
		int[] primitiveVertexCounts = new int[primitives.Count];
		for (int i = 0; i < primitives.Count; i++)
		{
			var jsonPrimitive = primitives[i];
			var jsonAttrbite = jsonPrimitive["attributes"]["POSITION"];
			var jsonAccessor = m_jsonData["accessors"][jsonAttrbite.AsInt];

			primitiveVertexCounts[i] = jsonAccessor["count"];
			totalVertexCount += primitiveVertexCounts[i];
		}

		int[][] indicies = new int[primitives.Count][];
		Vector3[] vertices = new Vector3[totalVertexCount];

		Vector3[] normals = null;
		if (primitives[0]["attributes"]["NORMAL"] != null)
		{
			normals = new Vector3[totalVertexCount];
		}

		Vector4[] tangents = null;
		if (primitives[0]["attributes"]["TANGENT"] != null)
		{
			tangents = new Vector4[totalVertexCount];
		}

		Vector2[] texcoords = null;
		if (primitives[0]["attributes"]["TEXCOORD_0"] != null)
		{
			texcoords = new Vector2[totalVertexCount];
		}

		Color[] colors = null;
		if (primitives[0]["attributes"]["COLOR_0"] != null)
		{
			colors = new Color[totalVertexCount];
		}

		BoneWeight[] boneWeights = null;
		if (primitives[0]["attributes"]["WEIGHTS_0"] != null)
		{
			boneWeights = new BoneWeight[totalVertexCount];
		}

		// Begin async processing of material and its texture
		OVRMaterialData matData = default(OVRMaterialData);
		Task transcodeTask = null;
		var jsonMaterial = primitives[0]["material"];
		if (jsonMaterial != null)
		{
			matData = ProcessMaterial(jsonMaterial.AsInt);
			matData.texture = ProcessTexture(matData.textureId);
			transcodeTask = Task.Run(() => { TranscodeTexture(ref matData.texture); });
		}

		int vertexOffset = 0;
		for (int i = 0; i < primitives.Count; i++)
		{
			var jsonPrimitive = primitives[i];

			int indicesAccessorId = jsonPrimitive["indices"].AsInt;
			var jsonAccessor = m_jsonData["accessors"][indicesAccessorId];
			OVRGLTFAccessor indicesReader = new OVRGLTFAccessor(jsonAccessor, m_jsonData);

			indicies[i] = new int[indicesReader.GetDataCount()];
			indicesReader.ReadAsInt(m_binaryChunk, ref indicies[i], 0);
			FlipTraingleIndices(ref indicies[i]);

			var jsonAttribute = jsonPrimitive["attributes"]["POSITION"];
			if (jsonAttribute != null)
			{
				jsonAccessor = m_jsonData["accessors"][jsonAttribute.AsInt];
				OVRGLTFAccessor dataReader = new OVRGLTFAccessor(jsonAccessor, m_jsonData);
				dataReader.ReadAsVector3(m_binaryChunk, ref vertices, vertexOffset, GLTFToUnitySpace);
			}

			jsonAttribute = jsonPrimitive["attributes"]["NORMAL"];
			if (jsonAttribute != null)
			{
				jsonAccessor = m_jsonData["accessors"][jsonAttribute.AsInt];
				OVRGLTFAccessor dataReader = new OVRGLTFAccessor(jsonAccessor, m_jsonData);
				dataReader.ReadAsVector3(m_binaryChunk, ref normals, vertexOffset, GLTFToUnitySpace);
			}

			jsonAttribute = jsonPrimitive["attributes"]["TANGENT"];
			if (jsonAttribute != null)
			{
				jsonAccessor = m_jsonData["accessors"][jsonAttribute.AsInt];
				OVRGLTFAccessor dataReader = new OVRGLTFAccessor(jsonAccessor, m_jsonData);
				dataReader.ReadAsVector4(m_binaryChunk, ref tangents, vertexOffset, GLTFToUnityTangent);
			}

			jsonAttribute = jsonPrimitive["attributes"]["TEXCOORD_0"];
			if (jsonAttribute != null)
			{
				jsonAccessor = m_jsonData["accessors"][jsonAttribute.AsInt];
				OVRGLTFAccessor dataReader = new OVRGLTFAccessor(jsonAccessor, m_jsonData);
				dataReader.ReadAsVector2(m_binaryChunk, ref texcoords, vertexOffset);
			}

			jsonAttribute = jsonPrimitive["attributes"]["COLOR_0"];
			if (jsonAttribute != null)
			{
				jsonAccessor = m_jsonData["accessors"][jsonAttribute.AsInt];
				OVRGLTFAccessor dataReader = new OVRGLTFAccessor(jsonAccessor, m_jsonData);
				dataReader.ReadAsColor(m_binaryChunk, ref colors, vertexOffset);
			}

			jsonAttribute = jsonPrimitive["attributes"]["WEIGHTS_0"];
			if (jsonAttribute != null)
			{
				jsonAccessor = m_jsonData["accessors"][jsonAttribute.AsInt];
				OVRGLTFAccessor weightReader = new OVRGLTFAccessor(jsonAccessor, m_jsonData);

				var jointAttribute = jsonPrimitive["attributes"]["JOINTS_0"];
				var jointAccessor = m_jsonData["accessors"][jointAttribute.AsInt];
				OVRGLTFAccessor jointReader = new OVRGLTFAccessor(jointAccessor, m_jsonData);

				Vector4[] weights = new Vector4[weightReader.GetDataCount()];
				Vector4[] joints = new Vector4[jointReader.GetDataCount()];

				weightReader.ReadAsBoneWeights(m_binaryChunk, ref weights, 0);
				jointReader.ReadAsVector4(m_binaryChunk, ref joints, 0, Vector4.one);

				for (int w = 0; w < weights.Length; w++)
				{
					boneWeights[vertexOffset + w].boneIndex0 = (int)joints[w].x;
					boneWeights[vertexOffset + w].boneIndex1 = (int)joints[w].y;
					boneWeights[vertexOffset + w].boneIndex2 = (int)joints[w].z;
					boneWeights[vertexOffset + w].boneIndex3 = (int)joints[w].w;

					boneWeights[vertexOffset + w].weight0 = weights[w].x;
					boneWeights[vertexOffset + w].weight1 = weights[w].y;
					boneWeights[vertexOffset + w].weight2 = weights[w].z;
					boneWeights[vertexOffset + w].weight3 = weights[w].w;
				}
			}

			vertexOffset += primitiveVertexCounts[i];
		}

		Mesh mesh = new Mesh();
		mesh.vertices = vertices;
		mesh.normals = normals;
		mesh.tangents = tangents;
		mesh.colors = colors;
		mesh.uv = texcoords;
		mesh.boneWeights = boneWeights;
		mesh.subMeshCount = primitives.Count;

		int baseVertex = 0;
		for(int i = 0; i < primitives.Count; i++)
		{
			mesh.SetIndices(indicies[i], MeshTopology.Triangles, i, false, baseVertex);
			baseVertex += primitiveVertexCounts[i];
		}

		mesh.RecalculateBounds();
		meshData.mesh = mesh;

		if (transcodeTask != null)
		{
			transcodeTask.Wait();
			meshData.material = CreateUnityMaterial(matData, loadMips);
		}
		return meshData;
	}

	private static void FlipTraingleIndices(ref int[] indices)
	{
		for (int i = 0; i < indices.Length; i += 3)
		{
			int a = indices[i];
			indices[i] = indices[i + 2];
			indices[i + 2] = a;
		}
	}

	private void ProcessSkin(JSONNode skinNode, SkinnedMeshRenderer renderer)
	{
		Matrix4x4[] inverseBindMatrices = null;
		if (skinNode["inverseBindMatrices"] != null)
		{
			int inverseBindMatricesId = skinNode["inverseBindMatrices"].AsInt;
			var jsonInverseBindMatrices = m_jsonData["accessors"][inverseBindMatricesId];

			OVRGLTFAccessor dataReader = new OVRGLTFAccessor(jsonInverseBindMatrices, m_jsonData);
			inverseBindMatrices = new Matrix4x4[dataReader.GetDataCount()];
			dataReader.ReadAsMatrix4x4(m_binaryChunk, ref inverseBindMatrices, 0, GLTFToUnitySpace);
		}

		if (skinNode["skeleton"] != null)
		{
			var skeletonRootId = skinNode["skeleton"].AsInt;
			renderer.rootBone = m_Nodes[skeletonRootId].transform;
		}

		Transform[] bones = null;
		if (skinNode["joints"] != null)
		{
			var joints = skinNode["joints"].AsArray;

			bones = new Transform[joints.Count];
			for (int i = 0; i < joints.Count; i++)
			{
				bones[i] = m_Nodes[joints[i]].transform;
			}
		}

		renderer.sharedMesh.bindposes = inverseBindMatrices;
		renderer.bones = bones;
	}

	private OVRMaterialData ProcessMaterial(int matId)
	{
		OVRMaterialData matData = new OVRMaterialData();

		var jsonMaterial = m_jsonData["materials"][matId];

		var jsonAlphaMode = jsonMaterial["alphaMode"];
		bool alphaBlendMode = jsonAlphaMode != null && jsonAlphaMode.Value == "BLEND";

		var jsonPbrDetails = jsonMaterial["pbrMetallicRoughness"];

		var jsonBaseColor = jsonPbrDetails["baseColorTexture"];
		if (jsonBaseColor != null)
		{
			int textureId = jsonBaseColor["index"].AsInt;
			matData.textureId = textureId;
		}
		else
		{
			var jsonTextrure = jsonMaterial["emissiveTexture"];
			if (jsonTextrure != null)
			{
				int textureId = jsonTextrure["index"].AsInt;
				matData.textureId = textureId;
			}
		}

		matData.shader = alphaBlendMode ? m_AlphaBlendShader : m_Shader;
		return matData;
	}

	private OVRTextureData ProcessTexture(int textureId)
	{
		var jsonTexture = m_jsonData["textures"][textureId];

		int imageSource = -1;
		var jsonExtensions = jsonTexture["extensions"];
		if (jsonExtensions != null)
		{
			var baisuExtension = jsonExtensions["KHR_texture_basisu"];
			if (baisuExtension != null)
			{
				imageSource = baisuExtension["source"].AsInt;
			}
		}
		else
		{
			imageSource = jsonTexture["source"].AsInt;
		}
		var jsonSource = m_jsonData["images"][imageSource];

		int sampler = jsonTexture["sampler"].AsInt;
		var jsonSampler = m_jsonData["samplers"][sampler];

		int bufferViewId = jsonSource["bufferView"].AsInt;
		var jsonBufferView = m_jsonData["bufferViews"][bufferViewId];
		OVRGLTFAccessor dataReader = new OVRGLTFAccessor(jsonBufferView, m_jsonData, true);

		OVRTextureData textureData = new OVRTextureData();
		if (jsonSource["mimeType"].Value == "image/ktx2")
		{
			textureData.data = dataReader.ReadAsKtxTexture(m_binaryChunk);
			textureData.format = OVRTextureFormat.KTX2;
		}
		else
		{
			Debug.LogWarning("Unsupported image mimeType.");
		}
		return textureData;
	}

	private void TranscodeTexture(ref OVRTextureData textureData)
	{
		if (textureData.format == OVRTextureFormat.KTX2)
		{
			OVRKtxTexture.Load(textureData.data, ref textureData);
		}
		else
		{
			Debug.LogWarning("Only KTX2 textures can be trascoded.");
		}
	}

	private Material CreateUnityMaterial(OVRMaterialData matData, bool loadMips)
	{
		Material mat = new Material(matData.shader);

		if (matData.texture.format == OVRTextureFormat.KTX2)
		{
			Texture2D texture;
			texture = new Texture2D(matData.texture.width, matData.texture.height, matData.texture.transcodedFormat, loadMips);
			texture.LoadRawTextureData(matData.texture.data);
			ApplyTextureQuality(m_TextureQuality, ref texture);
			if (loadMips && mat.HasProperty("_MainTexMMBias"))
				mat.SetFloat("_MainTexMMBias", m_TextureMipmapBias);
			texture.Apply(updateMipmaps: false, makeNoLongerReadable: true);
			mat.mainTexture = texture;
		}

		return mat;
	}

	private OVRGLTFInputNode GetInputNodeType(string name)
	{
		foreach (var item in InputNodeNameMap)
		{
			if (name.Contains(item.Key))
			{
				return item.Value;
			}
		}
		return OVRGLTFInputNode.None;
	}

	private void ProcessAnimations()
	{
		Dictionary<int, OVRGLTFInputNode> inputNodeIDTypeMap = new Dictionary<int, OVRGLTFInputNode>();
		for(int id=0; id<m_Nodes.Count; id++)
		{
			GameObject obj = m_Nodes[id];
			OVRGLTFInputNode inputNodeType = GetInputNodeType(obj.name);
			if (inputNodeType == OVRGLTFInputNode.None)
				continue;
			inputNodeIDTypeMap.Add(id, inputNodeType);
		}

		var animations = m_jsonData["animations"];
		foreach (JSONNode animation in animations.AsArray)
		{
			//We don't need animation name at this moment
			//string name = animation["name"].ToString();
			var channels = animation["channels"].AsArray;
			foreach (JSONNode channel in channels)
			{
				int nodeId = channel["target"]["node"].AsInt;
				OVRGLTFInputNode inputNodeType = inputNodeIDTypeMap[nodeId];
				if (!m_AnimationNodes.ContainsKey(inputNodeType))
				{
					m_AnimationNodes.Add(inputNodeType, new OVRGLTFAnimatinonNode(m_jsonData, m_binaryChunk, inputNodeType, m_Nodes[nodeId]));
				}
				m_AnimationNodes[inputNodeType].AddChannel(channel, animation["samplers"]);
			}
		}
	}
}
