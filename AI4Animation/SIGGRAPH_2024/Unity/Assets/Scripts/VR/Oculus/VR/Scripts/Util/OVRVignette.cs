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
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// A component to apply a Colored vignette effect to the camera
/// </summary>
[RequireComponent(typeof(Camera))]
[ExecuteInEditMode]
public class OVRVignette : MonoBehaviour {

	/// <summary>
	/// Controls the number of triangles in the vignette mesh.
	/// </summary>
	public enum MeshComplexityLevel
	{
		VerySimple,
		Simple,
		Normal,
		Detailed,
		VeryDetailed
	}

	/// <summary>
	/// Controls the falloff appearance.
	/// </summary>
	public enum FalloffType
	{
		Linear,
		Quadratic
	}

	private static readonly string QUADRATIC_FALLOFF = "QUADRATIC_FALLOFF";

	[SerializeField]
	[HideInInspector]
	private Shader VignetteShader;

	// These are only used at startup.
	[SerializeField]
	[Tooltip("Controls the number of triangles used for the vignette mesh." +
		" Normal is best for most purposes.")]
	private MeshComplexityLevel MeshComplexity = MeshComplexityLevel.Normal;
	[SerializeField]
	[Tooltip("Controls how the falloff looks.")]
	private FalloffType Falloff = FalloffType.Linear;

	// These can be controlled dynamically at runtime
	[Tooltip("The Vertical FOV of the vignette")]
	public float VignetteFieldOfView = 60;
	[Tooltip("The Aspect ratio of the vignette controls the " +
		"Horizontal FOV. (Larger numbers are wider)")]
	public float VignetteAspectRatio = 1f;
	[Tooltip("The width of the falloff for the vignette in degrees")]
	public float VignetteFalloffDegrees = 10f;
	[ColorUsage(false)]
	[Tooltip("The color of the vignette. Alpha value is ignored")]
	public Color VignetteColor;

	private Camera _Camera;
	private MeshFilter _OpaqueMeshFilter;
	private MeshFilter _TransparentMeshFilter;
	private MeshRenderer _OpaqueMeshRenderer;
	private MeshRenderer _TransparentMeshRenderer;

	private Mesh _OpaqueMesh;
	private Mesh _TransparentMesh;
	private Material _OpaqueMaterial;
	private Material _TransparentMaterial;

	private int _ShaderScaleAndOffset0Property;
	private int _ShaderScaleAndOffset1Property;

	private Vector4[] _TransparentScaleAndOffset0 = new Vector4[2];
	private Vector4[] _TransparentScaleAndOffset1 = new Vector4[2];
	private Vector4[] _OpaqueScaleAndOffset0 = new Vector4[2];
	private Vector4[] _OpaqueScaleAndOffset1 = new Vector4[2];

	private bool _OpaqueVignetteVisible = false;
	private bool _TransparentVignetteVisible = false;

#if UNITY_EDITOR
	// in the editor, allow these to be changed at runtime
	private MeshComplexityLevel _InitialMeshComplexity;
	private FalloffType _InitialFalloff;
#endif

	private int GetTriangleCount()
	{
		switch(MeshComplexity)
		{
			case MeshComplexityLevel.VerySimple: return 32;
			case MeshComplexityLevel.Simple: return 64;
			case MeshComplexityLevel.Normal: return 128;
			case MeshComplexityLevel.Detailed: return 256;
			case MeshComplexityLevel.VeryDetailed: return 512;
			default: return 128;
		}
	}

	private void BuildMeshes()
	{
#if UNITY_EDITOR
		_InitialMeshComplexity = MeshComplexity;
#endif
		int triangleCount = GetTriangleCount();

		Vector3[] innerVerts = new Vector3[triangleCount];
		Vector2[] innerUVs = new Vector2[triangleCount];
		Vector3[] outerVerts = new Vector3[triangleCount];
		Vector2[] outerUVs = new Vector2[triangleCount];
		int[] tris = new int[triangleCount * 3];
		for (int i = 0; i < triangleCount; i += 2)
		{
			float angle = 2 * i * Mathf.PI / triangleCount;

			float x = Mathf.Cos(angle);
			float y = Mathf.Sin(angle);

			outerVerts[i] = new Vector3(x, y, 0);
			outerVerts[i + 1] = new Vector3(x, y, 0);
			outerUVs[i] = new Vector2(0, 1);
			outerUVs[i + 1] = new Vector2(1, 1);

			innerVerts[i] = new Vector3(x, y, 0);
			innerVerts[i + 1] = new Vector3(x, y, 0);
			innerUVs[i] = new Vector2(0, 1);
			innerUVs[i + 1] = new Vector2(1, 0);

			int ti = i * 3;
			tris[ti] = i;
			tris[ti + 1] = i + 1;
			tris[ti + 2] = (i + 2) % triangleCount;
			tris[ti + 3] = i + 1;
			tris[ti + 4] = (i + 3) % triangleCount;
			tris[ti + 5] = (i + 2) % triangleCount;
		}

		if (_OpaqueMesh != null)
		{
			DestroyImmediate(_OpaqueMesh);
		}

		if (_TransparentMesh != null)
		{
			DestroyImmediate(_TransparentMesh);
		}

		_OpaqueMesh = new Mesh()
		{
			name = "Opaque Vignette Mesh",
			hideFlags = HideFlags.HideAndDontSave
		};
		_TransparentMesh = new Mesh()
		{
			name = "Transparent Vignette Mesh",
			hideFlags = HideFlags.HideAndDontSave
		};

		_OpaqueMesh.vertices = outerVerts;
		_OpaqueMesh.uv = outerUVs;
		_OpaqueMesh.triangles = tris;
		_OpaqueMesh.UploadMeshData(true);
		_OpaqueMesh.bounds = new Bounds(Vector3.zero, Vector3.one * 10000);
		_OpaqueMeshFilter.sharedMesh = _OpaqueMesh;

		_TransparentMesh.vertices = innerVerts;
		_TransparentMesh.uv = innerUVs;
		_TransparentMesh.triangles = tris;
		_TransparentMesh.UploadMeshData(true);
		_TransparentMesh.bounds = new Bounds(Vector3.zero, Vector3.one * 10000);
		_TransparentMeshFilter.sharedMesh = _TransparentMesh;
	}

	private void BuildMaterials()
	{
#if UNITY_EDITOR
		_InitialFalloff = Falloff;
#endif
		if (VignetteShader == null)
		{
			VignetteShader = Shader.Find("Oculus/OVRVignette");
		}
		if (VignetteShader == null)
		{
			Debug.LogError("Could not find Vignette Shader! Vignette will not be drawn!");
			return;
		}

		if (_OpaqueMaterial == null)
		{
			_OpaqueMaterial = new Material(VignetteShader)
			{
				name = "Opaque Vignette Material",
				hideFlags = HideFlags.HideAndDontSave,
				renderQueue = (int)RenderQueue.Background
			};
			_OpaqueMaterial.SetFloat("_BlendSrc", (float)BlendMode.One);
			_OpaqueMaterial.SetFloat("_BlendDst", (float)BlendMode.Zero);
			_OpaqueMaterial.SetFloat("_ZWrite", 1);
		}
		_OpaqueMeshRenderer.sharedMaterial = _OpaqueMaterial;

		if (_TransparentMaterial == null)
		{
			_TransparentMaterial = new Material(VignetteShader)
			{
				name = "Transparent Vignette Material",
				hideFlags = HideFlags.HideAndDontSave,
				renderQueue = (int)RenderQueue.Overlay
			};

			_TransparentMaterial.SetFloat("_BlendSrc", (float)BlendMode.SrcAlpha);
			_TransparentMaterial.SetFloat("_BlendDst", (float)BlendMode.OneMinusSrcAlpha);
			_TransparentMaterial.SetFloat("_ZWrite", 0);
		}

		if (Falloff == FalloffType.Quadratic)
		{
			_TransparentMaterial.EnableKeyword(QUADRATIC_FALLOFF);
		}
		else
		{
			_TransparentMaterial.DisableKeyword(QUADRATIC_FALLOFF);
		}
		_TransparentMeshRenderer.sharedMaterial = _TransparentMaterial;
	}

	private void OnEnable()
	{
#if UNITY_2019_1_OR_NEWER
		RenderPipelineManager.beginCameraRendering += OnBeginCameraRendering;
#elif UNITY_2018_1_OR_NEWER
		UnityEngine.Experimental.Rendering.RenderPipeline.beginCameraRendering += OnBeginCameraRendering;
#endif
	}

	private void OnDisable()
	{
#if UNITY_2019_1_OR_NEWER
		RenderPipelineManager.beginCameraRendering -= OnBeginCameraRendering;
#elif UNITY_2018_1_OR_NEWER
		UnityEngine.Experimental.Rendering.RenderPipeline.beginCameraRendering -= OnBeginCameraRendering;
#endif
		DisableRenderers();
	}

	private void Awake()
	{
		_Camera = GetComponent<Camera>();
		_ShaderScaleAndOffset0Property = Shader.PropertyToID("_ScaleAndOffset0");
		_ShaderScaleAndOffset1Property = Shader.PropertyToID("_ScaleAndOffset1");

		GameObject opaqueObject = new GameObject("Opaque Vignette") { hideFlags = HideFlags.HideAndDontSave };
		opaqueObject.transform.SetParent(_Camera.transform, false);
		_OpaqueMeshFilter = opaqueObject.AddComponent<MeshFilter>();
		_OpaqueMeshRenderer = opaqueObject.AddComponent<MeshRenderer>();

		_OpaqueMeshRenderer.receiveShadows = false;
		_OpaqueMeshRenderer.shadowCastingMode = ShadowCastingMode.Off;
		_OpaqueMeshRenderer.lightProbeUsage = LightProbeUsage.Off;
		_OpaqueMeshRenderer.reflectionProbeUsage = ReflectionProbeUsage.Off;
		_OpaqueMeshRenderer.allowOcclusionWhenDynamic = false;
		_OpaqueMeshRenderer.enabled = false;

		GameObject transparentObject = new GameObject("Transparent Vignette") { hideFlags = HideFlags.HideAndDontSave };
		transparentObject.transform.SetParent(_Camera.transform, false);
		_TransparentMeshFilter = transparentObject.AddComponent<MeshFilter>();
		_TransparentMeshRenderer = transparentObject.AddComponent<MeshRenderer>();

		_TransparentMeshRenderer.receiveShadows = false;
		_TransparentMeshRenderer.shadowCastingMode = ShadowCastingMode.Off;
		_TransparentMeshRenderer.lightProbeUsage = LightProbeUsage.Off;
		_TransparentMeshRenderer.reflectionProbeUsage = ReflectionProbeUsage.Off;
		_TransparentMeshRenderer.allowOcclusionWhenDynamic = false;
		_TransparentMeshRenderer.enabled = false;

		BuildMeshes();
		BuildMaterials();
	}

	private void GetTanFovAndOffsetForStereoEye(Camera.StereoscopicEye eye, out float tanFovX, out float tanFovY, out float offsetX, out float offsetY)
	{
		var pt = _Camera.GetStereoProjectionMatrix(eye).transpose;

		var right = pt * new Vector4(-1, 0, 0, 1);
		var left = pt * new Vector4(1, 0, 0, 1);
		var up = pt * new Vector4(0, -1, 0, 1);
		var down = pt * new Vector4(0, 1, 0, 1);

		float rightTanFovX = right.z / right.x;
		float leftTanFovX = left.z / left.x;
		float upTanFovY = up.z / up.y;
		float downTanFovY = down.z / down.y;

		offsetX = -(rightTanFovX + leftTanFovX) / 2;
		offsetY = -(upTanFovY + downTanFovY) / 2;

		tanFovX = (rightTanFovX - leftTanFovX) / 2;
		tanFovY = (upTanFovY - downTanFovY) / 2;
	}

	private void GetTanFovAndOffsetForMonoEye(out float tanFovX, out float tanFovY, out float offsetX, out float offsetY)
	{
		// When calculating from Unity's camera fields, this is the calculation used.
		// We can't use this for stereo eyes because VR projection matrices are usually asymmetric.
		tanFovY = Mathf.Tan(Mathf.Deg2Rad * _Camera.fieldOfView * 0.5f);
		tanFovX = tanFovY * _Camera.aspect;
		offsetX = 0f;
		offsetY = 0f;
	}

	private bool VisibilityTest(float scaleX, float scaleY, float offsetX, float offsetY)
	{
		// because the corners of our viewport are the furthest from the center of our vignette,
		// we only need to test that the farthest corner is outside the vignette ring.
		return new Vector2((1 + Mathf.Abs(offsetX)) / scaleX, (1 + Mathf.Abs(offsetY)) / scaleY).sqrMagnitude > 1.0f;
	}

	private void Update()
	{
#if UNITY_EDITOR
		if (MeshComplexity != _InitialMeshComplexity)
		{
			// rebuild meshes
			BuildMeshes();
		}

		if(Falloff != _InitialFalloff)
		{
			// rebuild materials
			BuildMaterials();
		}
#endif

		// The opaque material could not be created, so just return
		if (_OpaqueMaterial == null)
		{
			return;
		}

		float tanInnerFovY = Mathf.Tan(VignetteFieldOfView * Mathf.Deg2Rad * 0.5f);
		float tanInnerFovX = tanInnerFovY * VignetteAspectRatio;
		float tanMiddleFovX = Mathf.Tan((VignetteFieldOfView + VignetteFalloffDegrees) * Mathf.Deg2Rad * 0.5f);
		float tanMiddleFovY = tanMiddleFovX * VignetteAspectRatio;

		_TransparentVignetteVisible = false;
		_OpaqueVignetteVisible = false;

		for (int i = 0; i < 2; i++)
		{
			float tanFovX, tanFovY, offsetX, offsetY;
			if (_Camera.stereoEnabled)
			{
				GetTanFovAndOffsetForStereoEye((Camera.StereoscopicEye)i, out tanFovX, out tanFovY, out offsetX, out offsetY);
			}
			else
			{
				GetTanFovAndOffsetForMonoEye(out tanFovX, out tanFovY, out offsetX, out offsetY);
			}

			float borderScale = new Vector2((1 + Mathf.Abs(offsetX)) / VignetteAspectRatio, 1 + Mathf.Abs(offsetY)).magnitude * 1.01f;

			float innerScaleX = tanInnerFovX / tanFovX;
			float innerScaleY = tanInnerFovY / tanFovY;
			float middleScaleX = tanMiddleFovX / tanFovX;
			float middleScaleY = tanMiddleFovY / tanFovY;
			float outerScaleX = borderScale * VignetteAspectRatio;
			float outerScaleY = borderScale;

			// test for visibility.
			_TransparentVignetteVisible |= VisibilityTest(innerScaleX, innerScaleY, offsetX, offsetY);
			_OpaqueVignetteVisible |= VisibilityTest(middleScaleX, middleScaleY, offsetX, offsetY);

			_OpaqueScaleAndOffset0[i] = new Vector4(outerScaleX, outerScaleY, offsetX, offsetY);
			_OpaqueScaleAndOffset1[i] = new Vector4(middleScaleX, middleScaleY, offsetX, offsetY);
			_TransparentScaleAndOffset0[i] = new Vector4(middleScaleX, middleScaleY, offsetX, offsetY);
			_TransparentScaleAndOffset1[i] = new Vector4(innerScaleX, innerScaleY, offsetX, offsetY);
		}

		// if the vignette falloff is less than or equal to zero, we don't need to draw
		// the transparent mesh.
		_TransparentVignetteVisible &= VignetteFalloffDegrees > 0.0f;

		_OpaqueMaterial.SetVectorArray(_ShaderScaleAndOffset0Property, _OpaqueScaleAndOffset0);
		_OpaqueMaterial.SetVectorArray(_ShaderScaleAndOffset1Property, _OpaqueScaleAndOffset1);
		_OpaqueMaterial.color = VignetteColor;
		_TransparentMaterial.SetVectorArray(_ShaderScaleAndOffset0Property, _TransparentScaleAndOffset0);
		_TransparentMaterial.SetVectorArray(_ShaderScaleAndOffset1Property, _TransparentScaleAndOffset1);
		_TransparentMaterial.color = VignetteColor;
	}

	private void EnableRenderers()
	{
		_OpaqueMeshRenderer.enabled = _OpaqueVignetteVisible;
		_TransparentMeshRenderer.enabled = _TransparentVignetteVisible;
	}

	private void DisableRenderers()
	{
		_OpaqueMeshRenderer.enabled = false;
		_TransparentMeshRenderer.enabled = false;
	}

	// Objects are enabled on pre cull and disabled on post render so they only draw in this camera
	private void OnPreCull()
	{
		EnableRenderers();
	}

	private void OnPostRender()
	{
		DisableRenderers();
	}

#if UNITY_2019_1_OR_NEWER
	private void OnBeginCameraRendering(ScriptableRenderContext context, Camera camera)
#else
	private void OnBeginCameraRendering(Camera camera)
#endif
	{
		if (camera == _Camera)
		{
			EnableRenderers();
		}
		else
		{
			DisableRenderers();
		}
	}
}
