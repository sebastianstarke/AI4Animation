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

using UnityEngine;
using System;
using System.Collections;
using System.Runtime.InteropServices;
using System.Collections.Generic;

/// <summary>
/// Add OVROverlay script to an object with an optional mesh primitive
/// rendered as a TimeWarp overlay instead by drawing it into the eye buffer.
/// This will take full advantage of the display resolution and avoid double
/// resampling of the texture.
///
/// We support 3 types of Overlay shapes right now
///		1. Quad : This is most common overlay type , you render a quad in Timewarp space.
///		2. Cylinder: [Mobile Only][Experimental], Display overlay as partial surface of a cylinder
///			* The cylinder's center will be your game object's center
///			* We encoded the cylinder's parameters in transform.scale,
///				**[scale.z] is the radius of the cylinder
///				**[scale.y] is the height of the cylinder
///				**[scale.x] is the length of the arc of cylinder
///		* Limitations
///				**Only the half of the cylinder can be displayed, which means the arc angle has to be smaller than 180 degree,  [scale.x] / [scale.z] <= PI
///				**Your camera has to be inside of the inscribed sphere of the cylinder, the overlay will be faded out automatically when the camera is close to the inscribed sphere's surface.
///				**Translation only works correctly with vrDriver 1.04 or above
///		3. Cubemap: Display overlay as a cube map
///		4. OffcenterCubemap: [Mobile Only] Display overlay as a cube map with a texture coordinate offset
///			* The actually sampling will looks like [color = texture(cubeLayerSampler, normalize(direction) + offset)] instead of [color = texture( cubeLayerSampler, direction )]
///			* The extra center offset can be feed from transform.position
///			* Note: if transform.position's magnitude is greater than 1, which will cause some cube map pixel always invisible
///					Which is usually not what people wanted, we don't kill the ability for developer to do so here, but will warn out.
///		5. Equirect: Display overlay as a 360-degree equirectangular skybox.
/// </summary>
[ExecuteInEditMode]
public class OVROverlay : MonoBehaviour
{
#region Interface

	/// <summary>
	/// Determines the on-screen appearance of a layer.
	/// </summary>
	public enum OverlayShape
	{
		Quad = OVRPlugin.OverlayShape.Quad,
		Cylinder = OVRPlugin.OverlayShape.Cylinder,
		Cubemap = OVRPlugin.OverlayShape.Cubemap,
		OffcenterCubemap = OVRPlugin.OverlayShape.OffcenterCubemap,
		Equirect = OVRPlugin.OverlayShape.Equirect,
		ReconstructionPassthrough = OVRPlugin.OverlayShape.ReconstructionPassthrough,
		SurfaceProjectedPassthrough = OVRPlugin.OverlayShape.SurfaceProjectedPassthrough,
		Fisheye = OVRPlugin.OverlayShape.Fisheye,
		KeyboardHandsPassthrough = OVRPlugin.OverlayShape.KeyboardHandsPassthrough,
		KeyboardMaskedHandsPassthrough = OVRPlugin.OverlayShape.KeyboardMaskedHandsPassthrough,
	}

	/// <summary>
	/// Whether the layer appears behind or infront of other content in the scene.
	/// </summary>
	public enum OverlayType
	{
		None,
		Underlay,
		Overlay,
	};

	/// <summary>
	/// Specify overlay's type
	/// </summary>
	[Tooltip("Specify overlay's type")]
	public OverlayType currentOverlayType = OverlayType.Overlay;

	/// <summary>
	/// If true, the texture's content is copied to the compositor each frame.
	/// </summary>
	[Tooltip("If true, the texture's content is copied to the compositor each frame.")]
	public bool isDynamic = false;

	/// <summary>
	/// If true, the layer would be used to present protected content (e.g. HDCP), the content won't be shown in screenshots or recordings.
	/// </summary>
	[Tooltip("If true, the layer would be used to present protected content (e.g. HDCP), the content won't be shown in screenshots or recordings.")]
	public bool isProtectedContent = false;

	//Source and dest rects
	public Rect srcRectLeft = new Rect();
	public Rect srcRectRight = new Rect();
	public Rect destRectLeft = new Rect();
	public Rect destRectRight = new Rect();

	// Used to support legacy behavior where the top left was considered the origin
	public bool invertTextureRects = false;

	private OVRPlugin.TextureRectMatrixf textureRectMatrix = OVRPlugin.TextureRectMatrixf.zero;

	public bool overrideTextureRectMatrix = false;

	public bool overridePerLayerColorScaleAndOffset = false;

	public	Vector4 colorScale = Vector4.one;

	public Vector4 colorOffset = Vector4.zero;

	//Warning: Developers should only use this supersample setting if they absolutely have the budget and need for it. It is extremely expensive, and will not be relevant for most developers.
	public bool useExpensiveSuperSample = false;

	//Warning: Developers should only use this sharpening setting if they absolutely have the budget and need for it. It is extremely expensive, and will not be relevant for most developers.
	public bool useExpensiveSharpen = false;

	//Property that can hide overlays when required. Should be false when present, true when hidden.
	public bool hidden = false;


	/// <summary>
	/// If true, the layer will be created as an external surface. externalSurfaceObject contains the Surface object. It's effective only on Android.
	/// </summary>
	[Tooltip("If true, the layer will be created as an external surface. externalSurfaceObject contains the Surface object. It's effective only on Android.")]
	public bool isExternalSurface = false;

	/// <summary>
	/// The width which will be used to create the external surface. It's effective only on Android.
	/// </summary>
	[Tooltip("The width which will be used to create the external surface. It's effective only on Android.")]
	public int externalSurfaceWidth = 0;

	/// <summary>
	/// The height which will be used to create the external surface. It's effective only on Android.
	/// </summary>
	[Tooltip("The height which will be used to create the external surface. It's effective only on Android.")]
	public int externalSurfaceHeight = 0;

	/// <summary>
	/// The compositionDepth defines the order of the OVROverlays in composition. The overlay/underlay with smaller compositionDepth would be composited in the front of the overlay/underlay with larger compositionDepth.
	/// </summary>
	[Tooltip("The compositionDepth defines the order of the OVROverlays in composition. The overlay/underlay with smaller compositionDepth would be composited in the front of the overlay/underlay with larger compositionDepth.")]
	public int compositionDepth = 0;
	private int layerCompositionDepth = 0;

	/// <summary>
	/// The noDepthBufferTesting will stop layer's depth buffer compositing even if the engine has "Depth buffer sharing" enabled on Rift.
	/// </summary>
	[Tooltip("The noDepthBufferTesting will stop layer's depth buffer compositing even if the engine has \"Shared Depth Buffer\" enabled. The layer's ordering will be used instead which is determined by it's composition depth and overlay/underlay type.")]
	public bool noDepthBufferTesting = true;

	//Format corresponding to the source texture for this layer. sRGB by default, but can be modified if necessary
	public OVRPlugin.EyeTextureFormat layerTextureFormat = OVRPlugin.EyeTextureFormat.R8G8B8A8_sRGB;

	/// <summary>
	/// Specify overlay's shape
	/// </summary>
	[Tooltip("Specify overlay's shape")]
	public OverlayShape currentOverlayShape = OverlayShape.Quad;
	private OverlayShape prevOverlayShape = OverlayShape.Quad;

	/// <summary>
	/// The left- and right-eye Textures to show in the layer.
	/// \note If you need to change the texture on a per-frame basis, please use OverrideOverlayTextureInfo(..) to avoid caching issues.
	/// </summary>
	[Tooltip("The left- and right-eye Textures to show in the layer.")]
	public Texture[] textures = new Texture[] { null, null };

	[Tooltip("When checked, the texture is treated as if the alpha was already premultiplied")]
	public bool isAlphaPremultiplied = false;

	[Tooltip("When checked, the layer will use bicubic filtering")]
	public bool useBicubicFiltering = false;

	[Tooltip("When checked, the cubemap will retain the legacy rotation which was rotated 180 degrees around the Y axis comapred to Unity's definition of cubemaps. This setting will be deprecated in the near future, therefore it is recommended to fix the cubemap texture instead.")]
	public bool useLegacyCubemapRotation = false;

	[Tooltip("When checked, the layer will use efficient super sampling")]
	public bool useEfficientSupersample = false;

	[Tooltip("When checked, the layer will use efficient sharpen.  Must have anisotropic filtering and mipmaps enabled.")]
	public bool useEfficientSharpen = false;

	/// <summary>
	/// Preview the overlay in the editor using a mesh renderer.
	/// </summary>
	public bool previewInEditor {
		get {
			return _previewInEditor;
		}
		set {
			if (_previewInEditor != value) {
				_previewInEditor = value;
				SetupEditorPreview();
			}
		}
	}

	[SerializeField]
	internal bool _previewInEditor = false;

#if UNITY_EDITOR
	private GameObject previewObject;
#endif

	protected IntPtr[] texturePtrs = new IntPtr[] { IntPtr.Zero, IntPtr.Zero };

	/// <summary>
	/// The Surface object (Android only).
	/// </summary>
	public System.IntPtr externalSurfaceObject;

	public delegate void ExternalSurfaceObjectCreated();
	/// <summary>
	/// Will be triggered after externalSurfaceTextueObject get created.
	/// </summary>
	public ExternalSurfaceObjectCreated externalSurfaceObjectCreated;

	/// <summary>
	/// Use this function to set texture and texNativePtr when app is running
	/// GetNativeTexturePtr is a slow behavior, the value should be pre-cached
	/// </summary>
	public void OverrideOverlayTextureInfo(Texture srcTexture, IntPtr nativePtr, UnityEngine.XR.XRNode node)
	{
		int index = (node == UnityEngine.XR.XRNode.RightEye) ? 1 : 0;

		if (textures.Length <= index)
			return;

		textures[index] = srcTexture;
		texturePtrs[index] = nativePtr;

		isOverridePending = true;
	}

	protected bool isOverridePending;

	internal const int maxInstances = 15;
	public static OVROverlay[] instances = new OVROverlay[maxInstances];

	public int layerId { get; private set; } = 0; // The layer's internal handle in the compositor.

#endregion

	private static Material tex2DMaterial;
	private static Material cubeMaterial;

	private OVRPlugin.LayerLayout layout {
		get {
#if UNITY_ANDROID && !UNITY_EDITOR
			if (textures.Length == 2 && textures[1] != null && textures[1] != textures[0])
				return OVRPlugin.LayerLayout.Stereo;
#endif
			return OVRPlugin.LayerLayout.Mono;
		}
	}

	private struct LayerTexture {
		public Texture appTexture;
		public IntPtr appTexturePtr;
		public Texture[] swapChain;
		public IntPtr[] swapChainPtr;
	};
	private LayerTexture[] layerTextures;

	private OVRPlugin.LayerDesc layerDesc;
	private int stageCount = -1;

	private int layerIndex = -1; // Controls the composition order based on wake-up time.
	private GCHandle layerIdHandle;
	private IntPtr layerIdPtr = IntPtr.Zero;

	private int frameIndex = 0;
	private int prevFrameIndex = -1;

	private Renderer rend;


	private int texturesPerStage { get { return (layout == OVRPlugin.LayerLayout.Stereo) ? 2 : 1; } }

	private static bool NeedsTexturesForShape(OverlayShape shape)
	{
		return !IsPassthroughShape(shape);
	}

	private bool CreateLayer(int mipLevels, int sampleCount, OVRPlugin.EyeTextureFormat etFormat, int flags, OVRPlugin.Sizei size, OVRPlugin.OverlayShape shape)
	{
		if (!layerIdHandle.IsAllocated || layerIdPtr == IntPtr.Zero)
		{
			layerIdHandle = GCHandle.Alloc(layerId, GCHandleType.Pinned);
			layerIdPtr = layerIdHandle.AddrOfPinnedObject();
		}

		if (layerIndex == -1)
		{
			for (int i = 0; i < maxInstances; ++i)
			{
				if (instances[i] == null || instances[i] == this)
				{
					layerIndex = i;
					instances[i] = this;
					break;
				}
			}
		}


		bool needsSetup = (
			isOverridePending ||
			layerDesc.MipLevels != mipLevels ||
			layerDesc.SampleCount != sampleCount ||
			layerDesc.Format != etFormat ||
			layerDesc.Layout != layout ||
			layerDesc.LayerFlags != flags ||
			!layerDesc.TextureSize.Equals(size) ||
			layerDesc.Shape != shape ||
			layerCompositionDepth != compositionDepth)
			;

		if (!needsSetup)
			return false;

		OVRPlugin.LayerDesc desc = OVRPlugin.CalculateLayerDesc(shape, layout, size, mipLevels, sampleCount, etFormat, flags);


		OVRPlugin.EnqueueSetupLayer(desc, compositionDepth, layerIdPtr);


		layerId = (int)layerIdHandle.Target;

		if (layerId > 0)
		{
			layerDesc = desc;
			layerCompositionDepth = compositionDepth;
			if (isExternalSurface)
			{
				stageCount = 1;
			}
			else
			{
				stageCount = OVRPlugin.GetLayerTextureStageCount(layerId);
			}
		}

		isOverridePending = false;

		return true;
	}

	private bool CreateLayerTextures(bool useMipmaps, OVRPlugin.Sizei size, bool isHdr)
	{
		if (isExternalSurface)
		{
			if (externalSurfaceObject == System.IntPtr.Zero)
			{
				externalSurfaceObject = OVRPlugin.GetLayerAndroidSurfaceObject(layerId);
				if (externalSurfaceObject != System.IntPtr.Zero)
				{
					Debug.LogFormat("GetLayerAndroidSurfaceObject returns {0}", externalSurfaceObject);
					if (externalSurfaceObjectCreated != null)
					{
						externalSurfaceObjectCreated();
					}
				}
			}
			return false;
		}

		bool needsCopy = false;

		if (stageCount <= 0)
			return false;

		// For newer SDKs, blit directly to the surface that will be used in compositing.

		if (layerTextures == null)
			layerTextures = new LayerTexture[texturesPerStage];

		for (int eyeId = 0; eyeId < texturesPerStage; ++eyeId)
		{
			if (layerTextures[eyeId].swapChain == null)
				layerTextures[eyeId].swapChain = new Texture[stageCount];

			if (layerTextures[eyeId].swapChainPtr == null)
				layerTextures[eyeId].swapChainPtr = new IntPtr[stageCount];

			for (int stage = 0; stage < stageCount; ++stage)
			{
				Texture sc = layerTextures[eyeId].swapChain[stage];
				IntPtr scPtr = layerTextures[eyeId].swapChainPtr[stage];

				if (sc != null && scPtr != IntPtr.Zero && size.w == sc.width && size.h == sc.height)
					continue;

				if (scPtr == IntPtr.Zero)
					scPtr = OVRPlugin.GetLayerTexture(layerId, stage, (OVRPlugin.Eye)eyeId);

				if (scPtr == IntPtr.Zero)
					continue;

				var txFormat = (isHdr) ? TextureFormat.RGBAHalf : TextureFormat.RGBA32;

				if (currentOverlayShape != OverlayShape.Cubemap && currentOverlayShape != OverlayShape.OffcenterCubemap)
					sc = Texture2D.CreateExternalTexture(size.w, size.h, txFormat, useMipmaps, true, scPtr);
				else
					sc = Cubemap.CreateExternalTexture(size.w, txFormat, useMipmaps, scPtr);

				layerTextures[eyeId].swapChain[stage] = sc;
				layerTextures[eyeId].swapChainPtr[stage] = scPtr;

				needsCopy = true;
			}
		}

		return needsCopy;
	}

	private void DestroyLayerTextures()
	{
		if (isExternalSurface)
		{
			return;
		}

		for (int eyeId = 0; layerTextures != null && eyeId < texturesPerStage; ++eyeId)
		{
			if (layerTextures[eyeId].swapChain != null)
			{
				for (int stage = 0; stage < stageCount; ++stage)
					DestroyImmediate(layerTextures[eyeId].swapChain[stage]);
			}
		}

		layerTextures = null;
	}

	private void DestroyLayer()
	{
		if (layerIndex != -1)
		{
			// Turn off the overlay if it was on.
			OVRPlugin.EnqueueSubmitLayer(true, false, false, IntPtr.Zero, IntPtr.Zero, -1, 0, OVRPose.identity.ToPosef_Legacy(), Vector3.one.ToVector3f(), layerIndex, (OVRPlugin.OverlayShape)prevOverlayShape);
			instances[layerIndex] = null;
			layerIndex = -1;
		}

		if (layerIdPtr != IntPtr.Zero)
		{
			OVRPlugin.EnqueueDestroyLayer(layerIdPtr);
			layerIdPtr = IntPtr.Zero;
			layerIdHandle.Free();
			layerId = 0;
		}

		layerDesc = new OVRPlugin.LayerDesc();

		frameIndex = 0;
		prevFrameIndex = -1;
	}

	/// <summary>
	/// Sets the source and dest rects for both eyes. Source explains what portion of the source texture is used, and
	/// dest is what portion of the destination texture is rendered into.
	/// </summary>
	public void SetSrcDestRects(Rect srcLeft, Rect srcRight, Rect destLeft, Rect destRight)
	{
		srcRectLeft = srcLeft;
		srcRectRight = srcRight;
		destRectLeft = destLeft;
		destRectRight = destRight;
	}

	public void UpdateTextureRectMatrix()
	{
		// External surfaces are encoded with reversed UV's, so our texture rects are also inverted
		Rect srcRectLeftConverted = new Rect(srcRectLeft.x, isExternalSurface ^ invertTextureRects ? 1 - srcRectLeft.y - srcRectLeft.height : srcRectLeft.y, srcRectLeft.width, srcRectLeft.height);
		Rect srcRectRightConverted = new Rect(srcRectRight.x, isExternalSurface ^ invertTextureRects ? 1 - srcRectRight.y - srcRectRight.height : srcRectRight.y, srcRectRight.width, srcRectRight.height);
		Rect destRectLeftConverted = new Rect(destRectLeft.x, isExternalSurface ^ invertTextureRects ? 1 - destRectLeft.y - destRectLeft.height : destRectLeft.y, destRectLeft.width, destRectLeft.height);
		Rect destRectRightConverted = new Rect(destRectRight.x, isExternalSurface ^ invertTextureRects ? 1 - destRectRight.y - destRectRight.height : destRectRight.y, destRectRight.width, destRectRight.height);
		textureRectMatrix.leftRect = srcRectLeftConverted;
		textureRectMatrix.rightRect = srcRectRightConverted;

		// Fisheye layer requires a 0.5f offset for texture to be centered on the fisheye projection
		if (currentOverlayShape == OverlayShape.Fisheye)
		{
			destRectLeftConverted.x -= 0.5f;
			destRectLeftConverted.y -= 0.5f;
			destRectRightConverted.x -= 0.5f;
			destRectRightConverted.y -= 0.5f;
		}

		float leftWidthFactor = srcRectLeft.width / destRectLeft.width;
		float leftHeightFactor = srcRectLeft.height / destRectLeft.height;
		textureRectMatrix.leftScaleBias = new Vector4(leftWidthFactor, leftHeightFactor, srcRectLeftConverted.x - destRectLeftConverted.x * leftWidthFactor, srcRectLeftConverted.y - destRectLeftConverted.y * leftHeightFactor);
		float rightWidthFactor = srcRectRight.width / destRectRight.width;
		float rightHeightFactor = srcRectRight.height / destRectRight.height;
		textureRectMatrix.rightScaleBias = new Vector4(rightWidthFactor, rightHeightFactor, srcRectRightConverted.x - destRectRightConverted.x * rightWidthFactor, srcRectRightConverted.y - destRectRightConverted.y * rightHeightFactor);
	}

	public void SetPerLayerColorScaleAndOffset(Vector4 scale, Vector4 offset)
	{
		colorScale = scale;
		colorOffset = offset;
	}

	private bool LatchLayerTextures()
	{
		if (isExternalSurface)
		{
			return true;
		}

		for (int i = 0; i < texturesPerStage; ++i)
		{
			if (textures[i] != layerTextures[i].appTexture || layerTextures[i].appTexturePtr == IntPtr.Zero)
			{
				if (textures[i] != null)
				{
#if UNITY_EDITOR
					var assetPath = UnityEditor.AssetDatabase.GetAssetPath(textures[i]);
					var importer = (UnityEditor.TextureImporter)UnityEditor.TextureImporter.GetAtPath(assetPath);
					if (importer && importer.textureType != UnityEditor.TextureImporterType.Default)
					{
						Debug.LogError("Need Default Texture Type for overlay");
						return false;
					}
#endif
					var rt = textures[i] as RenderTexture;
					if (rt && !rt.IsCreated())
						rt.Create();

					layerTextures[i].appTexturePtr = (texturePtrs[i] != IntPtr.Zero) ? texturePtrs[i] : textures[i].GetNativeTexturePtr();

					if (layerTextures[i].appTexturePtr != IntPtr.Zero)
						layerTextures[i].appTexture = textures[i];
				}
			}

			if (currentOverlayShape == OverlayShape.Cubemap)
			{
				if (textures[i] as Cubemap == null)
				{
					Debug.LogError("Need Cubemap texture for cube map overlay");
					return false;
				}
			}
		}

#if !UNITY_ANDROID || UNITY_EDITOR
		if (currentOverlayShape == OverlayShape.OffcenterCubemap)
		{
			Debug.LogWarning("Overlay shape " + currentOverlayShape + " is not supported on current platform");
			return false;
		}
#endif

		if (layerTextures[0].appTexture == null || layerTextures[0].appTexturePtr == IntPtr.Zero)
			return false;

		return true;
	}

	private OVRPlugin.LayerDesc GetCurrentLayerDesc()
	{
		OVRPlugin.Sizei textureSize = new OVRPlugin.Sizei() { w = 0, h = 0 };

		if (isExternalSurface)
		{
			textureSize.w = externalSurfaceWidth;
			textureSize.h = externalSurfaceHeight;
		}
		else if (NeedsTexturesForShape(currentOverlayShape))
		{
			if (textures[0] == null)
			{
				Debug.LogWarning("textures[0] hasn't been set");
			}
			textureSize.w = textures[0] ? textures[0].width : 0;
			textureSize.h = textures[0] ? textures[0].height : 0;
		}

		OVRPlugin.LayerDesc newDesc = new OVRPlugin.LayerDesc() {
			Format = layerTextureFormat,
			LayerFlags = isExternalSurface ? 0 : (int)OVRPlugin.LayerFlags.TextureOriginAtBottomLeft,
			Layout = layout,
			MipLevels = 1,
			SampleCount = 1,
			Shape = (OVRPlugin.OverlayShape)currentOverlayShape,
			TextureSize = textureSize
		};

		var tex2D = textures[0] as Texture2D;
		if (tex2D != null)
		{
			if (tex2D.format == TextureFormat.RGBAHalf || tex2D.format == TextureFormat.RGBAFloat)
				newDesc.Format = OVRPlugin.EyeTextureFormat.R16G16B16A16_FP;

			newDesc.MipLevels = tex2D.mipmapCount;
		}

		var texCube = textures[0] as Cubemap;
		if (texCube != null)
		{
			if (texCube.format == TextureFormat.RGBAHalf || texCube.format == TextureFormat.RGBAFloat)
				newDesc.Format = OVRPlugin.EyeTextureFormat.R16G16B16A16_FP;

			newDesc.MipLevels = texCube.mipmapCount;
		}

		var rt = textures[0] as RenderTexture;
		if (rt != null)
		{
			newDesc.SampleCount = rt.antiAliasing;

			if (rt.format == RenderTextureFormat.ARGBHalf || rt.format == RenderTextureFormat.ARGBFloat || rt.format == RenderTextureFormat.RGB111110Float)
				newDesc.Format = OVRPlugin.EyeTextureFormat.R16G16B16A16_FP;
		}

		if (isProtectedContent)
		{
			newDesc.LayerFlags |= (int)OVRPlugin.LayerFlags.ProtectedContent;
		}

		if (isExternalSurface)
		{
			newDesc.LayerFlags |= (int)OVRPlugin.LayerFlags.AndroidSurfaceSwapChain;
		}

		if (useBicubicFiltering)
		{
			newDesc.LayerFlags |= (int)OVRPlugin.LayerFlags.BicubicFiltering;
		}

		return newDesc;
	}

	private Rect GetBlitRect(int eyeId)
	{
		if (texturesPerStage == 2)
		{
			return eyeId == 0 ? srcRectLeft : srcRectRight;
		}
		else
		{
			// Get intersection of both rects if we use the same texture for both eyes
			float minX = Mathf.Min(srcRectLeft.x, srcRectRight.x);
			float minY = Mathf.Min(srcRectLeft.y, srcRectRight.y);
			float maxX = Mathf.Max(srcRectLeft.x + srcRectLeft.width, srcRectRight.x + srcRectRight.width);
			float maxY = Mathf.Max(srcRectLeft.y + srcRectLeft.height, srcRectRight.y + srcRectRight.height);
			return new Rect(minX, minY, maxX - minX, maxY - minY);
		}
	}

	// A blit method that only draws into the specified rect by setting the viewport.
	private void BlitSubImage(Texture src, RenderTexture dst, Material mat, Rect rect)
	{
		var p = RenderTexture.active;
		RenderTexture.active = dst;

		// our rects are inverted
		rect.y = 1 - rect.y - rect.height;
		var viewport = new Rect(dst.width * rect.x, dst.height * rect.y, dst.width * rect.width, dst.height * rect.height);

		// do our blit using GL ops
		GL.PushMatrix();
		GL.Viewport(viewport);
		GL.LoadPixelMatrix(viewport.x, viewport.x + viewport.width, viewport.y, viewport.y + viewport.height);
		tex2DMaterial.mainTexture = src;
		tex2DMaterial.SetPass(0);

		GL.Begin(GL.TRIANGLES);
		GL.TexCoord(new Vector2(rect.x, rect.y));
		GL.Vertex3(viewport.x, viewport.y, 0);
		GL.TexCoord(new Vector2(rect.x, rect.y + rect.height * 2));
		GL.Vertex3(viewport.x, viewport.y + viewport.height * 2, 0);
		GL.TexCoord(new Vector2(rect.x + rect.width * 2, rect.y));
		GL.Vertex3(viewport.x + viewport.width * 2, viewport.y, 0);
		GL.End();
		GL.Flush();

		GL.PopMatrix();
		RenderTexture.active = p;
	}

	private bool PopulateLayer(int mipLevels, bool isHdr, OVRPlugin.Sizei size, int sampleCount, int stage)
	{
		if (isExternalSurface)
		{
			return true;
		}

		bool ret = false;

		RenderTextureFormat rtFormat = (isHdr) ? RenderTextureFormat.ARGBHalf : RenderTextureFormat.ARGB32;

		for (int eyeId = 0; eyeId < texturesPerStage; ++eyeId)
		{
			Texture et = layerTextures[eyeId].swapChain[stage];
			if (et == null)
				continue;

			for (int mip = 0; mip < mipLevels; ++mip)
			{
				bool dataIsLinear = isHdr || (QualitySettings.activeColorSpace == ColorSpace.Linear);

				var rt = textures[eyeId] as RenderTexture;
#if UNITY_ANDROID && !UNITY_EDITOR
				dataIsLinear = true; //HACK: Graphics.CopyTexture causes linear->srgb conversion on target write with D3D but not GLES.
#endif
				// PC requries premultiplied Alpha
				bool requiresPremultipliedAlpha = !Application.isMobilePlatform;

				bool linearToSRGB = !isHdr && dataIsLinear;
				// if the texture needs to be premultiplied, premultiply it unless its already premultiplied
				bool premultiplyAlpha = requiresPremultipliedAlpha && !isAlphaPremultiplied;

				bool bypassBlit = !linearToSRGB && !premultiplyAlpha && rt != null && rt.format == rtFormat;

				RenderTexture tempRTDst = null;

				if (!bypassBlit)
				{
					int width = size.w >> mip;
					if (width < 1) width = 1;
					int height = size.h >> mip;
					if (height < 1) height = 1;
					RenderTextureDescriptor descriptor = new RenderTextureDescriptor(width, height, rtFormat, 0);
					descriptor.msaaSamples = sampleCount;
					descriptor.useMipMap = true;
					descriptor.autoGenerateMips = false;
					descriptor.sRGB = false;

					tempRTDst = RenderTexture.GetTemporary(descriptor);

					if (!tempRTDst.IsCreated())
					{
						tempRTDst.Create();
					}

					tempRTDst.DiscardContents();

					Material blitMat = null;
					if (currentOverlayShape != OverlayShape.Cubemap && currentOverlayShape != OverlayShape.OffcenterCubemap)
					{
						blitMat = tex2DMaterial;
					}
					else
					{
						blitMat = cubeMaterial;
					}

					blitMat.SetInt("_linearToSrgb", linearToSRGB ? 1 : 0);
					blitMat.SetInt("_premultiply", premultiplyAlpha ? 1 : 0);
					blitMat.SetInt("_flip", OVRPlugin.nativeXrApi == OVRPlugin.XrApi.OpenXR ? 1 : 0);
				}

				if (currentOverlayShape != OverlayShape.Cubemap && currentOverlayShape != OverlayShape.OffcenterCubemap)
				{
					if (bypassBlit)
					{
						Graphics.CopyTexture(textures[eyeId], 0, mip, et, 0, mip);
					}
					else
					{
						if (overrideTextureRectMatrix)
						{
							BlitSubImage(textures[eyeId], tempRTDst, tex2DMaterial, GetBlitRect(eyeId));
						}
						else
						{
							Graphics.Blit(textures[eyeId], tempRTDst, tex2DMaterial);
						}
						//Resolve, decompress, swizzle, etc not handled by simple CopyTexture.
						Graphics.CopyTexture(tempRTDst, 0, 0, et, 0, mip);
					}
				}
				else // Cubemap
				{
					for (int face = 0; face < 6; ++face)
					{
						cubeMaterial.SetInt("_face", face);
						if (bypassBlit)
						{
							Graphics.CopyTexture(textures[eyeId], face, mip, et, face, mip);
						}
						else
						{
							//Resolve, decompress, swizzle, etc not handled by simple CopyTexture.
							Graphics.Blit(textures[eyeId], tempRTDst, cubeMaterial);
							Graphics.CopyTexture(tempRTDst, 0, 0, et, face, mip);
						}
					}
				}
				if (tempRTDst != null)
				{
					RenderTexture.ReleaseTemporary(tempRTDst);
				}

				ret = true;
			}
		}

		return ret;
	}

	private bool SubmitLayer(bool overlay, bool headLocked, bool noDepthBufferTesting, OVRPose pose, Vector3 scale, int frameIndex)
	{
		int rightEyeIndex = (texturesPerStage >= 2) ? 1 : 0;
		if (overrideTextureRectMatrix)
		{
			UpdateTextureRectMatrix();
		}
		bool noTextures = isExternalSurface || !NeedsTexturesForShape(currentOverlayShape);
		bool isOverlayVisible = OVRPlugin.EnqueueSubmitLayer(overlay, headLocked, noDepthBufferTesting,
			noTextures ? System.IntPtr.Zero : layerTextures[0].appTexturePtr,
			noTextures ? System.IntPtr.Zero : layerTextures[rightEyeIndex].appTexturePtr,
			layerId, frameIndex, pose.flipZ().ToPosef_Legacy(), scale.ToVector3f(), layerIndex, (OVRPlugin.OverlayShape)currentOverlayShape,
			overrideTextureRectMatrix, textureRectMatrix, overridePerLayerColorScaleAndOffset, colorScale, colorOffset,
			useExpensiveSuperSample, useBicubicFiltering, useEfficientSupersample, useEfficientSharpen, useExpensiveSharpen,
			hidden, isProtectedContent
			);
		prevOverlayShape = currentOverlayShape;

		return isOverlayVisible;
	}

	private void SetupEditorPreview()
	{
#if UNITY_EDITOR
			if (previewInEditor && previewObject == null)
			{
				previewObject = new GameObject();
				previewObject.hideFlags = HideFlags.HideAndDontSave;
				previewObject.transform.SetParent(this.transform, false);
				OVROverlayMeshGenerator generator = previewObject.AddComponent<OVROverlayMeshGenerator>();
				generator.SetOverlay(this);

			}
			else if (!previewInEditor && previewObject != null)
			{
				UnityEngine.Object.DestroyImmediate(previewObject);
				previewObject = null;
			}
#endif
	}

	public static bool IsPassthroughShape(OverlayShape shape)
	{
		return OVRPlugin.IsPassthroughShape((OVRPlugin.OverlayShape)shape);
	}

#region Unity Messages

	void Awake()
	{
		if (Application.isPlaying)
		{
			if (tex2DMaterial == null)
				tex2DMaterial = new Material(Shader.Find("Oculus/Texture2D Blit"));

			if (cubeMaterial == null)
				cubeMaterial = new Material(Shader.Find("Oculus/Cubemap Blit"));
		}

		rend = GetComponent<Renderer>();

		if (textures.Length == 0)
			textures = new Texture[] { null };

		// Backward compatibility
		if (rend != null && textures[0] == null)
			textures[0] = rend.sharedMaterial.mainTexture;

		SetupEditorPreview();
	}

	static public string OpenVROverlayKey { get { return "unity:" + Application.companyName + "." + Application.productName; } }
	private ulong OpenVROverlayHandle = OVR.OpenVR.OpenVR.k_ulOverlayHandleInvalid;

	void OnEnable()
	{
		if (OVRManager.OVRManagerinitialized)
			InitOVROverlay();

#if UNITY_EDITOR
		if (previewObject != null) {
			previewObject.SetActive(true);
		}
#endif
	}

	void InitOVROverlay()
	{
#if USING_XR_SDK_OPENXR
		if (!OVRPlugin.UnityOpenXR.Enabled)
		{
#endif
			if (!OVRManager.isHmdPresent)
			{
				enabled = false;
				return;
			}
#if USING_XR_SDK_OPENXR
		}
#endif

		constructedOverlayXRDevice = OVRManager.XRDevice.Unknown;
		if (OVRManager.loadedXRDevice == OVRManager.XRDevice.OpenVR)
		{
			OVR.OpenVR.CVROverlay overlay = OVR.OpenVR.OpenVR.Overlay;
			if (overlay != null)
			{
				OVR.OpenVR.EVROverlayError error = overlay.CreateOverlay(OpenVROverlayKey + transform.name, gameObject.name, ref OpenVROverlayHandle);
				if (error != OVR.OpenVR.EVROverlayError.None)
				{
					enabled = false;
					return;
				}
			}
			else
			{
				enabled = false;
				return;
			}
		}

		constructedOverlayXRDevice = OVRManager.loadedXRDevice;
		xrDeviceConstructed = true;
	}

	void OnDisable()
	{

#if UNITY_EDITOR
		if (previewObject != null) {
			previewObject.SetActive(false);
		}
#endif

		if ((gameObject.hideFlags & HideFlags.DontSaveInBuild) != 0)
			return;

		if (!OVRManager.OVRManagerinitialized)
			return;

		if (OVRManager.loadedXRDevice != constructedOverlayXRDevice)
			return;

		if (OVRManager.loadedXRDevice == OVRManager.XRDevice.Oculus)
		{
			DestroyLayerTextures();
			DestroyLayer();
		}
		else if (OVRManager.loadedXRDevice == OVRManager.XRDevice.OpenVR)
		{
			if (OpenVROverlayHandle != OVR.OpenVR.OpenVR.k_ulOverlayHandleInvalid)
			{
				OVR.OpenVR.CVROverlay overlay = OVR.OpenVR.OpenVR.Overlay;
				if (overlay != null)
				{
					overlay.DestroyOverlay(OpenVROverlayHandle);
				}
				OpenVROverlayHandle = OVR.OpenVR.OpenVR.k_ulOverlayHandleInvalid;
			}
		}
		constructedOverlayXRDevice = OVRManager.XRDevice.Unknown;
		xrDeviceConstructed = false;
	}

	void OnDestroy()
	{
		DestroyLayerTextures();
		DestroyLayer();

#if UNITY_EDITOR
		if (previewObject != null) {
			GameObject.DestroyImmediate(previewObject);
		}
#endif
	}

	void ComputePoseAndScale(ref OVRPose pose, ref Vector3 scale, ref bool overlay, ref bool headLocked)
	{
		Camera headCamera = Camera.main;

		overlay = (currentOverlayType == OverlayType.Overlay);
		headLocked = false;
		for (var t = transform; t != null && !headLocked; t = t.parent)
			headLocked |= (t == headCamera.transform);

		pose = (headLocked) ? transform.ToHeadSpacePose(headCamera) : transform.ToTrackingSpacePose(headCamera);
		scale = transform.lossyScale;
		for (int i = 0; i < 3; ++i)
			scale[i] /= headCamera.transform.lossyScale[i];

		if (currentOverlayShape == OverlayShape.Cubemap)
		{
			if (useLegacyCubemapRotation)
			{
#if UNITY_ANDROID && !UNITY_EDITOR
				pose.orientation = pose.orientation * Quaternion.AngleAxis(180, Vector3.up);
#endif
			}
			else
			{
#if UNITY_STANDALONE_WIN || UNITY_EDITOR
				pose.orientation = pose.orientation * Quaternion.AngleAxis(180, Vector3.up);
#endif
			}
			pose.position = headCamera.transform.position;
		}
	}

	bool ComputeSubmit(ref OVRPose pose, ref Vector3 scale, ref bool overlay, ref bool headLocked)
	{
		ComputePoseAndScale(ref pose, ref scale, ref overlay, ref headLocked);

		// Pack the offsetCenter directly into pose.position for offcenterCubemap
		if (currentOverlayShape == OverlayShape.OffcenterCubemap)
		{
			pose.position = transform.position;
			if (pose.position.magnitude > 1.0f)
			{
				Debug.LogWarning("Your cube map center offset's magnitude is greater than 1, which will cause some cube map pixel always invisible .");
				return false;
			}
		}

		// Cylinder overlay sanity checking when not using OpenXR
		if (OVRPlugin.nativeXrApi != OVRPlugin.XrApi.OpenXR && currentOverlayShape == OverlayShape.Cylinder)
		{
			float arcAngle = scale.x / scale.z / (float)Math.PI * 180.0f;
			if (arcAngle > 180.0f)
			{
				Debug.LogWarning("Cylinder overlay's arc angle has to be below 180 degree, current arc angle is " + arcAngle + " degree." );
				return false;
			}
		}

		if (OVRPlugin.nativeXrApi == OVRPlugin.XrApi.OpenXR && currentOverlayShape == OverlayShape.Fisheye)
		{
			Debug.LogWarning("Fisheye overlay shape is not support on OpenXR");
			return false;
		}

		return true;
	}

	void OpenVROverlayUpdate(Vector3 scale, OVRPose pose)
	{
		OVR.OpenVR.CVROverlay overlayRef = OVR.OpenVR.OpenVR.Overlay;
		if (overlayRef == null)
			return;

		Texture overlayTex = textures[0];

		if (overlayTex != null)
		{
			OVR.OpenVR.EVROverlayError error = overlayRef.ShowOverlay(OpenVROverlayHandle);
			if (error == OVR.OpenVR.EVROverlayError.InvalidHandle || error == OVR.OpenVR.EVROverlayError.UnknownOverlay)
			{
				if (overlayRef.FindOverlay(OpenVROverlayKey + transform.name, ref OpenVROverlayHandle) != OVR.OpenVR.EVROverlayError.None)
					return;
			}

			OVR.OpenVR.Texture_t tex = new OVR.OpenVR.Texture_t();
			tex.handle = overlayTex.GetNativeTexturePtr();
			tex.eType = SystemInfo.graphicsDeviceVersion.StartsWith("OpenGL") ? OVR.OpenVR.ETextureType.OpenGL : OVR.OpenVR.ETextureType.DirectX;
			tex.eColorSpace = OVR.OpenVR.EColorSpace.Auto;
			overlayRef.SetOverlayTexture(OpenVROverlayHandle, ref tex);

			OVR.OpenVR.VRTextureBounds_t textureBounds = new OVR.OpenVR.VRTextureBounds_t();
			textureBounds.uMin = (0 + OpenVRUVOffsetAndScale.x) * OpenVRUVOffsetAndScale.z;
			textureBounds.vMin = (1 + OpenVRUVOffsetAndScale.y) * OpenVRUVOffsetAndScale.w;
			textureBounds.uMax = (1 + OpenVRUVOffsetAndScale.x) * OpenVRUVOffsetAndScale.z;
			textureBounds.vMax = (0 + OpenVRUVOffsetAndScale.y) * OpenVRUVOffsetAndScale.w;

			overlayRef.SetOverlayTextureBounds(OpenVROverlayHandle, ref textureBounds);

			OVR.OpenVR.HmdVector2_t vecMouseScale = new OVR.OpenVR.HmdVector2_t();
			vecMouseScale.v0 = OpenVRMouseScale.x;
			vecMouseScale.v1 = OpenVRMouseScale.y;
			overlayRef.SetOverlayMouseScale(OpenVROverlayHandle, ref vecMouseScale);

			overlayRef.SetOverlayWidthInMeters(OpenVROverlayHandle, scale.x);

			Matrix4x4 mat44 = Matrix4x4.TRS(pose.position, pose.orientation, Vector3.one);

			OVR.OpenVR.HmdMatrix34_t pose34 = mat44.ConvertToHMDMatrix34();

			overlayRef.SetOverlayTransformAbsolute(OpenVROverlayHandle, OVR.OpenVR.ETrackingUniverseOrigin.TrackingUniverseStanding, ref pose34);

		}
	}

	private Vector4 OpenVRUVOffsetAndScale = new Vector4(0, 0, 1.0f, 1.0f);
	private Vector2 OpenVRMouseScale = new Vector2(1, 1);
	private OVRManager.XRDevice constructedOverlayXRDevice;
	private bool xrDeviceConstructed = false;

	void LateUpdate()
	{
		if (!OVRManager.OVRManagerinitialized)
			return;
		if (!xrDeviceConstructed)
		{
			InitOVROverlay();
		}

		if (OVRManager.loadedXRDevice != constructedOverlayXRDevice)
		{
			Debug.LogError("Warning-XR Device was switched during runtime with overlays still enabled. When doing so, all overlays constructed with the previous XR device must first be disabled.");
			return;
		}
		// The overlay must be specified every eye frame, because it is positioned relative to the
		// current head location.  If frames are dropped, it will be time warped appropriately,
		// just like the eye buffers.
		bool requiresTextures = !isExternalSurface && NeedsTexturesForShape(currentOverlayShape);
		if (currentOverlayType == OverlayType.None ||
			(requiresTextures && (textures.Length < texturesPerStage || textures[0] == null)))
		{
			return;
		}

		OVRPose pose = OVRPose.identity;
		Vector3 scale = Vector3.one;
		bool overlay = false;
		bool headLocked = false;
		if (!ComputeSubmit(ref pose, ref scale, ref overlay, ref headLocked))
			return;

		if (OVRManager.loadedXRDevice == OVRManager.XRDevice.OpenVR)
		{
			if (currentOverlayShape == OverlayShape.Quad)
				OpenVROverlayUpdate(scale, pose);
			//No more Overlay processing is required if we're on OpenVR
			return;
		}

		OVRPlugin.LayerDesc newDesc = GetCurrentLayerDesc();
		bool isHdr = (newDesc.Format == OVRPlugin.EyeTextureFormat.R16G16B16A16_FP);

		// If the layer and textures are created but sizes differ, force re-creating them.
		// If the layer needed textures but does not anymore (or vice versa), re-create as well.
		bool textureSizesDiffer = !layerDesc.TextureSize.Equals(newDesc.TextureSize) && layerId > 0;
		bool needsTextures = NeedsTexturesForShape(currentOverlayShape);
		bool needsTextureChanged = NeedsTexturesForShape(prevOverlayShape) != needsTextures;
		if (textureSizesDiffer || needsTextureChanged)
		{
			DestroyLayerTextures();
			DestroyLayer();
		}

		bool createdLayer = CreateLayer(newDesc.MipLevels, newDesc.SampleCount, newDesc.Format, newDesc.LayerFlags, newDesc.TextureSize, newDesc.Shape);


		if (layerIndex == -1 || layerId <= 0)
		{
			if (createdLayer)
			{
				// Propagate the current shape and avoid the permanent state of "needs texture changed"
				prevOverlayShape = currentOverlayShape;
			}
			return;
		}

		if (needsTextures)
		{
			bool useMipmaps = (newDesc.MipLevels > 1);

			createdLayer |= CreateLayerTextures(useMipmaps, newDesc.TextureSize, isHdr);

			if (!isExternalSurface && (layerTextures[0].appTexture as RenderTexture != null))
				isDynamic = true;

			if (!LatchLayerTextures())
				return;

			// Don't populate the same frame image twice.
			if (frameIndex > prevFrameIndex)
			{
				int stage = frameIndex % stageCount;
				if (!PopulateLayer(newDesc.MipLevels, isHdr, newDesc.TextureSize, newDesc.SampleCount, stage))
					return;
			}
		}

		bool isOverlayVisible = SubmitLayer(overlay, headLocked, noDepthBufferTesting, pose, scale, frameIndex);

		prevFrameIndex = frameIndex;
		if (isDynamic)
			++frameIndex;

		// Backward compatibility: show regular renderer if overlay isn't visible.
		if (rend)
			rend.enabled = !isOverlayVisible;
	}

#endregion
}
