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

//#define DEBUG_OVERLAY_CANVAS
using UnityEngine;

[RequireComponent(typeof(Canvas))]
public class OVROverlayCanvas : MonoBehaviour
{
	[SerializeField, HideInInspector]
	private Shader _transparentShader = null;
	[SerializeField, HideInInspector]
	private Shader _opaqueShader = null;

	RectTransform _rectTransform;
	Canvas _canvas;
	Camera _camera;
	OVROverlay _overlay;
	RenderTexture _renderTexture;
	MeshRenderer _meshRenderer;

	Mesh _quad;
	Material _defaultMat;

	public int MaxTextureSize = 1600;
	public int MinTextureSize = 200;
	public float PixelsPerUnit = 1f;
	public int DrawRate = 1;
	public int DrawFrameOffset = 0;
	public bool Expensive = false;
	public int Layer = 0;

	public enum DrawMode
	{
		Opaque,
		OpaqueWithClip,
		TransparentDefaultAlpha,
		TransparentCorrectAlpha
	}


	public DrawMode Opacity = DrawMode.OpaqueWithClip;

	private bool ScaleViewport = Application.isMobilePlatform;

	// Start is called before the first frame update
	void Start()
	{
		_canvas = GetComponent<Canvas>();

		_rectTransform = _canvas.GetComponent<RectTransform>();

		float rectWidth = _rectTransform.rect.width;
		float rectHeight = _rectTransform.rect.height;

		float aspectX = rectWidth >= rectHeight ? 1 : rectWidth / rectHeight;
		float aspectY = rectHeight >= rectWidth ? 1 : rectHeight / rectWidth;

		// if we are scaling the viewport we don't need to add a border
		int pixelBorder = ScaleViewport ? 0 : 8;
		int innerWidth = Mathf.CeilToInt(aspectX * (MaxTextureSize - pixelBorder * 2));
		int innerHeight = Mathf.CeilToInt(aspectY * (MaxTextureSize - pixelBorder * 2));
		int width = innerWidth + pixelBorder * 2;
		int height = innerHeight + pixelBorder * 2;

		float paddedWidth = rectWidth * (width / (float)innerWidth);
		float paddedHeight = rectHeight * (height / (float)innerHeight);

		float insetRectWidth = innerWidth / (float)width;
		float insetRectHeight = innerHeight / (float)height;

		// ever so slightly shrink our opaque mesh to avoid black borders
		Vector2 opaqueTrim = Opacity == DrawMode.Opaque ? new Vector2(0.005f / _rectTransform.lossyScale.x, 0.005f / _rectTransform.lossyScale.y) : Vector2.zero;

		_renderTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
		// if we can't scale the viewport, generate mipmaps instead
		_renderTexture.useMipMap = !ScaleViewport;

		GameObject overlayCamera = new GameObject(name + " Overlay Camera")
		{
#if !DEBUG_OVERLAY_CANVAS
			hideFlags = HideFlags.HideInHierarchy | HideFlags.NotEditable
#endif
		};
		overlayCamera.transform.SetParent(transform, false);

		_camera = overlayCamera.AddComponent<Camera>();
		_camera.stereoTargetEye = StereoTargetEyeMask.None;
		_camera.transform.position = transform.position - transform.forward;
		_camera.orthographic = true;
		_camera.enabled = false;
		_camera.targetTexture = _renderTexture;
		_camera.cullingMask = 1 << gameObject.layer;
		_camera.clearFlags = CameraClearFlags.SolidColor;
		_camera.backgroundColor = Color.clear;
		_camera.orthographicSize = 0.5f * paddedHeight * _rectTransform.localScale.y;
		_camera.nearClipPlane = 0.99f;
		_camera.farClipPlane = 1.01f;

		_quad = new Mesh()
		{
			name = name + " Overlay Quad",
			hideFlags = HideFlags.HideAndDontSave
		};

		_quad.vertices = new Vector3[] { new Vector3(-0.5f, -0.5f), new Vector3(-0.5f, 0.5f), new Vector3(0.5f, 0.5f), new Vector3(0.5f, -0.5f) };
		_quad.uv = new Vector2[] { new Vector2(0, 0), new Vector2(0, 1), new Vector2(1, 1), new Vector2(1, 0) };
		_quad.triangles = new int[] { 0, 1, 2, 2, 3, 0 };
		_quad.bounds = new Bounds(Vector3.zero, Vector3.one);
		_quad.UploadMeshData(true);

		switch(Opacity)
		{
			case DrawMode.Opaque:
				_defaultMat = new Material(_opaqueShader);
				break;
			case DrawMode.OpaqueWithClip:
				_defaultMat = new Material(_opaqueShader);
				_defaultMat.EnableKeyword("WITH_CLIP");
				break;
			case DrawMode.TransparentDefaultAlpha:
				_defaultMat = new Material(_transparentShader);
				_defaultMat.EnableKeyword("ALPHA_SQUARED");
				break;
			case DrawMode.TransparentCorrectAlpha:
				_defaultMat = new Material(_transparentShader);
				break;
		}
		_defaultMat.mainTexture = _renderTexture;
		_defaultMat.color = Color.black;
		_defaultMat.mainTextureOffset = new Vector2(0.5f - 0.5f * insetRectWidth, 0.5f - 0.5f * insetRectHeight);
		_defaultMat.mainTextureScale = new Vector2(insetRectWidth, insetRectHeight);

		GameObject meshRenderer = new GameObject(name + " MeshRenderer")
		{
#if !DEBUG_OVERLAY_CANVAS
			hideFlags = HideFlags.HideInHierarchy | HideFlags.NotEditable
#endif
		};

		meshRenderer.transform.SetParent(transform, false);
		meshRenderer.AddComponent<MeshFilter>().sharedMesh = _quad;
		_meshRenderer = meshRenderer.AddComponent<MeshRenderer>();
		_meshRenderer.sharedMaterial = _defaultMat;
		meshRenderer.layer = Layer;
		meshRenderer.transform.localScale = new Vector3(rectWidth - opaqueTrim.x, rectHeight - opaqueTrim.y, 1);

		GameObject overlay = new GameObject(name + " Overlay")
		{
#if !DEBUG_OVERLAY_CANVAS
			hideFlags = HideFlags.HideInHierarchy | HideFlags.NotEditable
#endif
		};
		overlay.transform.SetParent(transform, false);
		_overlay = overlay.AddComponent<OVROverlay>();
		_overlay.isDynamic = true;
		_overlay.noDepthBufferTesting = true;
		_overlay.isAlphaPremultiplied = !Application.isMobilePlatform;
		_overlay.textures[0] = _renderTexture;
		_overlay.currentOverlayType = OVROverlay.OverlayType.Underlay;
		_overlay.transform.localScale = new Vector3(paddedWidth, paddedHeight, 1);
		_overlay.useExpensiveSuperSample = Expensive;
	}	

	private void OnDestroy()
	{
		Destroy(_defaultMat);
		Destroy(_quad);
		Destroy(_renderTexture);
	}

	private void OnEnable()
	{
		if (_overlay)
		{
			_meshRenderer.enabled = true;
			_overlay.enabled = true;
		}
		if (_camera)
		{
			_camera.enabled = true;
		}
	}

	private void OnDisable()
	{
		if (_overlay)
		{
			_overlay.enabled = false;
			_meshRenderer.enabled = false;
		}

		if (_camera)
		{
			_camera.enabled = false;
		}
	}

	private static readonly Plane[] _FrustumPlanes = new Plane[6];
	protected virtual bool ShouldRender()
	{
		if (DrawRate > 1)
		{
			if (Time.frameCount % DrawRate != DrawFrameOffset % DrawRate)
			{
				return false;
			}
		}

		if (Camera.main != null)
		{
			// Perform Frustum culling
			for (int i = 0; i < 2; i++)
			{
				var eye = (Camera.StereoscopicEye)i;
				var mat = Camera.main.GetStereoProjectionMatrix(eye) * Camera.main.GetStereoViewMatrix(eye);
				GeometryUtility.CalculateFrustumPlanes(mat, _FrustumPlanes);
				if (GeometryUtility.TestPlanesAABB(_FrustumPlanes, _meshRenderer.bounds))
				{
					return true;
				}
			}
			return false;
		}

		return true;
	}

	private void Update()
	{
		if (ShouldRender())
		{
			if (ScaleViewport)
			{
				if (Camera.main != null)
				{
					float d = (Camera.main.transform.position - transform.position).magnitude;

					float size = PixelsPerUnit * Mathf.Max(_rectTransform.rect.width * transform.lossyScale.x, _rectTransform.rect.height * transform.lossyScale.y) / d;

					// quantize to even pixel sizes
					const float quantize = 8;
					float pixelHeight = Mathf.Ceil(size / quantize * _renderTexture.height) * quantize;

					// clamp between or min size and our max size
					pixelHeight = Mathf.Clamp(pixelHeight, MinTextureSize, _renderTexture.height);

					float innerPixelHeight = pixelHeight - 2;

					_camera.orthographicSize = 0.5f * _rectTransform.rect.height * _rectTransform.localScale.y * pixelHeight / innerPixelHeight;

					float aspect = (_rectTransform.rect.width / _rectTransform.rect.height);

					float innerPixelWidth = innerPixelHeight * aspect;
					float pixelWidth = Mathf.Ceil((innerPixelWidth + 2) * 0.5f) * 2;

					float sizeX = pixelWidth / _renderTexture.width;
					float sizeY = pixelHeight / _renderTexture.height;

					// trim a half pixel off each size if this is opaque (transparent should fade)
					float inset = Opacity == DrawMode.Opaque ? 1.001f : 0;
				
					float innerSizeX = (innerPixelWidth - inset) / _renderTexture.width;
					float innerSizeY = (innerPixelHeight - inset) / _renderTexture.height;

					// scale the camera rect
					_camera.rect = new Rect((1 - sizeX) / 2, (1 - sizeY) / 2, sizeX, sizeY);

					Rect src = new Rect(0.5f - (0.5f * innerSizeX), 0.5f - (0.5f * innerSizeY), innerSizeX, innerSizeY);

					_defaultMat.mainTextureOffset = src.min;
					_defaultMat.mainTextureScale = src.size;

					// update the overlay to use this same size
					_overlay.overrideTextureRectMatrix = true;
					src.y = 1 - src.height - src.y;
					Rect dst = new Rect(0, 0, 1, 1);
					_overlay.SetSrcDestRects(src, src, dst, dst);
				}
			}

			_camera.Render();			
		}
	}	

	public bool overlayEnabled
	{
		get
		{
			return _overlay && _overlay.enabled;
		}
		set
		{
			if (_overlay)
			{
				_overlay.enabled = value;
				_defaultMat.color = value ? Color.black : Color.white;
			}
		}
	}
}
