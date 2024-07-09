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
using System.Collections; // required for Coroutines

/// <summary>
/// Fades the screen from black after a new scene is loaded. Fade can also be controlled mid-scene using SetUIFade and SetFadeLevel
/// </summary>
public class OVRScreenFade : MonoBehaviour
{
	public static OVRScreenFade instance { get; private set; }

	[Tooltip("Fade duration")]
	public float fadeTime = 2.0f;

	[Tooltip("Screen color at maximum fade")]
	public Color fadeColor = new Color(0.01f, 0.01f, 0.01f, 1.0f);

	public bool fadeOnStart = true;

	/// <summary>
	/// The render queue used by the fade mesh. Reduce this if you need to render on top of it.
	/// </summary>
	public int renderQueue = 5000;

	/// <summary>
	/// Renders the current alpha value being used to fade the screen.
	/// </summary>
	public float currentAlpha { get { return Mathf.Max(explicitFadeAlpha, animatedFadeAlpha, uiFadeAlpha); } }

	private float explicitFadeAlpha = 0.0f;
	private float animatedFadeAlpha = 0.0f;
	private float uiFadeAlpha = 0.0f;

	private MeshRenderer fadeRenderer;
	private MeshFilter fadeMesh;
	private Material fadeMaterial = null;
	private bool isFading = false;

	/// <summary>
	/// Automatically starts a fade in
	/// </summary>
	void Start()
	{
		if (gameObject.name.StartsWith("OculusMRC_"))
		{
			Destroy(this);
			return;
		}

		// create the fade material
		fadeMaterial = new Material(Shader.Find("Oculus/Unlit Transparent Color"));
		fadeMesh = gameObject.AddComponent<MeshFilter>();
		fadeRenderer = gameObject.AddComponent<MeshRenderer>();

		var mesh = new Mesh();
		fadeMesh.mesh = mesh;

		Vector3[] vertices = new Vector3[4];

		float width = 2f;
		float height = 2f;
		float depth = 1f;

		vertices[0] = new Vector3(-width, -height, depth);
		vertices[1] = new Vector3(width, -height, depth);
		vertices[2] = new Vector3(-width, height, depth);
		vertices[3] = new Vector3(width, height, depth);

		mesh.vertices = vertices;

		int[] tri = new int[6];

		tri[0] = 0;
		tri[1] = 2;
		tri[2] = 1;

		tri[3] = 2;
		tri[4] = 3;
		tri[5] = 1;

		mesh.triangles = tri;

		Vector3[] normals = new Vector3[4];

		normals[0] = -Vector3.forward;
		normals[1] = -Vector3.forward;
		normals[2] = -Vector3.forward;
		normals[3] = -Vector3.forward;

		mesh.normals = normals;

		Vector2[] uv = new Vector2[4];

		uv[0] = new Vector2(0, 0);
		uv[1] = new Vector2(1, 0);
		uv[2] = new Vector2(0, 1);
		uv[3] = new Vector2(1, 1);

		mesh.uv = uv;

		explicitFadeAlpha = 0.0f;
		animatedFadeAlpha = 0.0f;
		uiFadeAlpha = 0.0f;

		if (fadeOnStart)
		{
			FadeIn();
		}

		instance = this;
	}

	/// <summary>
	/// Start a fade in
	/// </summary>
	public void FadeIn()
	{
		StartCoroutine(Fade(1.0f, 0.0f));
	}

	/// <summary>
	/// Start a fade out
	/// </summary>
	public void FadeOut()
	{
		StartCoroutine(Fade(0,1));
	}

	/// <summary>
	/// Starts a fade in when a new level is loaded
	/// </summary>
	void OnLevelFinishedLoading(int level)
	{
		FadeIn();
	}

	void OnEnable()
	{
		if (!fadeOnStart)
		{
			explicitFadeAlpha = 0.0f;
			animatedFadeAlpha = 0.0f;
			uiFadeAlpha = 0.0f;
		}
	}

	/// <summary>
	/// Cleans up the fade material
	/// </summary>
	void OnDestroy()
	{
		instance = null;

		if (fadeRenderer != null)
			Destroy(fadeRenderer);

		if (fadeMaterial != null)
			Destroy(fadeMaterial);

		if (fadeMesh != null)
			Destroy(fadeMesh);
	}

	/// <summary>
	/// Set the UI fade level - fade due to UI in foreground
	/// </summary>
	public void SetUIFade(float level)
	{
		uiFadeAlpha = Mathf.Clamp01(level);
		SetMaterialAlpha();
	}

	/// <summary>
	/// Override current fade level
	/// </summary>
	/// <param name="level"></param>
	public void SetExplicitFade(float level)
	{
		explicitFadeAlpha = level;
		SetMaterialAlpha();
	}

	/// <summary>
	/// Fades alpha from 1.0 to 0.0
	/// </summary>
	IEnumerator Fade(float startAlpha, float endAlpha)
	{
		float elapsedTime = 0.0f;
		while (elapsedTime < fadeTime)
		{
			elapsedTime += Time.deltaTime;
			animatedFadeAlpha = Mathf.Lerp(startAlpha, endAlpha, Mathf.Clamp01(elapsedTime / fadeTime));
			SetMaterialAlpha();
			yield return new WaitForEndOfFrame();
		}
		animatedFadeAlpha = endAlpha;
		SetMaterialAlpha();
	}

	/// <summary>
	/// Update material alpha. UI fade and the current fade due to fade in/out animations (or explicit control)
	/// both affect the fade. (The max is taken)
	/// </summary>
	private void SetMaterialAlpha()
	{
		Color color = fadeColor;
		color.a = currentAlpha;
		isFading = color.a > 0;
		if (fadeMaterial != null)
		{
			fadeMaterial.color = color;
			fadeMaterial.renderQueue = renderQueue;
			fadeRenderer.material = fadeMaterial;
			fadeRenderer.enabled = isFading;
		}
	}
}
