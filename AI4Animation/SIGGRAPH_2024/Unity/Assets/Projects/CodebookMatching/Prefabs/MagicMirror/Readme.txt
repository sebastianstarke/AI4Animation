----------------------------------------
Magic Mirror Pro - Recursive Edition (c) 2018 Digital Ruby, LLC
License: May use for personal or commercial products. Source code may not be re-distributed in any way or sold.
----------------------------------------
Version: 1.0.5
----------------------------------------

Introduction
----------------------------------------
Magic Mirror Pro is the premier asset on the Unity Asset Store to do mirrors and reflective surfaces and is the only asset anywhere that allows true, life-like mirror recursion. Setup two mirrors facing each other and be amazed!

Please run the included demo scene to see an example of the mirror script, mirror recursion and water.

Getting Started
----------------------------------------
Simply drop a mirror prefab in your scene and you are done. To view water in the demo scene, import Standard Assets -> Environment.

Customization can be done. Make sure the MagicMirrorScript is in a child object of your custom mirror game object, and make sure the game object that this script is on has no children objects.

Prefabs
----------------------------------------
MagicMirrorPrefab - standard smooth mirror
MagicMirrorBumpPrefab - mirror with bump map

MagicMirrorScript parameters
----------------------------------------
Renderer ReflectRenderer - the renderer to render the mirror in. If you are making your own custom object, set this to your mesh renderer.
LayerMask ReflectionMask - the layers to render in the mirror. Water is supported, but remove any expensive layers or particle systems that you don't want to reflect.
LayerMask ReflectionMaskRecursion - Reflection mask for recursion. Set to 0 to match the ReflectionMask property.
bool ReflectSkybox - Whether to render the default Unity skybox in the reflections. Leave off unless you have an outdoor mirror.
string ReflectionSamplerName - Sets the reflection texture. If you have a custom mirror material, you can use this sampler2D to show the reflection.
int MaximumPerPixelLightsToReflect - For forward rendering mirrors, limits the maximum per pixel lights.
float ClipPlaneOffset - Leave this as is normally. Adjusting will change how close or far the mirror reflects from the mirror plane.
int RenderTextureSize - Reflection render texture size.
RenderingPath ReflectionCameraRenderingPath - The reflection camera render path. Set to 'UsePlayerSettings' to take on the observing camera rendering path. DO NOT CHANGE AT RUNTIME.
bool NormalIsForward - Whether the mirror normal is forward. True for quads, false for planes which use up.
float AspectRatio - Aspect ratio for reflection cameras. 0 for default.
float FieldOfView - Field of view for reflection camera, 0 for default.
float NearPlane - Near plane for reflection camera, 0 for default.
float FarPlane - Far plane for reflection camera, 0 for default.
int RecursionLimit - Recursion limit. Reflections will render off each other up to this many times. Be careful for performance.
float RecursionRenderTextureSizeReducerPower - Reduce render texture size as recursion increases, formula = Mathf.Pow(RecursionRenderTextureSizeReducerPower, recursionLevel) * RenderTextureSize.
RenderTextureFormat RenderTextureFormat - Render texture format for reflection.

MirrorMaterial parameters
----------------------------------------
_MainTex("Detail Texture", 2D) = "clear" {}
	Allow setting an overlay, like a dirt, smear or other texture.
_RecursionLimitTex ("Recursion Limit Texture", 2D) = "grey" {}
	The texture to show if the recursion limit is reached.
_Color ("Detail Tint Color", Color) = (1,1,1,1)
	Tint the detail texture.
_SpecColor ("Specular Color", Color) = (1,1,1,1)
	Adjust the specular color highlight.
_SpecularArea ("Specular Area", Range (0, 0.99)) = 0.1
	Adjust the specular area/power.
_SpecularIntensity ("Specular Intensity", Range (0, 10)) = 0.75
	Adjust the specular intensity.
_ReflectionColor ("Reflection Tint Color", Color) = (1,1,1,1)
	Adjust the reflection tint color.

Mirror bump material has an additional bump map texture.

Rendering
----------------------------------------
Both forward and deferred rendering are supported. The mirror itself can either render as forwad, deferred or use player settings to take on the initial observing camera rendering path. Please note that mirrors rendering with deferred rendering can have artifacts. If you see this, switch mirrors to forward rendering.

Recursion
----------------------------------------
Mirror recursion is supported up to 10 levels. More than 2 mirrors in view of each other is not currently supported.

*IMPORTANT* If you are using recursion PLEASE ensure vsync is turned on for all quality setting levels. You may turn it off temporarily to check performance, but don't accidently ship your game with it on.

Mirror recursion is not supported in edit mode due to the potential to freeze or crash the editor.

Maximum recursion per frame is limited to 100 to avoid completely freezing the editor or game.

Recursion Optimizations
- Shadows: In the first reflection, all shadows are supported. In the next two levels of recursion, only hard shadows are supported. Beyond that, no shadows are rendered.
- Water: Water is not rendered after the first recursive reflection.
- Anti-aliasing: Disabled for all recursive reflections.
- Soft particles: Disabled for all recursive reflections.
- Per pixel lights: Uses script setting 'MaximumPerPixelLightsToReflect' for the first recursive reflection, disabled after that.
- Render texture size can be reduced for each level of recursion. See mirror parameters section 'RecursionRenderTextureSizeReducerPower' parameter.

You can adjust this optimization. Search MagicMirrorScript.cs for // MAGIC MIRROR RECURSION OPTIMIZATION and adjust what happens at what levels of recursion.

Virtual Reality and Mobile
----------------------------------------
VR and mobile should work fine, but performance may be limited. You may want to avoid or limit water and mirror recursion. Single pass stereo is supported.

Due to Unity limitations, two render passes are needed for each eye, regardless of VR setting.

Water
----------------------------------------
Water is supported, but you will have to customize any water reflection scripts and shaders for water that can see mirrors.

Here is the new code for Unity Standard Assets -> Environment -> Water.cs:

private DigitalRuby.MagicMirror.MagicMirrorScript.CameraInfo m_info;

void Start()
{
	// you need to assign the source camera in the render callback down below...
	m_info = new DigitalRuby.MagicMirror.MagicMirrorScript.ReflectionCameraInfo
	{
		ReflectionCamera = m_ReflectionCamera,
		TargetTexture = m_ReflectionTexture,
		TargetTexture2 = m_ReflectionTexture2 // only needed if VR is enabled for right eye, else can be null
	};
}

void OnWillRenderObject()
{
	// ... snip ...

	// Render reflection if needed
	if (mode >= WaterMode.Reflective)
	{
		reflectionCamera.cullingMask = ~(1 << 4) & reflectLayers.value; // never render water layer

		// make sure to set the source camera
		m_info.SourceCamera = Camera.current;

		// tell the render method whether the source camera is a reflection
		string camName;
		m_info.SourceCameraIsReflection = DigitalRuby.MagicMirror.MagicMirrorScript.CameraIsReflection(Camera.current, out camName);

		// call the RenderReflection function which does the heavy lifting
		DigitalRuby.MagicMirror.MagicMirrorScript.RenderReflection(m_info, transform, Vector3.up, m_ClipPlaneOffset);

		// assign the appropriate textures for the water material
		GetComponent<Renderer>().sharedMaterial.SetTexture("_ReflectionTex", m_ReflectionTexture);

		// if supporting VR, make sure to have a second render texture for the right eye
		GetComponent<Renderer>().sharedMaterial.SetTexture("_ReflectionTex2", m_ReflectionTexture2);
	}

	// ... snip ...
}

If you want to support VR properly, you will need to modify the water shader to handle both eyes correctly. See MagicMirrorShader.shader for an example.

Troubleshooting and Performance
----------------------------------------
- Lower max recursion.
- Ensure you are using the latest Unity versions. For example from Unity 5.6.6 to Unity 2017.4, I observed almost double the frame rate.
- You can remove water from the mirror entirely if you need more performance, just set the cull mask of the mirror to not include water.
- If you were using the free version of magic mirror, please remove it entirely from your project before importing Magic Mirror Pro - Recursive edition.
- If you are using Weather Maker, another asset of mine, you should add the mirror camera to the Weather Maker ignore cameras if you have performance problems.

Support, Help, Contact
----------------------------------------
Please send any feedback or bug reports to support@digitalruby.com.

Thanks for using Magic Mirror Pro - Recursive Edition

- Jeff Johnson

