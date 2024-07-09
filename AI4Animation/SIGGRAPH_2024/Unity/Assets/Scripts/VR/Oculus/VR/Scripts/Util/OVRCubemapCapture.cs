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
using System.Collections;
using System.IO;

/// <summary>
/// Helper script for capture cubemap and save it into PNG or JPG file
/// </summary>

/// <description>
/// How it works:
/// 1) This script can be attached to a existing game object, you can also use prefab Assets\OVR\Prefabs\OVRCubemapCaptureProbe
///	There are 2 ways to trigger a capture if you attached this script to a game object.
///		* Automatic capturing: if [autoTriggerAfterLaunch] is true, a automatic capturing will be triggered after [autoTriggerDelay] seconds.
///		* Keyboard trigger: press key [triggeredByKey], a capturing will be triggered.
/// 2) If you like to trigger the screen capture in your code logic, just call static function [OVRCubemapCapture.TriggerCubemapCapture] with proper input arguments.
/// </description>

public class OVRCubemapCapture : MonoBehaviour
{
	/// <summary>
	/// Enable the automatic screenshot trigger, which will capture a cubemap after autoTriggerDelay (seconds)
	/// </summary>
	public bool autoTriggerAfterLaunch = true;
	public float autoTriggerDelay = 1.0f;
	private float autoTriggerElapse = 0.0f;

	/// <summary>
	/// Trigger cubemap screenshot if user pressed key triggeredByKey
	/// </summary>
	public KeyCode triggeredByKey = KeyCode.F8;

	/// <summary>
	/// The complete file path for saving the cubemap screenshot, including the filename and extension
	/// if pathName is blank, screenshots will be saved into %USERPROFILE%\Documents\OVR_ScreenShot360
	/// </summary>
	public string pathName;

	/// <summary>
	/// The cube face resolution
	/// </summary>
	public int cubemapSize = 2048;

	// Update is called once per frame
	void Update()
	{
		// Trigger after autoTriggerDelay
		if (autoTriggerAfterLaunch)
		{
			autoTriggerElapse += Time.deltaTime;
			if (autoTriggerElapse >= autoTriggerDelay)
			{
				autoTriggerAfterLaunch = false;
				TriggerCubemapCapture(transform.position, cubemapSize, pathName);
			}
		}

		//todo: enable for Unity Input System
#if ENABLE_LEGACY_INPUT_MANAGER
		// Trigger by press triggeredByKey
		if ( Input.GetKeyDown( triggeredByKey ) )
		{
			TriggerCubemapCapture(transform.position, cubemapSize, pathName);
		}
#endif
	}

	/// <summary>
	/// Generate unity cubemap at specific location and save into JPG/PNG
	/// </summary>
	/// <description>
	/// Default save folder: your app's persistentDataPath
	/// Default file name: using current time OVR_hh_mm_ss.png
	/// Note1: this will take a few seconds to finish
	/// Note2: if you only want to specify path not filename, please end [pathName] with "/"
	/// </description>

	public static void TriggerCubemapCapture(Vector3 capturePos, int cubemapSize = 2048, string pathName = null)
	{
		GameObject ownerObj = new GameObject("CubemapCamera", typeof(Camera));
		ownerObj.hideFlags = HideFlags.HideAndDontSave;
		ownerObj.transform.position = capturePos;
		ownerObj.transform.rotation = Quaternion.identity;
		Camera camComponent = ownerObj.GetComponent<Camera>();
		camComponent.farClipPlane = 10000.0f;
		camComponent.enabled = false;

		Cubemap cubemap = new Cubemap(cubemapSize, TextureFormat.RGB24, false);
		RenderIntoCubemap(camComponent, cubemap);
		SaveCubemapCapture(cubemap, pathName);
		DestroyImmediate(cubemap);
		DestroyImmediate(ownerObj);
	}


	public static void RenderIntoCubemap(Camera ownerCamera, Cubemap outCubemap)
	{
		int width = (int)outCubemap.width;
		int height = (int)outCubemap.height;

		CubemapFace[] faces = new CubemapFace[] { CubemapFace.PositiveX, CubemapFace.NegativeX, CubemapFace.PositiveY, CubemapFace.NegativeY, CubemapFace.PositiveZ, CubemapFace.NegativeZ };
		Vector3[] faceAngles = new Vector3[] { new Vector3(0.0f, 90.0f, 0.0f), new Vector3(0.0f, -90.0f, 0.0f), new Vector3(-90.0f, 0.0f, 0.0f), new Vector3(90.0f, 0.0f, 0.0f), new Vector3(0.0f, 0.0f, 0.0f), new Vector3(0.0f, 180.0f, 0.0f) };

		// Backup states
		RenderTexture backupRenderTex = RenderTexture.active;
		float backupFieldOfView = ownerCamera.fieldOfView;
		float backupAspect = ownerCamera.aspect;
		Quaternion backupRot = ownerCamera.transform.rotation;
		//RenderTexture backupRT = ownerCamera.targetTexture;

		// Enable 8X MSAA
		RenderTexture faceTexture = new RenderTexture(width, height, 24);
		faceTexture.antiAliasing = 8;
		faceTexture.dimension = UnityEngine.Rendering.TextureDimension.Tex2D;
		faceTexture.hideFlags = HideFlags.HideAndDontSave;

		// For intermediate saving
		Texture2D swapTex = new Texture2D(width, height, TextureFormat.RGB24, false);
		swapTex.hideFlags = HideFlags.HideAndDontSave;

		// Capture 6 Directions
		ownerCamera.targetTexture = faceTexture;
		ownerCamera.fieldOfView = 90;
		ownerCamera.aspect = 1.0f;

		Color[] mirroredPixels = new Color[swapTex.height * swapTex.width];
		for (int i = 0; i < faces.Length; i++)
		{
			ownerCamera.transform.eulerAngles = faceAngles[i];
			ownerCamera.Render();
			RenderTexture.active = faceTexture;
			swapTex.ReadPixels(new Rect(0, 0, width, height), 0, 0);

			// Mirror vertically to meet the standard of unity cubemap
			Color[] OrignalPixels = swapTex.GetPixels();
			for (int y1 = 0; y1 < height; y1++)
			{
				for (int x1 = 0; x1 < width; x1++)
				{
					mirroredPixels[y1 * width + x1] = OrignalPixels[((height - 1 - y1) * width) + x1];
				}
			};
			outCubemap.SetPixels(mirroredPixels, faces[i]);
		}

		outCubemap.SmoothEdges();

		// Restore states
		RenderTexture.active = backupRenderTex;
		ownerCamera.fieldOfView = backupFieldOfView;
		ownerCamera.aspect = backupAspect;
		ownerCamera.transform.rotation = backupRot;
		ownerCamera.targetTexture = backupRenderTex;

		DestroyImmediate(swapTex);
		DestroyImmediate(faceTexture);

		}


	/// <summary>
	/// Save unity cubemap into NPOT 6x1 cubemap/texture atlas in the following format PX NX PY NY PZ NZ
	/// </summary>
	/// <description>
	/// Supported format: PNG/JPG
	/// Default file name: using current time OVR_hh_mm_ss.png
	/// </description>

	public static bool SaveCubemapCapture(Cubemap cubemap, string pathName = null)
	{
		string fileName;
		string dirName;
		int width = cubemap.width;
		int height = cubemap.height;
		int x = 0;
		int y = 0;
		bool saveToPNG = true;

		if (string.IsNullOrEmpty(pathName))
		{
			dirName = Application.persistentDataPath + "/OVR_ScreenShot360/";
			fileName = null;
		}
		else
		{
			dirName = Path.GetDirectoryName(pathName);
			fileName = Path.GetFileName(pathName);

			if (dirName[dirName.Length - 1] != '/' || dirName[dirName.Length - 1] != '\\')
				dirName += "/";
		}

		if (string.IsNullOrEmpty(fileName))
			fileName = "OVR_" + System.DateTime.Now.ToString("hh_mm_ss") + ".png";

		string extName = Path.GetExtension(fileName);
		if (extName == ".png")
		{
			saveToPNG = true;
		}
		else if (extName == ".jpg")
		{
			saveToPNG = false;
		}
		else
		{
			Debug.LogError("Unsupported file format" + extName);
			return false;
		}

		// Validate path
		try
		{
			System.IO.Directory.CreateDirectory(dirName);
		}
		catch (System.Exception e)
		{
			Debug.LogError("Failed to create path " + dirName + " since " + e.ToString());
			return false;
		}


		// Create the new texture
		Texture2D tex = new Texture2D(width * 6, height, TextureFormat.RGB24, false);
		if (tex == null)
		{
			Debug.LogError("[OVRScreenshotWizard] Failed creating the texture!");
			return false;
		}

		// Merge all the cubemap faces into the texture
		// Reference cubemap format: http://docs.unity3d.com/Manual/class-Cubemap.html
		CubemapFace[] faces = new CubemapFace[] { CubemapFace.PositiveX, CubemapFace.NegativeX, CubemapFace.PositiveY, CubemapFace.NegativeY, CubemapFace.PositiveZ, CubemapFace.NegativeZ };
		for (int i = 0; i < faces.Length; i++)
		{
			// get the pixels from the cubemap
			Color[] srcPixels = null;
			Color[] pixels = cubemap.GetPixels(faces[i]);
			// if desired, flip them as they are ordered left to right, bottom to top
			srcPixels = new Color[pixels.Length];
			for (int y1 = 0; y1 < height; y1++)
			{
				for (int x1 = 0; x1 < width; x1++)
				{
					srcPixels[y1 * width + x1] = pixels[((height - 1 - y1) * width) + x1];
				}
			}
			// Copy them to the dest texture
			tex.SetPixels(x, y, width, height, srcPixels);
			x += width;
		}

		try
		{
			// Encode the texture and save it to disk
			byte[] bytes = saveToPNG ? tex.EncodeToPNG() : tex.EncodeToJPG();

			System.IO.File.WriteAllBytes(dirName + fileName, bytes);
			Debug.Log("Cubemap file created " + dirName + fileName);
		}
		catch (System.Exception e)
		{
			Debug.LogError("Failed to save cubemap file since " + e.ToString());
			return false;
		}

		DestroyImmediate(tex);
		return true;
	}

}
