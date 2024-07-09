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
using UnityEditor;
using System.IO;

/// <summary>
/// From the selected transform, takes a cubemap screenshot that can be submitted with the application
/// as a screenshot (or additionally used for reflection shaders).
/// </summary>
class OVRScreenshotWizard : ScriptableWizard
{
	public enum TexFormat
	{
		JPEG,	// 512kb at 1k x 1k resolution vs
		PNG,	// 5.3mb
	}

	public enum SaveMode {
		SaveCubemapScreenshot,
		SaveUnityCubemap,
		SaveBoth,
	}

	public GameObject		renderFrom = null;
	public int				size = 2048;
	public SaveMode			saveMode = SaveMode.SaveUnityCubemap;
	public string			cubeMapFolder = "Assets/Textures/Cubemaps";
	public TexFormat		textureFormat = TexFormat.PNG;

	/// <summary>
	/// Validates the user's input
	/// </summary>
	void OnWizardUpdate()
	{
		helpString = "Select a game object positioned in the place where\nyou want to render the cubemap screenshot from: ";
		isValid = (renderFrom != null);
	}

	/// <summary>
	/// Create the asset path if it is not available.
	/// Assuming the newFolderPath is stated with "Assets", which is a requirement.
	/// </summary>
 	static bool CreateAssetPath( string newFolderPath )
	{
		const  int maxFoldersCount = 32;
		string currentPath;
		string[] pathFolders;

		pathFolders = newFolderPath.Split (new char[]{ '/' }, maxFoldersCount);

		if (!string.Equals ("Assets", pathFolders [0], System.StringComparison.OrdinalIgnoreCase))
		{
			Debug.LogError( "Folder path has to be started with \" Assets \" " );
			return false;
		}

		currentPath = "Assets";
		for (int i = 1; i < pathFolders.Length; i++)
		{
			if (!string.IsNullOrEmpty(pathFolders[i]))
			{
				string newPath = currentPath + "/" + pathFolders[i];
				if (!AssetDatabase.IsValidFolder(newPath))
					AssetDatabase.CreateFolder(currentPath, pathFolders[i]);
				currentPath = newPath;
			}
		}

		Debug.Log( "Created path: " + currentPath );
		return true;
	}

	/// <summary>
	/// Renders the cubemap
	/// </summary>
	void OnWizardCreate()
	{
		if ( !AssetDatabase.IsValidFolder( cubeMapFolder ) )
		{
			if (!CreateAssetPath(cubeMapFolder))
			{
				Debug.LogError( "Created path failed: " + cubeMapFolder );
				return;
			}
		}

		bool existingCamera = true;
		bool existingCameraStateSave = true;
		Camera camera = renderFrom.GetComponent<Camera>();
		if (camera == null)
		{
			camera = renderFrom.AddComponent<Camera>();
			camera.farClipPlane = 10000f;
			existingCamera = false;
		}
		else
		{
			existingCameraStateSave = camera.enabled;
			camera.enabled = true;
		}
		// find the last screenshot saved
		if (cubeMapFolder[cubeMapFolder.Length-1] != '/')
		{
			cubeMapFolder += "/";
		}
		int idx = 0;
		string[] fileNames = Directory.GetFiles(cubeMapFolder);
		foreach(string fileName in fileNames)
		{
			if (!fileName.ToLower().EndsWith(".cubemap"))
			{
				continue;
			}
			string temp = fileName.Replace(cubeMapFolder + "vr_screenshot_", string.Empty);
			temp = temp.Replace(".cubemap", string.Empty);
			int tempIdx = 0;
			if (int.TryParse( temp, out tempIdx ))
			{
				if (tempIdx > idx)
				{
					idx = tempIdx;
				}
			}
		}
		string pathName = string.Format("{0}vr_screenshot_{1}.cubemap", cubeMapFolder, (++idx).ToString("d2"));
		Cubemap cubemap = new Cubemap(size, TextureFormat.RGB24, false);

		// render into cubemap
		if ((camera != null) && (cubemap != null))
		{
			// set up cubemap defaults
			OVRCubemapCapture.RenderIntoCubemap(camera, cubemap);
			if (existingCamera)
			{
				camera.enabled = existingCameraStateSave;
			}
			else
			{
				DestroyImmediate(camera);
			}
			// generate a regular texture as well?
			if ( ( saveMode == SaveMode.SaveCubemapScreenshot ) || ( saveMode == SaveMode.SaveBoth ) )
			{
				GenerateTexture(cubemap, pathName);
			}

			if ( ( saveMode == SaveMode.SaveUnityCubemap ) || ( saveMode == SaveMode.SaveBoth ) )
			{
				Debug.Log( "Saving: " + pathName );
				// by default the unity cubemap isn't saved
				AssetDatabase.CreateAsset( cubemap, pathName );
				// reimport as necessary
				AssetDatabase.SaveAssets();
				// select it in the project tree so developers can find it
				EditorGUIUtility.PingObject( cubemap );
				Selection.activeObject = cubemap;
			}
			AssetDatabase.Refresh();
		}
	}

	/// <summary>
	/// Generates a NPOT 6x1 cubemap in the following format PX NX PY NY PZ NZ
	/// </summary>
	void GenerateTexture(Cubemap cubemap, string pathName)
	{
		// Encode the texture and save it to disk
		pathName = pathName.Replace(".cubemap", (textureFormat == TexFormat.PNG) ? ".png" : ".jpg" ).ToLower();
		pathName = pathName.Replace( cubeMapFolder.ToLower(), "" );
		string format = textureFormat.ToString();
		string fullPath = EditorUtility.SaveFilePanel( string.Format( "Save Cubemap Screenshot as {0}", format ), "", pathName, format.ToLower() );
		if ( !string.IsNullOrEmpty( fullPath ) )
		{
			Debug.Log( "Saving: " + fullPath );
			OVRCubemapCapture.SaveCubemapCapture(cubemap, fullPath);
		}
	}

	/// <summary>
	/// Unity Editor menu option to take a screenshot
	/// </summary>
	[MenuItem("Oculus/Tools/OVR Screenshot Wizard", false, 100000)]
	static void TakeOVRScreenshot()
	{
		OVRScreenshotWizard wizard = ScriptableWizard.DisplayWizard<OVRScreenshotWizard>("OVR Screenshot Wizard", "Render Cubemap");
		if (wizard != null)
		{
			if (Selection.activeGameObject != null)
			wizard.renderFrom = Selection.activeGameObject;
			else
			wizard.renderFrom = Camera.main.gameObject;

			wizard.isValid = (wizard.renderFrom != null);
		}
	}
}
