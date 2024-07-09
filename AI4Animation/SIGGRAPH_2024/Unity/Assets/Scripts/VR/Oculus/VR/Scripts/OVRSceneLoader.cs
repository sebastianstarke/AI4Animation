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

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;

using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class OVRSceneLoader : MonoBehaviour
{
	public const string externalStoragePath = "/sdcard/Android/data";
	public const string sceneLoadDataName = "SceneLoadData.txt";
	public const string resourceBundleName = "asset_resources";

	public float sceneCheckIntervalSeconds = 1f;
	public float logCloseTime = 5.0f;

	public Canvas mainCanvas;
	public Text logTextBox;

	private AsyncOperation loadSceneOperation;
	private string formattedLogText;

	private float closeLogTimer;
	private bool closeLogDialogue;

	private bool canvasPosUpdated;

	private struct SceneInfo
	{
		public List<string> scenes;
		public long version;

		public SceneInfo(List<string> sceneList, long currentSceneEpochVersion)
		{
			scenes = sceneList;
			version = currentSceneEpochVersion;
		}
	}

	private string scenePath = "";
	private string sceneLoadDataPath = "";
	private List<AssetBundle> loadedAssetBundles = new List<AssetBundle>();
	private SceneInfo currentSceneInfo;

	private void Awake()
	{
		// Make it presist across scene to continue checking for changes
		DontDestroyOnLoad(this.gameObject);
	}

	void Start()
	{
		string applicationPath = Path.Combine(externalStoragePath, Application.identifier);
		scenePath = Path.Combine(applicationPath, "cache/scenes");
		sceneLoadDataPath = Path.Combine(scenePath, sceneLoadDataName);

		closeLogDialogue = false;
		StartCoroutine(DelayCanvasPosUpdate());

		currentSceneInfo = GetSceneInfo();
		// Check valid scene info has been fetched, and load the scenes
		if (currentSceneInfo.version != 0 && !string.IsNullOrEmpty(currentSceneInfo.scenes[0]))
		{
			LoadScene(currentSceneInfo);
		}
	}

	private void LoadScene(SceneInfo sceneInfo)
	{
		AssetBundle mainSceneBundle = null;
		Debug.Log("[OVRSceneLoader] Loading main scene: " + sceneInfo.scenes[0] + " with version " + sceneInfo.version.ToString());

		logTextBox.text += "Target Scene: " + sceneInfo.scenes[0] + "\n";
		logTextBox.text += "Version: " + sceneInfo.version.ToString() + "\n";

		// Load main scene and dependent additive scenes (if any)
		Debug.Log("[OVRSceneLoader] Loading scene bundle files.");
		// Fetch all files under scene cache path, excluding unnecessary files such as scene metadata file
		string[] bundles = Directory.GetFiles(scenePath, "*_*");
		logTextBox.text += "Loading " + bundles.Length + " bundle(s) . . . ";
		string mainSceneBundleFileName = "scene_" + sceneInfo.scenes[0].ToLower();
		try
		{
			foreach (string b in bundles)
			{
				var assetBundle = AssetBundle.LoadFromFile(b);
				if (assetBundle != null)
				{
					Debug.Log("[OVRSceneLoader] Loading file bundle: " + assetBundle.name == null ? "null" : assetBundle.name);
					loadedAssetBundles.Add(assetBundle);
				}
				else
				{
					Debug.LogError("[OVRSceneLoader] Loading file bundle failed");
				}

				if (assetBundle.name == mainSceneBundleFileName)
				{
					mainSceneBundle = assetBundle;
				}

				if (assetBundle.name == resourceBundleName)
				{
					OVRResources.SetResourceBundle(assetBundle);
				}
			}
		}
		catch(Exception e)
		{
			logTextBox.text += "<color=red>" + e.Message + "</color>";
			return;
		}
		logTextBox.text += "<color=green>DONE\n</color>";

		if (mainSceneBundle != null)
		{
			logTextBox.text += "Loading Scene: {0:P0}\n";
			formattedLogText = logTextBox.text;
			string[] scenePaths = mainSceneBundle.GetAllScenePaths();
			string sceneName = Path.GetFileNameWithoutExtension(scenePaths[0]);
			
			loadSceneOperation = SceneManager.LoadSceneAsync(sceneName);
			loadSceneOperation.completed += LoadSceneOperation_completed;
		}
		else
		{
			logTextBox.text += "<color=red>Failed to get main scene bundle.\n</color>";
		}
	}

	private void LoadSceneOperation_completed(AsyncOperation obj)
	{
		StartCoroutine(onCheckSceneCoroutine());
		StartCoroutine(DelayCanvasPosUpdate());

		closeLogTimer = 0;
		closeLogDialogue = true;

		logTextBox.text += "Log closing in {0} seconds.\n";
		formattedLogText = logTextBox.text;
	}

	public void Update()
	{
		// Display scene load percentage
		if (loadSceneOperation != null)
		{
			if (!loadSceneOperation.isDone)
			{
				logTextBox.text = string.Format(formattedLogText, loadSceneOperation.progress + 0.1f);
				if (loadSceneOperation.progress >= 0.9f)
				{
					logTextBox.text = formattedLogText.Replace("{0:P0}", "<color=green>DONE</color>");
					logTextBox.text += "Transitioning to new scene.\nLoad times will vary depending on scene complexity.\n";
					
				}
			}
		}

		UpdateCanvasPosition();

		// Wait a certain time before closing the log dialogue after the scene has transitioned
		if (closeLogDialogue)
		{
			if (closeLogTimer < logCloseTime)
			{
				closeLogTimer += Time.deltaTime;
				logTextBox.text = string.Format(formattedLogText, (int)(logCloseTime - closeLogTimer));
			}
			else
			{
				mainCanvas.gameObject.SetActive(false);
				closeLogDialogue = false;
			}
		}
	}

	private void UpdateCanvasPosition()
	{
		// Update canvas camera reference and position if the main camera has changed
		if (mainCanvas.worldCamera != Camera.main)
		{
			mainCanvas.worldCamera = Camera.main;
			if (Camera.main != null)
			{
				Vector3 newPosition = Camera.main.transform.position + Camera.main.transform.forward * 0.3f;
				gameObject.transform.position = newPosition;
				gameObject.transform.rotation = Camera.main.transform.rotation;
			}
		}
	}

	private SceneInfo GetSceneInfo()
	{
		SceneInfo sceneInfo = new SceneInfo();
		try
		{
			StreamReader reader = new StreamReader(sceneLoadDataPath);
			sceneInfo.version = System.Convert.ToInt64(reader.ReadLine());
			List<string> sceneList = new List<string>();
			while (!reader.EndOfStream)
			{
				sceneList.Add(reader.ReadLine());
			}
			sceneInfo.scenes = sceneList;
		}
		catch
		{
			logTextBox.text += "<color=red>Failed to get scene info data.\n</color>";
		}
		return sceneInfo;
	}

	// Update canvas position after a slight delay to get accurate headset position after scene transitions
	IEnumerator DelayCanvasPosUpdate()
	{
		yield return new WaitForSeconds(0.1f);
		UpdateCanvasPosition();
	}

	IEnumerator onCheckSceneCoroutine()
	{
		SceneInfo newSceneInfo;
		while (true)
		{
			newSceneInfo = GetSceneInfo();
			if (newSceneInfo.version != currentSceneInfo.version)
			{
				Debug.Log("[OVRSceneLoader] Scene change detected.");

				// Unload all asset bundles
				foreach (var b in loadedAssetBundles)
				{
					if (b != null)
					{
						b.Unload(true);
					}
				}
				loadedAssetBundles.Clear();

				// Unload all scenes in the hierarchy including main scene and 
				// its dependent additive scenes.
				int activeScenes = SceneManager.sceneCount;
				for (int i = 0; i < activeScenes; i++)
				{
					SceneManager.UnloadSceneAsync(SceneManager.GetSceneAt(i));
				}
				DestroyAllGameObjects();
				SceneManager.LoadSceneAsync("OVRTransitionScene");
				break;
			}
			yield return new WaitForSeconds(sceneCheckIntervalSeconds);
		}
	}

	void DestroyAllGameObjects()
	{
		foreach (GameObject go in Resources.FindObjectsOfTypeAll(typeof(GameObject)) as GameObject[])
		{
			Destroy(go);
		}
	}
}
