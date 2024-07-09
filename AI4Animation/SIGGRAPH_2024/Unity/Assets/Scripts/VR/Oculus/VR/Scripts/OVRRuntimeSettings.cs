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
using UnityEditor;

using System.IO;
using System;

public class OVRRuntimeSettings : ScriptableObject
{
	public OVRManager.ColorSpace colorSpace = OVRManager.ColorSpace.P3;

#if UNITY_EDITOR
	private static string GetOculusRuntimeSettingsAssetPath()
	{
		string resourcesPath = Path.Combine(Application.dataPath, "Resources");
		if (!Directory.Exists(resourcesPath))
		{
			Directory.CreateDirectory(resourcesPath);
		}

		string settingsAssetPath = Path.GetFullPath(Path.Combine(resourcesPath, "OculusRuntimeSettings.asset"));
		Uri configUri = new Uri(settingsAssetPath);
		Uri projectUri = new Uri(Application.dataPath);
		Uri relativeUri = projectUri.MakeRelativeUri(configUri);

		return relativeUri.ToString();
	}

	public static void CommitRuntimeSettings(OVRRuntimeSettings runtimeSettings)
	{
		string runtimeSettingsAssetPath = GetOculusRuntimeSettingsAssetPath();
		if (AssetDatabase.GetAssetPath(runtimeSettings) != runtimeSettingsAssetPath)
		{
			Debug.LogWarningFormat("The asset path of RuntimeSettings is wrong. Expect {0}, get {1}", runtimeSettingsAssetPath, AssetDatabase.GetAssetPath(runtimeSettings));
		}
		EditorUtility.SetDirty(runtimeSettings);
	}
#endif

	public static OVRRuntimeSettings GetRuntimeSettings()
	{
		OVRRuntimeSettings settings = null;
#if UNITY_EDITOR
		string oculusRuntimeSettingsAssetPath = GetOculusRuntimeSettingsAssetPath();
		try
		{
			settings = AssetDatabase.LoadAssetAtPath(oculusRuntimeSettingsAssetPath, typeof(OVRRuntimeSettings)) as OVRRuntimeSettings;
		}
		catch (System.Exception e)
		{
			Debug.LogWarningFormat("Unable to load RuntimeSettings from {0}, error {1}", oculusRuntimeSettingsAssetPath, e.Message);
		}

		if (settings == null && !BuildPipeline.isBuildingPlayer)
		{
			settings = ScriptableObject.CreateInstance<OVRRuntimeSettings>();

			AssetDatabase.CreateAsset(settings, oculusRuntimeSettingsAssetPath);
		}
#else
		settings = Resources.Load<OVRRuntimeSettings>("OculusRuntimeSettings");
		if (settings == null)
		{
			Debug.LogWarning("Failed to load runtime settings. Using default runtime settings instead.");
			settings = ScriptableObject.CreateInstance<OVRRuntimeSettings>();
		}
#endif
		return settings;
	}
}
