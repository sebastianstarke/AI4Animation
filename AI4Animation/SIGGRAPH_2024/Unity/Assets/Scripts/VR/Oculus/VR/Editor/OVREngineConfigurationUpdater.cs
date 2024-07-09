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

#if USING_XR_MANAGEMENT && (USING_XR_SDK_OCULUS || USING_XR_SDK_OPENXR)
#define USING_XR_SDK
#endif

#if UNITY_2020_1_OR_NEWER
#define REQUIRES_XR_SDK
#endif

using UnityEngine;
using UnityEditor;
using UnityEditor.Callbacks;
using System;
using System.IO;

[InitializeOnLoad]
class OVREngineConfigurationUpdater
{
	private const string prefName = "OVREngineConfigurationUpdater_Enabled";
	private const string menuItemName = "Oculus/Tools/Use Required Project Settings";
	private const string androidAssetsPath = "Assets/Plugins/Android/assets";
	private const string androidManifestPath = "Assets/Plugins/Android/AndroidManifest.xml";
	static bool setPrefsForUtilities;

	[MenuItem(menuItemName)]
	static void ToggleUtilities()
	{
		setPrefsForUtilities = !setPrefsForUtilities;
		Menu.SetChecked(menuItemName, setPrefsForUtilities);

		int newValue = (setPrefsForUtilities) ? 1 : 0;
		PlayerPrefs.SetInt(prefName, newValue);
		PlayerPrefs.Save();

		Debug.Log("Using required project settings: " + setPrefsForUtilities);
	}
	
	private static readonly string dashSupportEnableConfirmedKey = "Oculus_Utilities_OVREngineConfiguration_DashSupportEnableConfirmed_" + Application.unityVersion + OVRManager.utilitiesVersion;
	private static bool dashSupportEnableConfirmed
	{
		get
		{
			return PlayerPrefs.GetInt(dashSupportEnableConfirmedKey, 0) == 1;
		}

		set
		{
			PlayerPrefs.SetInt(dashSupportEnableConfirmedKey, value ? 1 : 0);
		}
	}


    static OVREngineConfigurationUpdater()
	{
		EditorApplication.delayCall += OnDelayCall;
		EditorApplication.update += OnUpdate;
	}

	static void OnDelayCall()
	{
		setPrefsForUtilities = PlayerPrefs.GetInt(prefName, 1) != 0;
		Menu.SetChecked(menuItemName, setPrefsForUtilities);

		if (!setPrefsForUtilities)
			return;

		OVRPlugin.AddCustomMetadata("build_target", EditorUserBuildSettings.activeBuildTarget.ToString());
		EnforceAndroidSettings();
	}

	static void OnUpdate()
	{
		if (!setPrefsForUtilities)
			return;
		
		EnforceBundleId();
		EnforceVRSupport();
		EnforceInstallLocation();
	}

	static void EnforceAndroidSettings()
	{
		if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android)
			return;

		if (PlayerSettings.defaultInterfaceOrientation != UIOrientation.LandscapeLeft)
		{
			Debug.Log("OVREngineConfigurationUpdater: Setting orientation to Landscape Left");
			// Default screen orientation must be set to landscape left.
			PlayerSettings.defaultInterfaceOrientation = UIOrientation.LandscapeLeft;
		}

#if !USING_XR_SDK && !REQUIRES_XR_SDK
#pragma warning disable 618
		if (!PlayerSettings.virtualRealitySupported)
#pragma warning restore 618
		{
			// NOTE: This value should not affect the main window surface
			// when Built-in VR support is enabled.

			// NOTE: On Adreno Lollipop, it is an error to have antiAliasing set on the
			// main window surface with front buffer rendering enabled. The view will
			// render black.
			// On Adreno KitKat, some tiling control modes will cause the view to render
			// black.
			if (QualitySettings.antiAliasing != 0 && QualitySettings.antiAliasing != 1)
			{
				Debug.Log("OVREngineConfigurationUpdater: Disabling antiAliasing");
				QualitySettings.antiAliasing = 1;
			}
		}
#endif

		if (QualitySettings.vSyncCount != 0)
		{
			Debug.Log("OVREngineConfigurationUpdater: Setting vsyncCount to 0");
			// We sync in the TimeWarp, so we don't want unity syncing elsewhere.
			QualitySettings.vSyncCount = 0;
		}
	}

	static void EnforceVRSupport()
	{
#if !USING_XR_SDK && !REQUIRES_XR_SDK
#pragma warning disable 618
		if (PlayerSettings.virtualRealitySupported)
#pragma warning restore 618
			return;
		
		var mgrs = GameObject.FindObjectsOfType<OVRManager>();
		for (int i = 0; i < mgrs.Length; ++i)
		{
			if (mgrs [i].isActiveAndEnabled)
			{
				Debug.Log ("Enabling Unity VR support");
#pragma warning disable 618
				PlayerSettings.virtualRealitySupported = true;
#pragma warning restore 618

				bool oculusFound = false;
				foreach (var device in UnityEngine.XR.XRSettings.supportedDevices)
					oculusFound |= (device == "Oculus");

				if (!oculusFound)
					Debug.LogError("Please add Oculus to the list of supported devices to use the Utilities.");

				return;
			}
		}
#endif
	}

	private static void EnforceBundleId()
	{
#if !USING_XR_SDK && !REQUIRES_XR_SDK
#pragma warning disable 618
		if (!PlayerSettings.virtualRealitySupported)
		{
			return;
		}
#pragma warning restore 618
#endif
		
#if USING_XR_SDK || !REQUIRES_XR_SDK
		if (PlayerSettings.applicationIdentifier == "" || PlayerSettings.applicationIdentifier == "com.Company.ProductName")
		{
			string defaultBundleId = "com.oculus.UnitySample";
			Debug.LogWarning("\"" + PlayerSettings.applicationIdentifier + "\" is not a valid bundle identifier. Defaulting to \"" + defaultBundleId + "\".");
			PlayerSettings.applicationIdentifier = defaultBundleId;
		}
#endif
	}

	private static void EnforceInstallLocation()
	{
		if (PlayerSettings.Android.preferredInstallLocation != AndroidPreferredInstallLocation.Auto)
			PlayerSettings.Android.preferredInstallLocation = AndroidPreferredInstallLocation.Auto;
	}
}

