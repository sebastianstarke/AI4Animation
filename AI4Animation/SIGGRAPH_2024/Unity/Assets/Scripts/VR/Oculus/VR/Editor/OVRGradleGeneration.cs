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

//#define BUILDSESSION

#if USING_XR_MANAGEMENT && (USING_XR_SDK_OCULUS || USING_XR_SDK_OPENXR)
#define USING_XR_SDK
#endif

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Xml;
using System.Diagnostics;
using System.Threading;
using Oculus.VR.Editor;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
#if UNITY_ANDROID
using UnityEditor.Android;
#endif

#if USING_XR_SDK_OPENXR
using UnityEngine.XR.OpenXR;
using UnityEditor.XR.OpenXR.Features;
#endif

#if USING_XR_SDK_OCULUS
using Unity.XR.Oculus;
#endif

[InitializeOnLoad]
public class OVRGradleGeneration
	: IPreprocessBuildWithReport, IPostprocessBuildWithReport
#if UNITY_ANDROID
	, IPostGenerateGradleAndroidProject
#endif
{
	public OVRADBTool adbTool;
	public Process adbProcess;

#if PRIORITIZE_OCULUS_XR_SETTINGS
	private int _callbackOrder = 3;
#else
	private int _callbackOrder = 99999; // be executed after OculusManifest in Oculus XR Plugin, which has callbackOrder 10000
#endif

	public int callbackOrder { get { return _callbackOrder; } }
	static private System.DateTime buildStartTime;
	static private System.Guid buildGuid;

#if UNITY_ANDROID
	public const string prefName = "OVRAutoIncrementVersionCode_Enabled";
	private const string menuItemAutoIncVersion = "Oculus/Tools/Auto Increment Version Code";
	static bool autoIncrementVersion = false;
#endif

#if UNITY_ANDROID && USING_XR_SDK_OCULUS
    static private bool symmetricWarningShown = false;
#endif

    static OVRGradleGeneration()
	{
		EditorApplication.delayCall += OnDelayCall;
	}

	static void OnDelayCall()
	{
#if UNITY_ANDROID
		autoIncrementVersion = PlayerPrefs.GetInt(prefName, 0) != 0;
		Menu.SetChecked(menuItemAutoIncVersion, autoIncrementVersion);
#endif
	}

#if UNITY_ANDROID
	[MenuItem(menuItemAutoIncVersion)]
	public static void ToggleUtilities()
	{
		autoIncrementVersion = !autoIncrementVersion;
		Menu.SetChecked(menuItemAutoIncVersion, autoIncrementVersion);

		int newValue = (autoIncrementVersion) ? 1 : 0;
		PlayerPrefs.SetInt(prefName, newValue);
		PlayerPrefs.Save();

		UnityEngine.Debug.Log("Auto Increment Version Code: " + autoIncrementVersion);
	}
#endif

	public void OnPreprocessBuild(BuildReport report)
	{
		bool useOpenXR = OVRPluginInfo.IsOVRPluginOpenXRActivated();

#if USING_XR_SDK_OPENXR
		UnityEngine.Debug.LogWarning("The installation of Unity OpenXR Plugin is detected, which should NOT be used in production when developing Meta Quest apps for production. Please uninstall the package, and install the Oculus XR Plugin from the Package Manager.");

		// OpenXR Plugin will remove all native plugins if they are not under the Feature folder. Include OVRPlugin to the build if MetaXRFeature is enabled.
		var metaXRFeature = FeatureHelpers.GetFeatureWithIdForBuildTarget(report.summary.platformGroup, Meta.XR.MetaXRFeature.featureId);
		if (metaXRFeature.enabled && !useOpenXR)
		{
			throw new BuildFailedException("OpenXR backend for Oculus Plugin is disabled, which is required to support Unity OpenXR Plugin. Please enable OpenXR backend for Oculus Plugin through the 'Oculus -> Tools -> OpenXR' menu.");
		}

		string ovrRootPath = OVRPluginInfo.GetUtilitiesRootPath();
		var importers = PluginImporter.GetAllImporters();
		foreach (var importer in importers)
		{
			if (!importer.GetCompatibleWithPlatform(report.summary.platform))
				continue;
			string fullAssetPath = Path.Combine(Directory.GetCurrentDirectory(), importer.assetPath);
#if UNITY_EDITOR_WIN
			fullAssetPath = fullAssetPath.Replace("/", "\\");
#endif
			if (fullAssetPath.StartsWith(ovrRootPath) && fullAssetPath.Contains("OVRPlugin"))
			{
				if (metaXRFeature.enabled)
					UnityEngine.Debug.LogFormat("[Meta] Native plugin included in build because of enabled MetaXRFeature: {0}", importer.assetPath);
				else
					UnityEngine.Debug.LogWarning("MetaXRFeature is not enabled in OpenXR Settings. Oculus Integration scripts will not be functional.");
				importer.SetIncludeInBuildDelegate(path => metaXRFeature.enabled);
			}

			// Only disable other OpenXR Loaders if the Meta XR feature is enabled
			if (metaXRFeature.enabled)
			{
				if (!fullAssetPath.StartsWith(ovrRootPath) && (fullAssetPath.Contains("libopenxr_loader.so") || fullAssetPath.Contains("openxr_loader.aar")))
				{
					UnityEngine.Debug.LogFormat("[Meta] libopenxr_loader.so from other packages will be disabled because of enabled MetaXRFeature: {0}", importer.assetPath);
					importer.SetIncludeInBuildDelegate(path => false);
				}
			}
		}
#endif

#if UNITY_ANDROID && !(USING_XR_SDK && UNITY_2019_3_OR_NEWER)
		// Generate error when Vulkan is selected as the perferred graphics API, which is not currently supported in Unity XR
		if (!PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.Android))
		{
			GraphicsDeviceType[] apis = PlayerSettings.GetGraphicsAPIs(BuildTarget.Android);
			if (apis.Length >= 1 && apis[0] == GraphicsDeviceType.Vulkan)
			{
				throw new BuildFailedException("The Vulkan Graphics API does not support XR in your configuration. To use Vulkan, you must use Unity 2019.3 or newer, and the XR Plugin Management.");
			}
		}
#endif

#if UNITY_ANDROID && USING_XR_SDK_OCULUS && OCULUS_XR_SYMMETRIC
        OculusSettings settings;
        UnityEditor.EditorBuildSettings.TryGetConfigObject<OculusSettings>("Unity.XR.Oculus.Settings", out settings);

        if (settings.SymmetricProjection && !symmetricWarningShown)
        {
            symmetricWarningShown = true;
            UnityEngine.Debug.LogWarning("Symmetric Projection is enabled in the Oculus XR Settings. To ensure best GPU performance, make sure at least FFR 1 is being used.");
        }
#endif

#if UNITY_ANDROID
#if USING_XR_SDK
        if (useOpenXR)
		{
			UnityEngine.Debug.LogWarning("Oculus Utilities Plugin with OpenXR is being used, which is under experimental status");

			if (PlayerSettings.colorSpace != ColorSpace.Linear)
			{
				throw new BuildFailedException("Oculus Utilities Plugin with OpenXR only supports linear lighting. Please set 'Rendering/Color Space' to 'Linear' in Player Settings");
			}
		}
#else
		if (useOpenXR)
		{
			throw new BuildFailedException("Oculus Utilities Plugin with OpenXR only supports XR Plug-in Managmenent with Oculus XR Plugin.");
		}
#endif
#endif

#if UNITY_ANDROID && USING_XR_SDK && !USING_COMPATIBLE_OCULUS_XR_PLUGIN_VERSION
		if (PlayerSettings.Android.targetArchitectures != AndroidArchitecture.ARM64)
			throw new BuildFailedException("Your project is using an Oculus XR Plugin version with known issues. Please navigate to the Package Manager and upgrade the Oculus XR Plugin to the latest verified version. When performing the upgrade" +
				", you must first \"Remove\" the Oculus XR Plugin package, and then \"Install\" the package at the verified version. Be sure to remove, then install, not just upgrade.");
#endif

		buildStartTime = System.DateTime.Now;
		buildGuid = System.Guid.NewGuid();

#if BUILDSESSION
		StreamWriter writer = new StreamWriter("build_session", false);
		UnityEngine.Debug.LogFormat("Build Session: {0}", buildGuid.ToString());
		writer.WriteLine(buildGuid.ToString());
		writer.Close();
#endif
	}

	public void OnPostGenerateGradleAndroidProject(string path)
	{
		UnityEngine.Debug.Log("OVRGradleGeneration triggered.");

		var targetOculusPlatform = new List<string>();
		if (OVRDeviceSelector.isTargetDeviceQuestFamily)
		{
			targetOculusPlatform.Add("quest");
		}
		UnityEngine.Debug.LogFormat("QuestFamily = {0}: Quest = {1}, Quest2 = {2}",
			OVRDeviceSelector.isTargetDeviceQuestFamily,
			OVRDeviceSelector.isTargetDeviceQuest,
			OVRDeviceSelector.isTargetDeviceQuest2);

		OVRProjectConfig projectConfig = OVRProjectConfig.GetProjectConfig();
		if (projectConfig != null && projectConfig.systemSplashScreen != null)
		{
			if (PlayerSettings.virtualRealitySplashScreen != null)
			{
				UnityEngine.Debug.LogWarning("Virtual Reality Splash Screen (in Player Settings) is active. It would be displayed after the system splash screen, before the first game frame be rendered.");
			}
			string splashScreenAssetPath = AssetDatabase.GetAssetPath(projectConfig.systemSplashScreen);
			if (Path.GetExtension(splashScreenAssetPath).ToLower() != ".png")
			{
				throw new BuildFailedException("Invalid file format of System Splash Screen. It has to be a PNG file to be used by the Quest OS. The asset path: " + splashScreenAssetPath);
			}
			else
			{
				string sourcePath = splashScreenAssetPath;
				string targetFolder = Path.Combine(path, "src/main/assets");
				string targetPath = targetFolder + "/vr_splash.png";
				UnityEngine.Debug.LogFormat("Copy splash screen asset from {0} to {1}", sourcePath, targetPath);
				try
				{
					File.Copy(sourcePath, targetPath, true);
				}
				catch(Exception e)
				{
					throw new BuildFailedException(e.Message);
				}
			}
		}

		PatchAndroidManifest(path);
	}

	public void PatchAndroidManifest(string path)
	{
		string manifestFolder = Path.Combine(path, "src/main");
		string file = manifestFolder + "/AndroidManifest.xml";

		bool patchedSecurityConfig = false;
		// If Enable NSC Config, copy XML file into gradle project
		OVRProjectConfig projectConfig = OVRProjectConfig.GetProjectConfig();
		if (projectConfig != null)
		{
			if (projectConfig.enableNSCConfig)
			{
				// If no custom xml security path is specified, look for the default location in the integrations package.
				string securityConfigFile = projectConfig.securityXmlPath;
				if (string.IsNullOrEmpty(securityConfigFile))
				{
					securityConfigFile = GetOculusProjectNetworkSecConfigPath();
				}
				else
				{
					Uri configUri = new Uri(Path.GetFullPath(securityConfigFile));
					Uri projectUri = new Uri(Application.dataPath);
					Uri relativeUri = projectUri.MakeRelativeUri(configUri);
					securityConfigFile = relativeUri.ToString();
				}

				string xmlDirectory = Path.Combine(path, "src/main/res/xml");
				try
				{
					if (!Directory.Exists(xmlDirectory))
					{
						Directory.CreateDirectory(xmlDirectory);
					}
					File.Copy(securityConfigFile, Path.Combine(xmlDirectory, "network_sec_config.xml"), true);
					patchedSecurityConfig = true;
				}
				catch (Exception e)
				{
					UnityEngine.Debug.LogError(e.Message);
				}
			}
		}

		OVRManifestPreprocessor.PatchAndroidManifest(file, enableSecurity: patchedSecurityConfig);
	}

	private static string GetOculusProjectNetworkSecConfigPath()
	{
		var so = ScriptableObject.CreateInstance(typeof(OVRPluginInfo));
		var script = MonoScript.FromScriptableObject(so);
		string assetPath = AssetDatabase.GetAssetPath(script);
		string editorDir = Directory.GetParent(assetPath).FullName;
		string configAssetPath = Path.GetFullPath(Path.Combine(editorDir, "network_sec_config.xml"));
		Uri configUri = new Uri(configAssetPath);
		Uri projectUri = new Uri(Application.dataPath);
		Uri relativeUri = projectUri.MakeRelativeUri(configUri);

		return relativeUri.ToString();
	}

	public void OnPostprocessBuild(BuildReport report)
	{
#if UNITY_ANDROID
		if(autoIncrementVersion)
		{
			if((report.summary.options & BuildOptions.Development) == 0)
			{
				PlayerSettings.Android.bundleVersionCode++;
				UnityEngine.Debug.Log("Incrementing version code to " + PlayerSettings.Android.bundleVersionCode);
			}
		}

		bool isExporting = true;
		foreach (var step in report.steps)
		{
			if (step.name.Contains("Compile scripts")
				|| step.name.Contains("Building scenes")
				|| step.name.Contains("Writing asset files")
				|| step.name.Contains("Preparing APK resources")
				|| step.name.Contains("Creating Android manifest")
				|| step.name.Contains("Processing plugins")
				|| step.name.Contains("Exporting project")
				|| step.name.Contains("Building Gradle project"))
			{
#if BUILDSESSION
				UnityEngine.Debug.LogFormat("build_step_" + step.name.ToLower().Replace(' ', '_') + ": {0}", step.duration.TotalSeconds.ToString());
#endif
				if(step.name.Contains("Building Gradle project"))
				{
					isExporting = false;
				}
			}
		}
#endif
		if (!report.summary.outputPath.Contains("OVRGradleTempExport"))
		{
#if BUILDSESSION
			UnityEngine.Debug.LogFormat("build_complete: {0}", (System.DateTime.Now - buildStartTime).TotalSeconds.ToString());
#endif
		}

#if UNITY_ANDROID
		if (!isExporting)
		{
			// Get the hosts path to Android SDK
			if (adbTool == null)
			{
				adbTool = new OVRADBTool(OVRConfig.Instance.GetAndroidSDKPath(false));
			}

			if (adbTool.isReady)
			{
				// Check to see if there are any ADB devices connected before continuing.
				List<string> devices = adbTool.GetDevices();
				if(devices.Count == 0)
				{
					return;
				}

				// Clear current logs on device
				Process adbClearProcess;
				adbClearProcess = adbTool.RunCommandAsync(new string[] { "logcat --clear" }, null);

				// Add a timeout if we cannot get a response from adb logcat --clear in time.
				Stopwatch timeout = new Stopwatch();
				timeout.Start();
				while (!adbClearProcess.WaitForExit(100))
				{
					if (timeout.ElapsedMilliseconds > 2000)
					{
						adbClearProcess.Kill();
						return;
					}
				}

				// Check if existing ADB process is still running, kill if needed
				if (adbProcess != null && !adbProcess.HasExited)
				{
					adbProcess.Kill();
				}

				// Begin thread to time upload and install
				var thread = new Thread(delegate ()
				{
					TimeDeploy();
				});
				thread.Start();
			}
		}
#endif
	}

#if UNITY_ANDROID
	public bool WaitForProcess;
	public bool TransferStarted;
	public DateTime UploadStart;
	public DateTime UploadEnd;
	public DateTime InstallEnd;

	public void TimeDeploy()
	{
		if (adbTool != null)
		{
			TransferStarted = false;
			DataReceivedEventHandler outputRecieved = new DataReceivedEventHandler(
				(s, e) =>
				{
					if (e.Data != null && e.Data.Length != 0 && !e.Data.Contains("\u001b"))
					{
						if (e.Data.Contains("free_cache"))
						{
							// Device recieved install command and is starting upload
							UploadStart = System.DateTime.Now;
							TransferStarted = true;
						}
						else if (e.Data.Contains("Running dexopt"))
						{
							// Upload has finished and Package Manager is starting install
							UploadEnd = System.DateTime.Now;
						}
						else if (e.Data.Contains("dex2oat took"))
						{
							// Package Manager finished install
							InstallEnd = System.DateTime.Now;
							WaitForProcess = false;
						}
						else if (e.Data.Contains("W PackageManager"))
						{
							// Warning from Package Manager is a failure in the install process
							WaitForProcess = false;
						}
					}
				}
			);

			WaitForProcess = true;
			adbProcess = adbTool.RunCommandAsync(new string[] { "logcat" }, outputRecieved);

			Stopwatch transferTimeout = new Stopwatch();
			transferTimeout.Start();
			while (adbProcess != null && !adbProcess.WaitForExit(100))
			{
				if (!WaitForProcess)
				{
					adbProcess.Kill();
				}

				if (!TransferStarted && transferTimeout.ElapsedMilliseconds > 5000)
				{
					adbProcess.Kill();
				}
			}
		}
	}
#endif
}
