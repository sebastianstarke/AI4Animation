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

#if UNITY_EDITOR_WIN && UNITY_ANDROID
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Reflection;
using System.IO;

using UnityEngine;
using UnityEditor;

public class OVRBundleTool : EditorWindow
{
	private static List<EditorSceneInfo> buildableScenes;
	private static Vector2 debugLogScroll = new Vector2(0, 0);
	private static bool invalidBuildableScene;

	private static string toolLog;
	private static bool deployScenesWhenDeployingApk;
	private static bool useOptionalTransitionApkPackage;
	private static GUIStyle windowStyle;
	private static GUIStyle logBoxStyle;
	private static GUIStyle statusStyle;
	private static Vector2 logBoxSize;
	private static Vector2 sceneScrollViewPos;

	private bool forceRestart = false;
	private bool showBundleManagement = false;
	private bool showOther = false;

	// Needed to ensure that APK checking does happen during editor start up, but will still happen when the window is opened/updated
	private static bool panelInitialized = false;

	const float spacesPerIndent = 12;

	private enum ApkStatus
	{
		UNKNOWN,
		OK,
		NOT_INSTALLED,
		DEVICE_NOT_CONNECTED,
	};

	public enum SceneBundleStatus
	{
		[Description("")]
		UNKNOWN,
		[Description("Queued")]
		QUEUED,
		[Description("Building")]
		BUILDING,
		[Description("Done")]
		DONE,
		[Description("Transferring")]
		TRANSFERRING,
		[Description("Deployed")]
		DEPLOYED,
	};

	public class EditorSceneInfo
	{
		public string scenePath;
		public string sceneName;
		public SceneBundleStatus buildStatus;
		public bool shouldDeploy;

		public EditorSceneInfo(string path, string name)
		{
			scenePath = path;
			sceneName = name;
			buildStatus = SceneBundleStatus.UNKNOWN;
			shouldDeploy = true;
		}
	}

	private enum GuiAction
	{
		None,
		OpenBuildSettingsWindow,
		BuildAndDeployScenes,
		BuildAndDeployApp,
		ClearDeviceBundles,
		ClearLocalBundles,
		LaunchApp,
		UninstallApk,
		ClearLog,
	}

	private GuiAction action = GuiAction.None;

	private static ApkStatus currentApkStatus;

	private const string deployScenesWhenDeployingApkPrefName = "OVRBundleTool_DeployScenesWithAPK";
	private const string useOptionalTransitionApkPackagePrefName = "OVRBundleTool_UseOptionalPackageName";

	[MenuItem("Oculus/OVR Build/OVR Scene Quick Preview %l", false, 10)]
	static void Init()
	{
		currentApkStatus = ApkStatus.UNKNOWN;

		EditorWindow.GetWindow(typeof(OVRBundleTool));

		invalidBuildableScene = false;
		InitializePanel();

		OVRPlugin.SetDeveloperMode(OVRPlugin.Bool.True);
		OVRPlugin.SendEvent("oculus_bundle_tool", "show_window");
	}

	public void OnEnable()
	{
		InitializePanel();
	}

	public static void InitializePanel()
	{
		panelInitialized = true;
		GetScenesFromBuildSettings();
        deployScenesWhenDeployingApk = EditorPrefs.GetBool(deployScenesWhenDeployingApkPrefName, true);
        useOptionalTransitionApkPackage = EditorPrefs.GetBool(useOptionalTransitionApkPackagePrefName, false);
		EditorBuildSettings.sceneListChanged += GetScenesFromBuildSettings;
	}

	private void OnGUI()
	{
		this.titleContent.text = "OVR Scene Quick Preview";

		if (panelInitialized)
		{
			CheckForTransitionAPK();
			CheckForDeployedScenes();
			panelInitialized = false;
		}

        if (windowStyle == null)
        {
            windowStyle = new GUIStyle();
            windowStyle.margin = new RectOffset(10, 10, 10, 10);
        }

		if (logBoxStyle == null)
		{
			logBoxStyle = new GUIStyle();
			logBoxStyle.margin.left = 5;
			logBoxStyle.wordWrap = true;
			logBoxStyle.normal.textColor = logBoxStyle.focused.textColor = EditorStyles.label.normal.textColor;
			logBoxStyle.richText = true;
		}

		if (statusStyle == null)
		{
			statusStyle = new GUIStyle(EditorStyles.label);
			statusStyle.richText = true;
		}

        EditorGUILayout.BeginVertical(windowStyle);

        GUILayout.BeginHorizontal(EditorStyles.helpBox);
        GUILayout.BeginVertical();
        EditorGUILayout.LabelField("OVR Scene Quick Preview generates a version of your app which supports hot-reloading "
			+ "content changes to individual scenes, reducing iteration time.",
            EditorStyles.wordWrappedLabel);

#if UNITY_2021_1_OR_NEWER
        if (EditorGUILayout.LinkButton("Documentation"))
#else
        if (GUILayout.Button("Documentation", GUILayout.ExpandWidth(false)))
#endif
        {
            Application.OpenURL("https://developer.oculus.com/documentation/unity/unity-build-android-tools/");
        }
        GUILayout.EndVertical();
        GUILayout.EndHorizontal();
		
		GUILayout.Space(10f);
		GUIContent transitionContent = new GUIContent("Modified APK [?]",
			"Build and deploy an APK that can hot-reload scenes. This enables fast iteration on content changes to scenes.");
		GUILayout.Label(transitionContent, EditorStyles.boldLabel);

		EditorGUILayout.BeginHorizontal();
		{
			GUILayout.Label("Status: ", statusStyle, GUILayout.ExpandWidth(false));

			string statusMesssage;
			switch (currentApkStatus)
			{
				case ApkStatus.OK:
					statusMesssage = "<color=green>APK installed. Ready to build and deploy scenes.</color>";
					break;
				case ApkStatus.NOT_INSTALLED:
					statusMesssage = "<color=red>APK not installed. Press \"Build and Deploy APK\" to install the modified APK.</color>";
					break;
				case ApkStatus.DEVICE_NOT_CONNECTED:
					statusMesssage = "<color=red>Device not connected via ADB. Please connect device and allow debugging.</color>";
					break;
				case ApkStatus.UNKNOWN:
				default:
					statusMesssage = "<color=red>Failed to get APK status!</color>";
					break;
			}
			GUILayout.Label(statusMesssage, statusStyle, GUILayout.ExpandWidth(true));
		}
		EditorGUILayout.EndHorizontal();

		EditorGUILayout.BeginHorizontal();
		if (GUILayout.Button("Build and Deploy APK", GUILayout.Width(200)))
		{
			action = GuiAction.BuildAndDeployApp;
		}
		EditorGUI.BeginDisabledGroup(currentApkStatus != ApkStatus.OK);
		if (GUILayout.Button("Launch APK", GUILayout.Width(120)))
		{
			action = GuiAction.LaunchApp;
		}
		EditorGUI.EndDisabledGroup();
		EditorGUILayout.EndHorizontal();

		GUILayout.Space(10f);

		GUIContent scenesContent = new GUIContent("Scenes [?]",
			"Build and deploy individual scenes, which can be hot-reloaded at runtime by the modified APK.");
		GUILayout.Label(scenesContent, EditorStyles.boldLabel);

		GUIContent buildSettingsBtnTxt = new GUIContent("Open Build Settings");
		GUIContent deployLabelTxt = new GUIContent("Deploy?",
			"If true, this scene will be hot-reloaded. To reduce iteration time, only deploy scenes under active iteration.");
		if (buildableScenes == null || buildableScenes.Count == 0)
		{
			string sceneErrorMessage;
			if (invalidBuildableScene)
			{
				sceneErrorMessage = "Invalid scene selection. \nPlease remove OVRTransitionScene in the project's build settings.";
			}
			else
			{
				sceneErrorMessage = "No scenes detected. \nTo get started, add scenes in the project's build settings.";
			}
			GUILayout.Label(sceneErrorMessage);

			var buildSettingBtnRt = GUILayoutUtility.GetRect(buildSettingsBtnTxt, GUI.skin.button, GUILayout.Width(150));
			if (GUI.Button(buildSettingBtnRt, buildSettingsBtnTxt))
			{
				action = GuiAction.OpenBuildSettingsWindow;
			}
		}
		else
		{
			float currWidth = EditorGUIUtility.currentViewWidth;
			float sceneNameWidth = Math.Max(EditorGUIUtility.currentViewWidth - 170, 60);
			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.LabelField("Scene Name", GUI.skin.box, GUILayout.Width(sceneNameWidth));
			EditorGUILayout.LabelField("Status", GUI.skin.box, GUILayout.Width(80));
			EditorGUILayout.LabelField(deployLabelTxt, GUI.skin.box, GUILayout.Width(60));
			EditorGUILayout.EndHorizontal();

			int scrollViewHeight = Math.Min(buildableScenes.Count * 21, 200);
			sceneScrollViewPos = EditorGUILayout.BeginScrollView(sceneScrollViewPos, GUILayout.MaxHeight(scrollViewHeight));
			foreach (EditorSceneInfo scene in buildableScenes)
			{
				EditorGUILayout.BeginHorizontal();
				EditorGUILayout.LabelField(scene.sceneName, GUILayout.Width(sceneNameWidth + 8));
				EditorGUILayout.LabelField(GetEnumDescription(scene.buildStatus), GUILayout.Width(80));
				scene.shouldDeploy = EditorGUILayout.Toggle(scene.shouldDeploy, GUILayout.Width(30));
				EditorGUILayout.EndHorizontal();
			}
			EditorGUILayout.EndScrollView();

			EditorGUILayout.BeginHorizontal();
			{
				if (GUILayout.Button("Build and Deploy Scene(s)", GUILayout.Width(200)))
				{
					action = GuiAction.BuildAndDeployScenes;
				}
				GUILayout.Space(10);
				GUIContent forceRestartLabel = new GUIContent("Force Restart [?]", "Relaunch the application after scene bundles are finished deploying.");
				forceRestart = GUILayout.Toggle(forceRestart, forceRestartLabel, GUILayout.ExpandWidth(true));
			}
			EditorGUILayout.EndHorizontal();
		}

		GUILayout.Space(10.0f);
		GUILayout.Label("Utilities", EditorStyles.boldLabel);

		showBundleManagement = EditorGUILayout.BeginFoldoutHeaderGroup(showBundleManagement, "Bundle Management", EditorStyles.foldoutHeader);
		if (showBundleManagement)
		{
			EditorGUILayout.BeginHorizontal();
			{
				EditorGUILayout.Space(EditorGUI.indentLevel * spacesPerIndent, false); //to match indentLevel
				GUIContent clearDeviceBundlesTxt = new GUIContent("Delete Device Bundles [?]",
					"Asset bundles to support hot-reloading are stored in an external location on-device. Click to delete them, freeing up space on-device.");
				if (GUILayout.Button(clearDeviceBundlesTxt, GUILayout.ExpandWidth(true)))
				{
					action = GuiAction.ClearDeviceBundles;
				}

				GUIContent clearLocalBundlesTxt = new GUIContent("Delete Local Bundles [?]",
					$"Locally, asset bundles are built into the \"{OVRBundleManager.BUNDLE_MANAGER_OUTPUT_PATH}\" folder at project root. "
					+ "Click to delete them, freeing up local space.");
				if (GUILayout.Button(clearLocalBundlesTxt, GUILayout.ExpandWidth(true)))
				{
					action = GuiAction.ClearLocalBundles;
				}
			}
			EditorGUILayout.EndHorizontal();
		}
		EditorGUILayout.EndFoldoutHeaderGroup();

		GUILayout.Space(5.0f);
		showOther = EditorGUILayout.BeginFoldoutHeaderGroup(showOther, "Other", EditorStyles.foldoutHeader);
		if (showOther)
		{
			const float otherLabelsWidth = 240f;
			EditorGUI.indentLevel++;
			EditorGUILayout.BeginHorizontal();

			GUIContent deployScenesWithApkLabel = new GUIContent("Deploy scenes with APK deploy [?]",
				"If checked, all scenes will be built & deployed when pressing \"Build and Deploy APK\". This takes longer, but provides more expected behavior.");
				
			EditorGUILayout.LabelField(deployScenesWithApkLabel, GUILayout.Width(otherLabelsWidth));
			bool newToggleValue = EditorGUILayout.Toggle(deployScenesWhenDeployingApk);

			if (newToggleValue != deployScenesWhenDeployingApk)
			{
				deployScenesWhenDeployingApk = newToggleValue;
				EditorPrefs.SetBool(deployScenesWhenDeployingApkPrefName, deployScenesWhenDeployingApk);
			}
			EditorGUILayout.EndHorizontal();

			EditorGUILayout.BeginHorizontal();
			GUIContent useOptionalTransitionPackageLabel = new GUIContent("Use optional APK package name [?]",
				"This allows both full build APK and transition APK to be installed on device. However, platform services like Entitlement check may fail.");

			EditorGUILayout.LabelField(useOptionalTransitionPackageLabel, GUILayout.Width(otherLabelsWidth));
			newToggleValue = EditorGUILayout.Toggle(useOptionalTransitionApkPackage);

			if (newToggleValue != useOptionalTransitionApkPackage)
			{
				useOptionalTransitionApkPackage = newToggleValue;
				EditorPrefs.SetBool(useOptionalTransitionApkPackagePrefName, useOptionalTransitionApkPackage);
				// New package name = new check for associated data
				CheckForTransitionAPK();
				CheckForDeployedScenes();
			}
			EditorGUILayout.EndHorizontal();

			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.Space(EditorGUI.indentLevel * spacesPerIndent, false); //to match indentLevel
			if (GUILayout.Button(buildSettingsBtnTxt, GUILayout.ExpandWidth(true)))
			{
				action = GuiAction.OpenBuildSettingsWindow;
			}
			if (GUILayout.Button("Uninstall APK", GUILayout.ExpandWidth(true)))
			{
				action = GuiAction.UninstallApk;
			}
			EditorGUILayout.EndHorizontal();
			EditorGUI.indentLevel--;
		}
		EditorGUILayout.EndFoldoutHeaderGroup();

		GUILayout.Space(6f);
		GUILayout.Label("", GUI.skin.horizontalSlider);
		GUILayout.Space(10f);

		EditorGUILayout.BeginHorizontal();
		GUILayout.Label("Log", EditorStyles.boldLabel);
		GUILayout.FlexibleSpace();
		if (GUILayout.Button("Clear Log", EditorStyles.miniButton))
		{
			action = GuiAction.ClearLog;
		}
		EditorGUILayout.EndHorizontal();

		debugLogScroll = EditorGUILayout.BeginScrollView(debugLogScroll, EditorStyles.helpBox, GUILayout.ExpandHeight(true));
		if (!string.IsNullOrEmpty(toolLog))
		{
			EditorGUILayout.SelectableLabel(toolLog, logBoxStyle, GUILayout.Height(logBoxSize.y));
		}
		EditorGUILayout.EndScrollView();

		EditorGUILayout.EndVertical();
	}

	private void Update()
	{
		switch (action)
		{
			case GuiAction.OpenBuildSettingsWindow:
				OpenBuildSettingsWindow();
				break;
			case GuiAction.BuildAndDeployScenes:
				OVRBundleManager.BuildDeployScenes(buildableScenes, forceRestart);
				CheckForDeployedScenes();
				break;
			case GuiAction.BuildAndDeployApp:
				OVRBundleManager.BuildDeployTransitionAPK();
				CheckForTransitionAPK();
				if (deployScenesWhenDeployingApk)
				{
					buildableScenes.ForEach(x => x.shouldDeploy = true);
					OVRBundleManager.BuildDeployScenes(buildableScenes, false);
					CheckForDeployedScenes();
				}
				OVRBundleManager.LaunchApplication();
				break;
			case GuiAction.ClearDeviceBundles:
				OVRBundleManager.DeleteRemoteAssetBundles();
				CheckForDeployedScenes();
				break;
			case GuiAction.ClearLocalBundles:
				OVRBundleManager.DeleteLocalAssetBundles();
				break;
			case GuiAction.LaunchApp:
				OVRBundleManager.LaunchApplication();
				break;
			case GuiAction.UninstallApk:
				OVRBundleManager.UninstallAPK();
				CheckForTransitionAPK();
				CheckForDeployedScenes();
				break;
			case GuiAction.ClearLog:
				PrintLog("", true);
				break;
			default:
				break;
		}

		action = GuiAction.None;
	}

	private static void OpenBuildSettingsWindow()
	{
		EditorWindow.GetWindow(System.Type.GetType("UnityEditor.BuildPlayerWindow,UnityEditor"));
	}

	public static void UpdateSceneBuildStatus(SceneBundleStatus status, int index = -1)
	{
		if (buildableScenes == null)
		{
			return;
		}
		
		if (index >= 0 && index < buildableScenes.Count)
		{
			buildableScenes[index].buildStatus = status;
		}
		else
		{
			// Update status for all scenes
			for (int i = 0; i < buildableScenes.Count; i++)
			{
				buildableScenes[i].buildStatus = status;
			}
		}
	}

	private static void GetScenesFromBuildSettings()
	{
		invalidBuildableScene = false;
		buildableScenes = new List<EditorSceneInfo>();
		for (int i = 0; i < EditorBuildSettings.scenes.Length; i++)
		{
			EditorBuildSettingsScene scene = EditorBuildSettings.scenes[i];
			if (scene.enabled)
			{
				if (Path.GetFileNameWithoutExtension(scene.path) != "OVRTransitionScene")
				{
					EditorSceneInfo sceneInfo = new EditorSceneInfo(scene.path, Path.GetFileNameWithoutExtension(scene.path));
					buildableScenes.Add(sceneInfo);
				}
				else
				{
					buildableScenes = null;
					invalidBuildableScene = true;
					return;
				}
			}
		}
	}

	private static void CheckForTransitionAPK()
	{
		OVRADBTool adbTool = new OVRADBTool(OVRConfig.Instance.GetAndroidSDKPath());
		if (adbTool.isReady)
		{
			string matchedPackageList, error;
			var transitionPackageName = PlayerSettings.GetApplicationIdentifier(BuildTargetGroup.Android);
			if (useOptionalTransitionApkPackage)
			{
				transitionPackageName += ".transition";
			}
			string[] packageCheckCommand = new string[] { "-d shell pm list package", transitionPackageName };
			if (adbTool.RunCommand(packageCheckCommand, null, out matchedPackageList, out error) == 0)
			{
				if (string.IsNullOrEmpty(matchedPackageList))
				{
					currentApkStatus = ApkStatus.NOT_INSTALLED;
				}
				else
				{
					// adb "list package" command returns all package names that contains the given query package name
					// Need to check if the transition package name is matched exactly
					if (matchedPackageList.Contains("package:" + transitionPackageName + "\r\n"))
					{
						if (useOptionalTransitionApkPackage)
						{
							// If optional package name is used, it is deterministic that the transition apk is installed
							currentApkStatus = ApkStatus.OK;
						}
						else
						{
							// get package info to check for TRANSITION_APK_VERSION_NAME
							string[] dumpPackageInfoCommand = new string[] { "-d shell dumpsys package", transitionPackageName };
							string packageInfo;
							if (adbTool.RunCommand(dumpPackageInfoCommand, null, out packageInfo, out error) == 0 &&
									!string.IsNullOrEmpty(packageInfo) &&
									packageInfo.Contains(OVRBundleManager.TRANSITION_APK_VERSION_NAME))
							{
								// Matched package name found, and the package info contains TRANSITION_APK_VERSION_NAME
								currentApkStatus = ApkStatus.OK;
							}
							else
							{
								currentApkStatus = ApkStatus.NOT_INSTALLED;
							}
						}
					}
					else
					{
						// No matached package name returned
						currentApkStatus = ApkStatus.NOT_INSTALLED;
					}
				}
			}
			else if (error.Contains("no devices found"))
			{
				currentApkStatus = ApkStatus.DEVICE_NOT_CONNECTED;
			}
			else
			{
				currentApkStatus = ApkStatus.UNKNOWN;
			}
		}
	}

	private static void CheckForDeployedScenes()
	{
		if (buildableScenes == null) return;
		UpdateSceneBuildStatus(SceneBundleStatus.UNKNOWN);

		string[] deployedBundleNames = OVRBundleManager.ListRemoteAssetBundleNames();
		if (deployedBundleNames == null) return;

		for (int i = 0; i < buildableScenes.Count; ++i)
		{
			string sceneBundleName = "scene_" + buildableScenes[i].sceneName;
			if (Array.FindIndex(deployedBundleNames, x => x.Equals(sceneBundleName, StringComparison.CurrentCultureIgnoreCase)) != -1)
			{
				UpdateSceneBuildStatus(SceneBundleStatus.DEPLOYED, i);
			}
		}
	}

	public static void PrintLog(string message, bool clear = false)
	{
		if (clear)
		{
			toolLog = message;
		}
		else
		{
			toolLog += message + "\n";
		}

		if (logBoxStyle != null)
		{
			GUIContent logContent = new GUIContent(toolLog);
			logBoxSize = logBoxStyle.CalcSize(logContent);

			debugLogScroll.y = float.MaxValue; //scroll to bottom on new data
		}
	}

	public static void PrintError(string error = "")
	{
		if(!string.IsNullOrEmpty(error))
		{
			toolLog += "<color=red>Failed!\n</color>" + error + "\n";
		}
		else
		{
			toolLog += "<color=red>Failed! Check Unity log for more details.\n</color>";
		}
	}

	public static void PrintWarning(string warning)
	{
		toolLog += "<color=yellow>Warning!\n" + warning + "</color>\n";
	}

	public static void PrintSuccess()
	{
		toolLog += "<color=green>Success!</color>\n";
	}

	public static string GetEnumDescription(Enum eEnum)
	{
		Type enumType = eEnum.GetType();
		MemberInfo[] memberInfo = enumType.GetMember(eEnum.ToString());
		if (memberInfo != null && memberInfo.Length > 0)
		{
			var attrs = memberInfo[0].GetCustomAttributes(typeof(DescriptionAttribute), false);
			if (attrs != null && attrs.Length > 0)
			{
				return ((DescriptionAttribute)attrs[0]).Description;
			}
		}
		return eEnum.ToString();
	}

	public static bool GetUseOptionalTransitionApkPackage()
	{
		return useOptionalTransitionApkPackage;
	}
}
#endif
