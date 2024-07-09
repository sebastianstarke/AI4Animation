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

#if UNITY_EDITOR

#if USING_XR_MANAGEMENT && (USING_XR_SDK_OCULUS || USING_XR_SDK_OPENXR)
#define USING_XR_SDK
#endif

#if UNITY_2020_1_OR_NEWER
#define REQUIRES_XR_SDK
#endif

using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using Assets.OVR.Scripts;
using Assets.Oculus.VR;
using Assets.Oculus.VR.Editor;
using Oculus.VR.Editor;

#if USING_XR_MANAGEMENT && USING_XR_SDK_OCULUS
using Unity.XR.Oculus;
#endif

/// <summary>
///Scans the project and warns about the following conditions:
///Audio sources > 16
///Using MSAA levels other than recommended level
///Excessive pixel lights (>1 on Mobile; >3 on Rift)
///Directional Lightmapping Modes (on Mobile; use Non-Directional)
///Preload audio setting on individual audio clips
///Decompressing audio clips on load
///Disabling occlusion mesh
///Android target API level set to 21 or higher
///Unity skybox use (on by default, but if you can't see the skybox switching to Color is much faster on Mobile)
///Lights marked as "baked" but that were not included in the last bake (and are therefore realtime).
///Lack of static batching and dynamic batching settings activated.
///Full screen image effects (Mobile)
///Warn about large textures that are marked as uncompressed.
///32-bit depth buffer (use 16)
///Use of projectors (Mobile; can be used carefully but slow enough to warrant a warning)
///Maybe in the future once quantified: Graphics jobs and IL2CPP on Mobile.
///Real-time global illumination
///No texture compression, or non-ASTC/ETC2 texture compression as a global setting (Mobile).
///Using deferred rendering
///Excessive texture resolution after LOD bias (>2k on Mobile; >4k on Rift)
///Not using trilinear or aniso filtering and not generating mipmaps
///Excessive render scale (>1.2)
///Slow physics settings: Sleep Threshold < 0.005, Default Contact Offset < 0.01, Solver Iteration Count > 6
///Shadows on when approaching the geometry or draw call limits
///Non-static objects with colliders that are missing rigidbodies on themselves or in the parent chain.
///No initialization of GPU/CPU throttling settings, or init to dangerous values (-1 or > 3)  (Mobile)
///Using inefficient effects: SSAO, motion blur, global fog, parallax mapping, etc.
///Too many Overlay layers
///Use of Standard shader or Standard Specular shader on Mobile.  More generally, excessive use of multipass shaders (legacy specular, etc).
///Multiple cameras with clears (on Mobile, potential for excessive fill cost)
///Excessive shader passes (>2)
///Material pointers that have been instanced in the editor (esp. if we could determine that the instance has no deltas from the original)
///Excessive draw calls (>150 on Mobile; >2000 on Rift)
///Excessive tris or verts (>100k on Mobile; >1M on Rift)
///Large textures, lots of prefabs in startup scene (for bootstrap optimization)
///GPU skinning: testing Android-only, as most Rift devs are GPU-bound.
/// </summary>
[InitializeOnLoadAttribute]
public class OVRLint : EditorWindow
{
	//TODO: The following require reflection or static analysis.
	///Use of ONSP reflections (Mobile)
	///Use of LoadLevelAsync / LoadLevelAdditiveAsync (on Mobile, this kills frame rate so dramatically it's probably better to just go to black and load synchronously)
	///Use of Linq in non-editor assemblies (common cause of GCs).  Minor: use of foreach.
	///Use of Unity WWW (exceptionally high overhead for large file downloads, but acceptable for tiny gets).
	///Declared but empty Awake/Start/Update/OnCollisionEnter/OnCollisionExit/OnCollisionStay.  Also OnCollision* star methods that declare the Collision  argument but do not reference it (omitting it short-circuits the collision contact calculation).

	public enum eRecordType
	{
		StaticCommon, // Applies to all Oculus hardware, can be identified without running the app
		StaticAndroid, // Applies to Android-based Oculus hardware, can be identified without running the app
		RuntimeCommon, // Applies to all Oculus hardware, can be identified only while running the app
		RuntimeAndroid, // Applies to Android-based Oculus hardware, can be identified only while running the app
	}

	private static List<FixRecord> mRecordsStaticCommon = new List<FixRecord>();
	private static List<FixRecord> mRecordsStaticAndroid = new List<FixRecord>();
	private static List<FixRecord> mRecordsRuntimeCommon = new List<FixRecord>();
	private static List<FixRecord> mRecordsRuntimeAndroid = new List<FixRecord>();

	bool mShowRecordsStaticCommon = false;
	bool mShowRecordsRuntimeCommon = false;
#if UNITY_ANDROID
	bool mShowRecordsStaticAndroid = false;
	bool mShowRecordsRuntimeAndroid = false;
#endif

	private static List<FixRecord> mRuntimeEditModeRequiredRecords = new List<FixRecord>();

	private Vector2 mScrollPosition;

	GUIStyle mFixIncompleteStyle;
	GUIStyle mFixCompleteStyle;

	[MenuItem("Oculus/Tools/OVR Performance Lint Tool")]
	static void Init()
	{
		// Get existing open window or if none, make a new one:
		EditorWindow.GetWindow(typeof(OVRLint));
		OVRPlugin.SendEvent("perf_lint", "activated");
		OVRLint.RunCheck();
	}

	void OnEnable()
	{
		var incompleteStyleTex = new Texture2D(1, 1);
		incompleteStyleTex.SetPixel(0, 0, new Color(0.4f, 0.4f, 0.4f, 0.2f));
		incompleteStyleTex.Apply();
		mFixIncompleteStyle = new GUIStyle();
		mFixIncompleteStyle.normal.background = incompleteStyleTex;
		mFixIncompleteStyle.padding = new RectOffset(8, 8, 2, 2);
		mFixIncompleteStyle.margin = new RectOffset(4, 4, 4, 4);

		var completeStyleTex = new Texture2D(1, 1);
		completeStyleTex.SetPixel(0, 0, new Color(0, 0.7f, 0, 0.2f));
		completeStyleTex.Apply();
		mFixCompleteStyle = new GUIStyle();
		mFixCompleteStyle.normal.background = completeStyleTex;
		mFixCompleteStyle.padding = new RectOffset(8, 8, 2, 2);
		mFixCompleteStyle.margin = new RectOffset(4, 4, 4, 4);
	}

	OVRLint()
	{
		EditorApplication.playModeStateChanged += HandlePlayModeState;
	}

	private static void HandlePlayModeState(PlayModeStateChange state)
	{
		if (state == PlayModeStateChange.EnteredEditMode)
		{
			ApplyEditModeRequiredFix();
		}
	}

	private static void ApplyEditModeRequiredFix()
	{
		// Apply runtime fixes that require edit mode when applying fix
		foreach (FixRecord record in mRuntimeEditModeRequiredRecords)
		{
			record.fixMethod(null, false, 0);
			OVRPlugin.SendEvent("perf_lint_apply_fix", record.category);
			record.complete = true;
		}
		mRuntimeEditModeRequiredRecords.Clear();
	}

	void OnGUI()
	{
		GUILayout.Label("OVR Performance Lint Tool", EditorStyles.boldLabel);
		if (GUILayout.Button("Refresh", EditorStyles.toolbarButton, GUILayout.ExpandWidth(false)))
		{
			RunCheck();
		}

		mScrollPosition = EditorGUILayout.BeginScrollView(mScrollPosition);

		mShowRecordsStaticCommon = EditorGUILayout.BeginFoldoutHeaderGroup(mShowRecordsStaticCommon, $"Common Issues ({mRecordsStaticCommon.Count})", EditorStyles.foldoutHeader);
		if (mShowRecordsStaticCommon)
			DisplayRecords(mRecordsStaticCommon);
		EditorGUILayout.EndFoldoutHeaderGroup();

#if UNITY_ANDROID
		mShowRecordsStaticAndroid = EditorGUILayout.BeginFoldoutHeaderGroup(mShowRecordsStaticAndroid, $"Quest Issues ({mRecordsStaticAndroid.Count})", EditorStyles.foldoutHeader);
		if (mShowRecordsStaticAndroid)
			DisplayRecords(mRecordsStaticAndroid);
		EditorGUILayout.EndFoldoutHeaderGroup();
#else
		EditorGUI.BeginDisabledGroup(true);
		EditorGUILayout.BeginFoldoutHeaderGroup(false, "Quest Issues (disabled: Build Target is not Android)");
		EditorGUILayout.EndFoldoutHeaderGroup();
		EditorGUI.EndDisabledGroup();
#endif

		if (Application.isPlaying)
		{
			mShowRecordsRuntimeCommon = EditorGUILayout.BeginFoldoutHeaderGroup(mShowRecordsRuntimeCommon, $"Common Runtime Issues ({mRecordsRuntimeCommon.Count})", EditorStyles.foldoutHeader);
			if (mShowRecordsRuntimeCommon)
				DisplayRecords(mRecordsRuntimeCommon);
			EditorGUILayout.EndFoldoutHeaderGroup();

#if UNITY_ANDROID
			mShowRecordsRuntimeAndroid = EditorGUILayout.BeginFoldoutHeaderGroup(mShowRecordsRuntimeAndroid, $"Quest Runtime Issues ({mRecordsRuntimeAndroid.Count})", EditorStyles.foldoutHeader);
			if (mShowRecordsRuntimeAndroid)
				DisplayRecords(mRecordsRuntimeAndroid);
			EditorGUILayout.EndFoldoutHeaderGroup();
#else
			EditorGUI.BeginDisabledGroup(true);
			EditorGUILayout.BeginFoldoutHeaderGroup(false, "Quest Runtime Issues (disabled: Build Target is not Android)");
			EditorGUILayout.EndFoldoutHeaderGroup();
			EditorGUI.EndDisabledGroup();
#endif
		}
		else
		{
			EditorGUI.BeginDisabledGroup(true);

			EditorGUILayout.BeginFoldoutHeaderGroup(false, "Common Runtime Issues (disabled: not in Play mode)");
			EditorGUILayout.EndFoldoutHeaderGroup();

#if UNITY_ANDROID
			EditorGUILayout.BeginFoldoutHeaderGroup(false, "Quest Runtime Issues (disabled: not in Play mode)");
			EditorGUILayout.EndFoldoutHeaderGroup();
#else
			EditorGUILayout.BeginFoldoutHeaderGroup(false, "Quest Runtime Issues (disabled: Build Target is not Android)");
			EditorGUILayout.EndFoldoutHeaderGroup();
#endif

			EditorGUI.EndDisabledGroup();
		}

		EditorGUILayout.EndScrollView();
	}

	void DisplayRecords(List<FixRecord> records)
	{
		for (int x = 0; x < records.Count; x++)
		{
			FixRecord record = records[x];

			int siblingRecordCount = 0;
			while (x + siblingRecordCount + 1 < records.Count && records[x + siblingRecordCount + 1].category.Equals(record.category))
				++siblingRecordCount;

			EditorGUILayout.BeginHorizontal(record.complete ? mFixCompleteStyle : mFixIncompleteStyle); //2-column wrapper for record
			EditorGUILayout.BeginVertical(GUILayout.Width(20)); //column 1: icon
			EditorGUILayout.LabelField(EditorGUIUtility.IconContent(record.complete ? "d_Progress" : "console.warnicon"), GUILayout.Width(20));
			EditorGUILayout.EndVertical(); //end column 1: icon
			EditorGUILayout.BeginVertical(GUILayout.ExpandWidth(true)); //column 2: label, message, objects
			GUILayout.Label(record.category, EditorStyles.boldLabel);

			if (!string.IsNullOrEmpty(record.message))
				GUILayout.Label(record.message, EditorStyles.wordWrappedLabel);

			for (int i = 0; i <= siblingRecordCount; ++i)
			{
				var iterRecord = records[x + i];
				if (iterRecord.targetObject)
					EditorGUILayout.ObjectField(iterRecord.targetObject, iterRecord.targetObject.GetType(), true);
			}
			EditorGUILayout.EndVertical(); //end column 2: label, message, objects

			if (record.buttonNames != null && record.buttonNames.Length > 0)
			{
				EditorGUILayout.BeginVertical(GUILayout.Width(200.0f)); //column 3: buttons
				GUI.enabled = !record.complete;

				for (int y = 0; y < record.buttonNames.Length; y++)
				{
					if (siblingRecordCount > 0)
						GUILayout.Label("(Applies to all entries)", EditorStyles.miniLabel);

					if (GUILayout.Button(record.buttonNames[y], EditorStyles.toolbarButton))
					{
						var undoObjects = new List<UnityEngine.Object>();
						for (int i = 0; i <= siblingRecordCount; ++i)
							if (records[x + i].targetObject)
								undoObjects.Add(records[x + i].targetObject);

						if (undoObjects.Count > 0)
							Undo.RecordObjects(undoObjects.ToArray(), record.category);

						for (int i = 0; i <= siblingRecordCount; ++i)
						{
							FixRecord thisRecord = records[x + i];

							if (thisRecord.editModeRequired)
							{
								// Add to the fix record list that requires edit mode
								mRuntimeEditModeRequiredRecords.Add(record);
							}
							else
							{
								thisRecord.fixMethod(thisRecord.targetObject, (i == siblingRecordCount), y);
								OVRPlugin.SendEvent("perf_lint_apply_fix", thisRecord.category);
								thisRecord.complete = true;
							}
						}

						if (mRuntimeEditModeRequiredRecords.Count != 0)
						{
							// Stop the scene to apply edit mode required records
							EditorApplication.ExecuteMenuItem("Edit/Play");
						}
					}
				}
				GUI.enabled = true;
				EditorGUILayout.EndVertical(); //end column 3: buttons
			}

			EditorGUILayout.EndHorizontal(); //end 3-column wrapper for record
			x += siblingRecordCount;
		}
	}


	public static int RunCheck()
	{
		mRecordsStaticCommon.Clear();
		mRecordsStaticAndroid.Clear();
		mRecordsRuntimeCommon.Clear();
		mRecordsRuntimeAndroid.Clear();
		mRuntimeEditModeRequiredRecords.Clear();

		CheckStaticCommonIssues();
#if UNITY_ANDROID
		CheckStaticAndroidIssues();
#endif

		if (EditorApplication.isPlaying)
		{
			CheckRuntimeCommonIssues();
#if UNITY_ANDROID
			CheckRuntimeAndroidIssues();
#endif
		}

		mRecordsStaticCommon.Sort(FixRecordSorter);
		mRecordsStaticAndroid.Sort(FixRecordSorter);
		mRecordsRuntimeCommon.Sort(FixRecordSorter);
		mRecordsRuntimeAndroid.Sort(FixRecordSorter);

		return mRecordsStaticCommon.Count + mRecordsStaticAndroid.Count + mRecordsRuntimeCommon.Count + mRecordsRuntimeAndroid.Count;
	}

	static int FixRecordSorter(FixRecord record1, FixRecord record2)
	{
		if (record1.sortOrder != record2.sortOrder)
			return record1.sortOrder.CompareTo(record2.sortOrder);
		else if (record1.category != record2.category)
			return record1.category.CompareTo(record2.category);
		else
			return record1.complete.CompareTo(record2.complete);
	}

	static void AddFix(eRecordType recordType, string category, string message, FixMethodDelegate method, UnityEngine.Object target, bool editModeRequired, params string[] buttons)
	{
		AddFix(recordType, 0/*sortOrder*/, category, message, method, target, editModeRequired, buttons);
	}

	static void AddFix(eRecordType recordType, int sortOrder, string category, string message, FixMethodDelegate method, UnityEngine.Object target, bool editModeRequired, params string[] buttons)
	{
		OVRPlugin.SendEvent("perf_lint_add_fix", category);
		var fixRecord = new FixRecord(sortOrder, category, message, method, target, editModeRequired, buttons);
		switch (recordType)
		{
			case eRecordType.StaticCommon: mRecordsStaticCommon.Add(fixRecord); break;
			case eRecordType.StaticAndroid: mRecordsStaticAndroid.Add(fixRecord); break;
			case eRecordType.RuntimeCommon: mRecordsRuntimeCommon.Add(fixRecord); break;
			case eRecordType.RuntimeAndroid: mRecordsRuntimeAndroid.Add(fixRecord); break;
		}
	}

	static void CheckStaticCommonIssues()
	{
		if (OVRManager.IsUnityAlphaOrBetaVersion())
		{
			AddFix(eRecordType.StaticCommon, "General", OVRManager.UnityAlphaOrBetaVersionWarningMessage, null, null, false);
		}

#if USING_XR_SDK_OPENXR
		AddFix(eRecordType.StaticCommon, -9999, "Unity OpenXR Plugin Detected", "Unity OpenXR Plugin should NOT be used in production when developing Oculus apps. Please uninstall the package, and install the Oculus XR Plugin from the Package Manager.\nWhen using the Oculus XR Plugin, you can enable OpenXR backend for Oculus Plugin through the 'Oculus -> Tools -> OVR Utilities Plugin' menu.", null, null, false);
#endif

		if (!OVRPluginInfo.IsOVRPluginOpenXRActivated() || OVRPluginInfo.IsOVRPluginUnityProvidedActivated())
		{
			AddFix(eRecordType.StaticCommon, -9999, "Set OVRPlugin to Oculus Utilities-provided (OpenXR backend)", "Oculus recommends using OpenXR plugin provided with its Oculus Utilities package.\nYou can enable OpenXR backend for Oculus through the 'Oculus -> Tools -> OVR Utilities Plugin' menu.", null, null, false);
		}

		if (QualitySettings.anisotropicFiltering != AnisotropicFiltering.Enable && QualitySettings.anisotropicFiltering != AnisotropicFiltering.ForceEnable)
		{
			AddFix(eRecordType.StaticCommon, "Optimize Aniso", "Anisotropic filtering is recommended for optimal image sharpness and GPU performance.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				// Ideally this would be multi-option: offer Enable or ForceEnable.
				QualitySettings.anisotropicFiltering = AnisotropicFiltering.Enable;
			}, null, false, "Fix");
		}

#if UNITY_ANDROID
		int recommendedPixelLightCount = 1;
#else
		int recommendedPixelLightCount = 3;
#endif

		if (QualitySettings.pixelLightCount > recommendedPixelLightCount)
		{
			AddFix(eRecordType.StaticCommon, "Optimize Pixel Light Count", "For GPU performance set no more than " + recommendedPixelLightCount + " pixel lights in Quality Settings (currently " + QualitySettings.pixelLightCount + ").", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				QualitySettings.pixelLightCount = recommendedPixelLightCount;
			}, null, false, "Fix");
		}

#if false
		// Should we recommend this?  Seems to be mutually exclusive w/ dynamic batching.
		if (!PlayerSettings.graphicsJobs)
		{
			AddFix (eRecordType.StaticCommon, "Optimize Graphics Jobs", "For CPU performance, please use graphics jobs.", delegate(UnityEngine.Object obj, bool last, int selected)
			{
				PlayerSettings.graphicsJobs = true;
			}, null, false, "Fix");
		}
#endif

		if ((!PlayerSettings.MTRendering || !PlayerSettings.GetMobileMTRendering(BuildTargetGroup.Android)))
		{
			AddFix(eRecordType.StaticCommon, "Optimize MT Rendering", "For CPU performance, please enable multithreaded rendering.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				PlayerSettings.SetMobileMTRendering(BuildTargetGroup.Standalone, true);
				PlayerSettings.SetMobileMTRendering(BuildTargetGroup.Android, true);
			}, null, false, "Fix");
		}

#if UNITY_ANDROID
		if (!PlayerSettings.use32BitDisplayBuffer)
		{
			AddFix(eRecordType.StaticCommon, "Optimize Display Buffer Format", "We recommend to enable use32BitDisplayBuffer.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				PlayerSettings.use32BitDisplayBuffer = true;
			}, null, false, "Fix");
		}
#endif

#if !UNITY_ANDROID && !USING_XR_SDK && !REQUIRES_XR_SDK
#pragma warning disable 618
		if (!PlayerSettings.VROculus.dashSupport)
		{
			AddFix(eRecordType.StaticCommon, "Enable Dash Integration", "We recommend to enable Dash Integration for better user experience.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				PlayerSettings.VROculus.dashSupport = true;
			}, null, false, "Fix");
		}

		if (!PlayerSettings.VROculus.sharedDepthBuffer)
		{
			AddFix(eRecordType.StaticCommon, "Enable Depth Buffer Sharing", "We recommend to enable Depth Buffer Sharing for better user experience on Oculus Dash.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				PlayerSettings.VROculus.sharedDepthBuffer = true;
			}, null, false, "Fix");
		}
#pragma warning restore 618
#endif

		BuildTargetGroup target = EditorUserBuildSettings.selectedBuildTargetGroup;
		var tier = UnityEngine.Rendering.GraphicsTier.Tier1;
		var tierSettings = UnityEditor.Rendering.EditorGraphicsSettings.GetTierSettings(target, tier);

		if ((tierSettings.renderingPath == RenderingPath.DeferredShading ||
			tierSettings.renderingPath == RenderingPath.DeferredLighting))
		{
			AddFix(eRecordType.StaticCommon, "Optimize Rendering Path", "For CPU performance, please do not use deferred shading.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				tierSettings.renderingPath = RenderingPath.Forward;
				UnityEditor.Rendering.EditorGraphicsSettings.SetTierSettings(target, tier, tierSettings);
			}, null, false, "Use Forward");
		}

		if (PlayerSettings.stereoRenderingPath == StereoRenderingPath.MultiPass)
		{
			AddFix(eRecordType.StaticCommon, "Optimize Stereo Rendering", "For CPU performance, please enable single-pass or instanced stereo rendering.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				PlayerSettings.stereoRenderingPath = StereoRenderingPath.Instancing;
			}, null, false, "Fix");
		}

		if (LightmapSettings.lightmaps.Length > 0 && LightmapSettings.lightmapsMode != LightmapsMode.NonDirectional)
		{
			AddFix(eRecordType.StaticCommon, "Optimize Lightmap Directionality", "Switching from directional lightmaps to non-directional lightmaps can save a small amount of GPU time.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				LightmapSettings.lightmapsMode = LightmapsMode.NonDirectional;
			}, null, false, "Switch to non-directional lightmaps");
		}

		if (Lightmapping.realtimeGI)
		{
			AddFix(eRecordType.StaticCommon, "Disable Realtime GI", "Disabling real-time global illumination can improve GPU performance.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				Lightmapping.realtimeGI = false;
			}, null, false, "Set Lightmapping.realtimeGI = false.");
		}

		var lights = GameObject.FindObjectsOfType<Light>();
		for (int i = 0; i < lights.Length; ++i)
		{
			if (lights[i].type != LightType.Directional && !lights[i].bakingOutput.isBaked && IsLightBaked(lights[i]))
			{
				AddFix(eRecordType.StaticCommon, "Unbaked Lights", "The following lights in the scene are marked as Baked, but they don't have up to date lightmap data. Generate the lightmap data, or set it to auto-generate, in Window->Lighting->Settings.", null, lights[i], false, null);
			}

			if (lights[i].shadows != LightShadows.None && !IsLightBaked(lights[i]))
			{
				AddFix(eRecordType.StaticCommon, "Optimize Shadows", "For CPU performance, consider disabling shadows on realtime lights.", delegate (UnityEngine.Object obj, bool last, int selected)
				{
					Light thisLight = (Light)obj;
					thisLight.shadows = LightShadows.None;
				}, lights[i], false, "Set \"Shadow Type\" to \"No Shadows\"");
			}
		}

		var sources = GameObject.FindObjectsOfType<AudioSource>();
		if (sources.Length > 16)
		{
			List<AudioSource> playingAudioSources = new List<AudioSource>();
			foreach (var audioSource in sources)
			{
				if (audioSource.isPlaying)
				{
					playingAudioSources.Add(audioSource);
				}
			}

			if (playingAudioSources.Count > 16)
			{
				// Sort playing audio sources by priority
				playingAudioSources.Sort(delegate (AudioSource x, AudioSource y)
				{
					return x.priority.CompareTo(y.priority);
				});
				for (int i = 16; i < playingAudioSources.Count; ++i)
				{
					AddFix(eRecordType.StaticCommon, "Optimize Audio Source Count", "For CPU performance, please disable all but the top 16 AudioSources.", delegate (UnityEngine.Object obj, bool last, int selected)
					{
						AudioSource audioSource = (AudioSource)obj;
						audioSource.enabled = false;
					}, playingAudioSources[i], false, "Disable");
				}
			}
		}

		for (int i = 0; i < sources.Length; ++i)
		{
			AudioSource audioSource = sources[i];
			if (audioSource.clip.loadType == AudioClipLoadType.DecompressOnLoad)
			{
				AddFix(eRecordType.StaticCommon, "Audio Loading", "For fast loading, please don't use decompress on load for audio clips", delegate (UnityEngine.Object obj, bool last, int selected)
				{
					AudioClip thisClip = (AudioClip)obj;
					if (selected == 0)
					{
						SetAudioLoadType(thisClip, AudioClipLoadType.CompressedInMemory, last);
					}
					else
					{
						SetAudioLoadType(thisClip, AudioClipLoadType.Streaming, last);
					}

				}, audioSource.clip, false, "Change to Compressed in Memory", "Change to Streaming");
			}

#if UNITY_2022_2_OR_NEWER
			if (GetAudioPreload(audioSource.clip))
#else
			if (audioSource.clip.preloadAudioData)
#endif
			{
				AddFix(eRecordType.StaticCommon, "Audio Preload", "For fast loading, please don't preload data for audio clips.", delegate (UnityEngine.Object obj, bool last, int selected)
				{
					SetAudioPreload(audioSource.clip, false, last);
				}, audioSource.clip, false, "Fix");
			}
		}

		if (Physics.defaultContactOffset < 0.01f)
		{
			AddFix(eRecordType.StaticCommon, "Optimize Contact Offset", "For CPU performance, please don't use default contact offset below 0.01.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				Physics.defaultContactOffset = 0.01f;
			}, null, false, "Fix");
		}

		if (Physics.sleepThreshold < 0.005f)
		{
			AddFix(eRecordType.StaticCommon, "Optimize Sleep Threshold", "For CPU performance, please don't use sleep threshold below 0.005.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				Physics.sleepThreshold = 0.005f;
			}, null, false, "Fix");
		}

		if (Physics.defaultSolverIterations > 8)
		{
			AddFix(eRecordType.StaticCommon, "Optimize Solver Iterations", "For CPU performance, please don't use excessive solver iteration counts.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				Physics.defaultSolverIterations = 8;
			}, null, false, "Fix");
		}

		var materials = Resources.FindObjectsOfTypeAll<Material>();
		for (int i = 0; i < materials.Length; ++i)
		{
			if (materials[i].shader.name.Contains("Parallax") || materials[i].IsKeywordEnabled("_PARALLAXMAP"))
			{
				AddFix(eRecordType.StaticCommon, "Optimize Shading", "For GPU performance, please don't use parallax-mapped materials.", delegate (UnityEngine.Object obj, bool last, int selected)
				{
					Material thisMaterial = (Material)obj;
					if (thisMaterial.IsKeywordEnabled("_PARALLAXMAP"))
					{
						thisMaterial.DisableKeyword("_PARALLAXMAP");
					}

					if (thisMaterial.shader.name.Contains("Parallax"))
					{
						var newName = thisMaterial.shader.name.Replace("-ParallaxSpec", "-BumpSpec");
						newName = newName.Replace("-Parallax", "-Bump");
						var newShader = Shader.Find(newName);
						if (newShader)
						{
							thisMaterial.shader = newShader;
						}
						else
						{
							Debug.LogWarning("Unable to find a replacement for shader " + materials[i].shader.name);
						}
					}
				}, materials[i], false, "Fix");
			}
		}

		var renderers = GameObject.FindObjectsOfType<Renderer>();
		for (int i = 0; i < renderers.Length; ++i)
		{
			if (renderers[i].sharedMaterial == null)
			{
				AddFix(eRecordType.StaticCommon, "Instanced Materials", "Please avoid instanced materials on renderers.", null, renderers[i], false);
			}
		}

		var overlays = GameObject.FindObjectsOfType<OVROverlay>();
		if (overlays.Length > 4)
		{
			AddFix(eRecordType.StaticCommon, "Optimize VR Layer Count", "For GPU performance, please use 4 or fewer VR layers.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				for (int i = 4; i < OVROverlay.instances.Length; ++i)
				{
					OVROverlay.instances[i].enabled = false;
				}
			}, null, false, "Fix");
		}
		for (int i = 0; i < overlays.Length; i++)
		{
			if (overlays[i].useLegacyCubemapRotation)
			{
				AddFix(eRecordType.StaticCommon, "Fix Cubemap Orientation", "Legacy cubemap rotation will be deprecated in the future. Please fix the cubemap texture instead.", delegate (UnityEngine.Object obj, bool last, int selected)
				{
					OVROverlay thisOverlay = (OVROverlay)obj;
					thisOverlay.useLegacyCubemapRotation = false;
				}, overlays[i], false, "Remove Legacy Rotation");
			}
		}

		var splashScreen = PlayerSettings.virtualRealitySplashScreen;
		if (splashScreen != null)
		{
			if (splashScreen.filterMode != FilterMode.Trilinear)
			{
				AddFix(eRecordType.StaticCommon, "Optimize VR Splash Filtering", "For visual quality, please use trilinear filtering on your VR splash screen.", delegate (UnityEngine.Object obj, bool last, int EditorSelectedRenderState)
				{
					var assetPath = AssetDatabase.GetAssetPath(splashScreen);
					var importer = (TextureImporter)TextureImporter.GetAtPath(assetPath);
					importer.filterMode = FilterMode.Trilinear;
					AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceUpdate);
				}, null, false, "Fix");
			}

			if (splashScreen.mipmapCount <= 1)
			{
				AddFix(eRecordType.StaticCommon, "Generate VR Splash Mipmaps", "For visual quality, please use mipmaps with your VR splash screen.", delegate (UnityEngine.Object obj, bool last, int EditorSelectedRenderState)
				{
					var assetPath = AssetDatabase.GetAssetPath(splashScreen);
					var importer = (TextureImporter)TextureImporter.GetAtPath(assetPath);
					importer.mipmapEnabled = true;
					AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceUpdate);
				}, null, false, "Fix");
			}
		}
	}

	static void CheckRuntimeCommonIssues()
	{
		if (!OVRPlugin.occlusionMesh)
		{
			AddFix(eRecordType.RuntimeCommon, "Occlusion Mesh", "Enabling the occlusion mesh saves substantial GPU resources, generally with no visual impact. Enable unless you have an exceptional use case.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				OVRPlugin.occlusionMesh = true;
			}, null, false, "Set OVRPlugin.occlusionMesh = true");
		}

		if (OVRManager.instance != null && !OVRManager.instance.useRecommendedMSAALevel)
		{
			AddFix(eRecordType.RuntimeCommon, "Optimize MSAA", "OVRManager can select the optimal antialiasing for the installed hardware at runtime. Recommend enabling this.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				var ovrManagers = GameObject.FindObjectsOfType<OVRManager>();
				foreach (var ovrManager in ovrManagers)
				{
					ovrManager.useRecommendedMSAALevel = true;
				}
			}, null, true, "Stop Play and Fix");
		}

		if (UnityEngine.XR.XRSettings.eyeTextureResolutionScale > 1.5)
		{
			AddFix(eRecordType.RuntimeCommon, "Optimize Render Scale", "Render scale above 1.5 is extremely expensive on the GPU, with little if any positive visual benefit.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				UnityEngine.XR.XRSettings.eyeTextureResolutionScale = 1.5f;
			}, null, false, "Fix");
		}
	}

#if UNITY_ANDROID
	static void CheckStaticAndroidIssues()
	{
		if (OVRDeviceSelector.isTargetDeviceQuestFamily && PlayerSettings.Android.targetArchitectures != AndroidArchitecture.ARM64)
		{
			// Quest store is only accepting 64-bit apps as of November 25th 2019
			AddFix(eRecordType.StaticAndroid, "Set Target Architecture to ARM64", "32-bit Quest apps are no longer being accepted on the Oculus Store.",
				delegate (UnityEngine.Object obj, bool last, int selected)
				{
					PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
				}, null, false, "Fix");
		}

		// Check that the minSDKVersion meets requirement, 29 for Quest
		AndroidSdkVersions recommendedAndroidMinSdkVersion = AndroidSdkVersions.AndroidApiLevel29;
		if ((int)PlayerSettings.Android.minSdkVersion != (int)recommendedAndroidMinSdkVersion)
		{
			AddFix(eRecordType.StaticAndroid, "Set Min Android API Level", "Oculus Quest recommend setting minumum API level to " + (int)recommendedAndroidMinSdkVersion, delegate (UnityEngine.Object obj, bool last, int selected)
			{
				PlayerSettings.Android.minSdkVersion = recommendedAndroidMinSdkVersion;
			}, null, false, "Fix");
		}

		// Check that compileSDKVersion meets minimal version 26 as required for Quest's headtracking feature. Recommend 29 to match the MinSdkVersion. Set (and allow) Auto to align with Unity Setup Tool.
		// Unity Sets compileSDKVersion in Gradle as the value used in targetSdkVersion
		AndroidSdkVersions requiredAndroidTargetSdkVersion = AndroidSdkVersions.AndroidApiLevel29;
		if (OVRDeviceSelector.isTargetDeviceQuestFamily &&
		    PlayerSettings.Android.targetSdkVersion < recommendedAndroidMinSdkVersion && PlayerSettings.Android.targetSdkVersion != AndroidSdkVersions.AndroidApiLevelAuto)
		{
			AddFix(eRecordType.StaticAndroid, "Set Android Target SDK Level", "Oculus Quest apps recommend setting target API level to " +
			                                                                  (int)requiredAndroidTargetSdkVersion, delegate (UnityEngine.Object obj, bool last, int selected)
			{
				PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto;
			}, null, false, "Fix");
		}

		// Check that Android TV Compatibility is disabled
		if (PlayerSettings.Android.androidTVCompatibility)
		{
			AddFix(eRecordType.StaticAndroid, "Disable Android TV Compatibility", "Apps with Android TV Compatibility enabled are not accepted by the Oculus Store.",
				delegate (UnityEngine.Object obj, bool last, int selected)
				{
					PlayerSettings.Android.androidTVCompatibility = false;
				}, null, false, "Fix");
		}

		if (!PlayerSettings.gpuSkinning)
		{
			AddFix(eRecordType.StaticAndroid, "Optimize GPU Skinning", "If you are CPU-bound, consider using GPU skinning.",
				delegate (UnityEngine.Object obj, bool last, int selected)
			{
				PlayerSettings.gpuSkinning = true;
			}, null, false, "Fix");
		}

#if USING_XR_SDK
		if (OVRPluginInfo.IsOVRPluginOpenXRActivated() && PlayerSettings.colorSpace != ColorSpace.Linear)
		{
			AddFix(eRecordType.StaticAndroid, "Set Color Space to Linear", "Oculus Utilities Plugin with OpenXR only supports linear lighting.",
				delegate (UnityEngine.Object obj, bool last, int selected)
			{
				PlayerSettings.colorSpace = ColorSpace.Linear;
			}, null, false, "Fix");
		}
#endif

#if USING_XR_MANAGEMENT && USING_XR_SDK_OCULUS && OCULUS_XR_SYMMETRIC
		OculusSettings settings;
		UnityEditor.EditorBuildSettings.TryGetConfigObject<OculusSettings>("Unity.XR.Oculus.Settings", out settings);
		if (settings.SymmetricProjection)
		{
			AddFix(eRecordType.StaticAndroid, "Symmetric Projection Optimization", "Symmetric Projection is enabled in the Oculus XR Settings. To ensure best GPU performance, make sure at least FFR 1 is being used.", null, null, false);
		}
#endif

		if (RenderSettings.skybox)
		{
			AddFix(eRecordType.StaticAndroid, "Optimize Clearing", "For GPU performance, please don't use Unity's built-in Skybox.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				RenderSettings.skybox = null;
			}, null, false, "Clear Skybox");
		}

		var materials = Resources.FindObjectsOfTypeAll<Material>();
		for (int i = 0; i < materials.Length; ++i)
		{
			if (materials[i].IsKeywordEnabled("_SPECGLOSSMAP") || materials[i].IsKeywordEnabled("_METALLICGLOSSMAP"))
			{
				AddFix(eRecordType.StaticAndroid, "Optimize Specular Material", "For GPU performance, please don't use specular shader on materials.", delegate (UnityEngine.Object obj, bool last, int selected)
				{
					Material thisMaterial = (Material)obj;
					thisMaterial.DisableKeyword("_SPECGLOSSMAP");
					thisMaterial.DisableKeyword("_METALLICGLOSSMAP");
				}, materials[i], false, "Fix");
			}

			if (materials[i].passCount > 2)
			{
				AddFix(eRecordType.StaticAndroid, "Material Passes", "Please use 2 or fewer passes in materials.", null, materials[i], false);
			}
		}

		ScriptingImplementation backend = PlayerSettings.GetScriptingBackend(UnityEditor.BuildTargetGroup.Android);
		if (backend != UnityEditor.ScriptingImplementation.IL2CPP)
		{
			AddFix(eRecordType.StaticAndroid, "Optimize Scripting Backend", "For CPU performance, please use IL2CPP.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				PlayerSettings.SetScriptingBackend(UnityEditor.BuildTargetGroup.Android, UnityEditor.ScriptingImplementation.IL2CPP);
			}, null, false, "Fix");
		}

		var monoBehaviours = GameObject.FindObjectsOfType<MonoBehaviour>();
		System.Type effectBaseType = System.Type.GetType("UnityStandardAssets.ImageEffects.PostEffectsBase");
		if (effectBaseType != null)
		{
			for (int i = 0; i < monoBehaviours.Length; ++i)
			{
				if (monoBehaviours[i].GetType().IsSubclassOf(effectBaseType))
				{
					AddFix(eRecordType.StaticAndroid, "Image Effects", "Please don't use image effects.", null, monoBehaviours[i], false);
				}
			}
		}

		var textures = Resources.FindObjectsOfTypeAll<Texture2D>();

		int maxTextureSize = 1024 * (1 << QualitySettings.globalTextureMipmapLimit);
		maxTextureSize = maxTextureSize * maxTextureSize;

		for (int i = 0; i < textures.Length; ++i)
		{
			if (textures[i].filterMode == FilterMode.Trilinear && textures[i].mipmapCount == 1)
			{
				AddFix(eRecordType.StaticAndroid, "Optimize Texture Filtering", "For GPU performance, please generate mipmaps or disable trilinear filtering for textures.", delegate (UnityEngine.Object obj, bool last, int selected)
				{
					Texture2D thisTexture = (Texture2D)obj;
					if (selected == 0)
					{
						thisTexture.filterMode = FilterMode.Bilinear;
					}
					else
					{
						SetTextureUseMips(thisTexture, true, last);
					}
				}, textures[i], false, "Switch to Bilinear", "Generate Mipmaps");
			}
		}

		var projectors = GameObject.FindObjectsOfType<Projector>();
		if (projectors.Length > 0)
		{
			AddFix(eRecordType.StaticAndroid, "Optimize Projectors", "For GPU performance, please don't use projectors.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				Projector[] thisProjectors = GameObject.FindObjectsOfType<Projector>();
				for (int i = 0; i < thisProjectors.Length; ++i)
				{
					thisProjectors[i].enabled = false;
				}
			}, null, false, "Disable Projectors");
		}

		if (EditorUserBuildSettings.androidBuildSubtarget != MobileTextureSubtarget.ASTC && EditorUserBuildSettings.androidBuildSubtarget != MobileTextureSubtarget.ETC2)
		{
			AddFix(eRecordType.StaticAndroid, "Optimize Texture Compression", "For GPU performance, please use ASTC or ETC2.", delegate (UnityEngine.Object obj, bool last, int selected)
			{
				EditorUserBuildSettings.androidBuildSubtarget = MobileTextureSubtarget.ETC2;
			}, null, false, "Fix");
		}

		var cameras = GameObject.FindObjectsOfType<Camera>();
		int clearCount = 0;
		for (int i = 0; i < cameras.Length; ++i)
		{
			if (cameras[i].clearFlags != CameraClearFlags.Nothing && cameras[i].clearFlags != CameraClearFlags.Depth)
				++clearCount;
		}

		if (clearCount > 2)
		{
			AddFix(eRecordType.StaticAndroid, "Camera Clears", "Please use 2 or fewer clears.", null, null, false);
		}

		for (int i = 0; i < cameras.Length; ++i)
		{
			if (cameras[i].forceIntoRenderTexture)
			{
				AddFix(eRecordType.StaticAndroid, "Optimize Mobile Rendering", "For GPU performance, please don't enable forceIntoRenderTexture on your camera, this might be a flag pollution created by post process stack you used before, \nif your post process had already been turned off, we strongly encourage you to disable forceIntoRenderTexture. If you still want to use post process for some reasons, \nyou can leave this one on, but be warned, enabling this flag will introduce huge GPU performance cost. To view your flag status, please turn on you inspector's debug mode",
				delegate (UnityEngine.Object obj, bool last, int selected)
				{
					Camera thisCamera = (Camera)obj;
					thisCamera.forceIntoRenderTexture = false;
				}, cameras[i], false, "Disable forceIntoRenderTexture");
			}
		}
	}

	static void CheckRuntimeAndroidIssues()
	{
		if (UnityStats.usedTextureMemorySize + UnityStats.vboTotalBytes > 1000000)
		{
			AddFix(eRecordType.RuntimeAndroid, "Graphics Memory", "Please use less than 1GB of vertex and texture memory.", null, null, false);
		}

		// Remove the CPU/GPU level test, which won't work in Unity Editor

		if (UnityStats.triangles > 100000 || UnityStats.vertices > 100000)
		{
			AddFix(eRecordType.RuntimeAndroid, "Triangles and Verts", "Please use less than 100000 triangles or vertices.", null, null, false);
		}

		// Warn for 50 if in non-VR mode?
		if (UnityStats.drawCalls > 100)
		{
			AddFix(eRecordType.RuntimeAndroid, "Draw Calls", "Please use less than 100 draw calls.", null, null, false);
		}
	}

#endif // UNITY_ANDROID


	enum LightmapType { Realtime = 4, Baked = 2, Mixed = 1 };

	static bool IsLightBaked(Light light)
	{
		return light.lightmapBakeType == LightmapBakeType.Baked;
	}

#if UNITY_2022_2_OR_NEWER
	static void SetAudioPreload(AudioClip clip, bool preload, bool refreshImmediately)
	{
		if (clip != null)
		{
			string assetPath = AssetDatabase.GetAssetPath(clip);
			AudioImporter importer = AssetImporter.GetAtPath(assetPath) as AudioImporter;
			if (importer != null)
			{
				var audioSettings = importer.defaultSampleSettings;
				if (preload != audioSettings.preloadAudioData)
				{
					audioSettings.preloadAudioData = preload;

					importer.defaultSampleSettings = audioSettings;
					AssetDatabase.ImportAsset(assetPath);
					if (refreshImmediately)
					{
						AssetDatabase.Refresh();
					}
				}
			}
		}
	}

	static bool GetAudioPreload(AudioClip clip)
	{
		if (clip != null)
		{
			string assetPath = AssetDatabase.GetAssetPath(clip);
			AudioImporter importer = AssetImporter.GetAtPath(assetPath) as AudioImporter;
			if (importer != null)
			{
				return importer.defaultSampleSettings.preloadAudioData;
			}
		}
		return false;
	}
#else
	static void SetAudioPreload(AudioClip clip, bool preload, bool refreshImmediately)
	{
		if (clip != null)
		{
			string assetPath = AssetDatabase.GetAssetPath(clip);
			AudioImporter importer = AssetImporter.GetAtPath(assetPath) as AudioImporter;
			if (importer != null)
			{
				if (preload != importer.preloadAudioData)
				{
					importer.preloadAudioData = preload;

					AssetDatabase.ImportAsset(assetPath);
					if (refreshImmediately)
					{
						AssetDatabase.Refresh();
					}
				}
			}
		}
	}
#endif

	static void SetAudioLoadType(AudioClip clip, AudioClipLoadType loadType, bool refreshImmediately)
	{
		if (clip != null)
		{
			string assetPath = AssetDatabase.GetAssetPath(clip);
			AudioImporter importer = AssetImporter.GetAtPath(assetPath) as AudioImporter;
			if (importer != null)
			{
				if (loadType != importer.defaultSampleSettings.loadType)
				{
					AudioImporterSampleSettings settings = importer.defaultSampleSettings;
					settings.loadType = loadType;
					importer.defaultSampleSettings = settings;

					AssetDatabase.ImportAsset(assetPath);
					if (refreshImmediately)
					{
						AssetDatabase.Refresh();
					}
				}
			}
		}
	}

	public static void SetTextureUseMips(Texture texture, bool useMips, bool refreshImmediately)
	{
		if (texture != null)
		{
			string assetPath = AssetDatabase.GetAssetPath(texture);
			TextureImporter tImporter = AssetImporter.GetAtPath(assetPath) as TextureImporter;
			if (tImporter != null && tImporter.mipmapEnabled != useMips)
			{
				tImporter.mipmapEnabled = useMips;

				AssetDatabase.ImportAsset(assetPath);
				if (refreshImmediately)
				{
					AssetDatabase.Refresh();
				}
			}
		}
	}

	static T FindComponentInParents<T>(GameObject obj) where T : Component
	{
		T component = null;
		if (obj != null)
		{
			Transform parent = obj.transform.parent;
			if (parent != null)
			{
				do
				{
					component = parent.GetComponent(typeof(T)) as T;
					parent = parent.parent;
				} while (parent != null && component == null);
			}
		}
		return component;
	}
}

#endif
