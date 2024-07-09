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

[CustomEditor(typeof(OVRManager))]
public class OVRManagerEditor : Editor
{
	private SerializedProperty _requestBodyTrackingPermissionOnStartup;
	private SerializedProperty _requestFaceTrackingPermissionOnStartup;
	private SerializedProperty _requestEyeTrackingPermissionOnStartup;
	private bool _expandPermissionsRequest;


	void OnEnable()
	{
		_requestBodyTrackingPermissionOnStartup = serializedObject.FindProperty(nameof(OVRManager.requestBodyTrackingPermissionOnStartup));
		_requestFaceTrackingPermissionOnStartup = serializedObject.FindProperty(nameof(OVRManager.requestFaceTrackingPermissionOnStartup));
		_requestEyeTrackingPermissionOnStartup = serializedObject.FindProperty(nameof(OVRManager.requestEyeTrackingPermissionOnStartup));

	}

	public override void OnInspectorGUI()
	{
		serializedObject.ApplyModifiedProperties();
		OVRRuntimeSettings runtimeSettings = OVRRuntimeSettings.GetRuntimeSettings();
		OVRProjectConfig projectConfig = OVRProjectConfig.GetProjectConfig();

#if UNITY_ANDROID
		OVRProjectConfigEditor.DrawTargetDeviceInspector(projectConfig);
		EditorGUILayout.Space();
#endif

		DrawDefaultInspector();

		bool modified = false;

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || UNITY_ANDROID
		OVRManager manager = (OVRManager)target;

		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Display", EditorStyles.boldLabel);

		OVRManager.ColorSpace colorGamut = runtimeSettings.colorSpace;
		OVREditorUtil.SetupEnumField(target, new GUIContent("Color Gamut",
				"The target color gamut when displayed on the HMD"), ref colorGamut, ref modified,
			"https://developer.oculus.com/documentation/unity/unity-color-space/");
		manager.colorGamut = colorGamut;

		if (modified)
		{
			runtimeSettings.colorSpace = colorGamut;
			OVRRuntimeSettings.CommitRuntimeSettings(runtimeSettings);
		}
#endif

		EditorGUILayout.Space();
		OVRProjectConfigEditor.DrawProjectConfigInspector(projectConfig);

#if UNITY_ANDROID
		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Mixed Reality Capture for Quest", EditorStyles.boldLabel);
		EditorGUI.indentLevel++;
		OVREditorUtil.SetupEnumField(target, "ActivationMode", ref manager.mrcActivationMode, ref modified);
		EditorGUI.indentLevel--;
#endif

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
		EditorGUILayout.Space();
		EditorGUILayout.BeginHorizontal();
		manager.expandMixedRealityCapturePropertySheet = EditorGUILayout.BeginFoldoutHeaderGroup(manager.expandMixedRealityCapturePropertySheet, "Mixed Reality Capture");
		OVREditorUtil.DisplayDocLink("https://developer.oculus.com/documentation/unity/unity-mrc/");
		EditorGUILayout.EndHorizontal();
		if (manager.expandMixedRealityCapturePropertySheet)
		{
			string[] layerMaskOptions = new string[32];
			for (int i=0; i<32; ++i)
			{
				layerMaskOptions[i] = LayerMask.LayerToName(i);
				if (layerMaskOptions[i].Length == 0)
				{
					layerMaskOptions[i] = "<Layer " + i.ToString() + ">";
				}
			}

			EditorGUI.indentLevel++;

			OVREditorUtil.SetupBoolField(target, "Enable MixedRealityCapture", ref manager.enableMixedReality, ref modified);
			OVREditorUtil.SetupEnumField(target, "Composition Method", ref manager.compositionMethod, ref modified);
			OVREditorUtil.SetupLayerMaskField(target, "Extra Hidden Layers", ref manager.extraHiddenLayers, layerMaskOptions, ref modified);
			OVREditorUtil.SetupLayerMaskField(target, "Extra Visible Layers", ref manager.extraVisibleLayers, layerMaskOptions, ref modified);
			OVREditorUtil.SetupBoolField(target, "Dynamic Culling Mask", ref manager.dynamicCullingMask, ref modified);

			// CompositionMethod.External is the only composition method that is available.
			// All other deprecated composition methods should fallback to the path below.
			{
				// CompositionMethod.External
				EditorGUILayout.Space();
				EditorGUILayout.LabelField("External Composition", EditorStyles.boldLabel);
				EditorGUI.indentLevel++;

				OVREditorUtil.SetupColorField(target, "Backdrop Color (Target, Rift)", ref manager.externalCompositionBackdropColorRift, ref modified);
				OVREditorUtil.SetupColorField(target, "Backdrop Color (Target, Quest)", ref manager.externalCompositionBackdropColorQuest, ref modified);
				EditorGUI.indentLevel--;
			}

			EditorGUI.indentLevel--;
		}
		EditorGUILayout.EndFoldoutHeaderGroup();
#endif

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || UNITY_ANDROID
		// Insight Passthrough section
#if UNITY_ANDROID
		bool passthroughCapabilityEnabled =
			projectConfig.insightPassthroughSupport != OVRProjectConfig.FeatureSupport.None;
		EditorGUI.BeginDisabledGroup(!passthroughCapabilityEnabled);
		GUIContent enablePassthroughContent = new GUIContent("Enable Passthrough", "Enables passthrough functionality for the scene. Can be toggled at runtime. Passthrough Capability must be enabled in the project settings.");
#else
		GUIContent enablePassthroughContent = new GUIContent("Enable Passthrough", "Enables passthrough functionality for the scene. Can be toggled at runtime.");
#endif
		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Insight Passthrough", EditorStyles.boldLabel);
#if UNITY_ANDROID
		if (!passthroughCapabilityEnabled) {
			EditorGUILayout.LabelField("Requires Passthrough Capability to be enabled in the General section of the Quest features.", EditorStyles.wordWrappedLabel);
		}
#endif
		OVREditorUtil.SetupBoolField(target, enablePassthroughContent, ref manager.isInsightPassthroughEnabled, ref modified);
#if UNITY_ANDROID
		EditorGUI.EndDisabledGroup();
#endif
#endif

		#region PermissionRequests

		EditorGUILayout.Space();
		_expandPermissionsRequest =
			EditorGUILayout.BeginFoldoutHeaderGroup(_expandPermissionsRequest, "Permission Requests On Startup");
		if (_expandPermissionsRequest)
		{
			void AddPermissionGroup(bool featureEnabled, string permissionName, SerializedProperty property)
			{
				using (new EditorGUI.DisabledScope(!featureEnabled))
				{
					if (!featureEnabled)
					{
						EditorGUILayout.LabelField($"Requires {permissionName} Capability to be enabled in the Quest features section.",
							EditorStyles.wordWrappedLabel);
					}

					var label = new GUIContent(permissionName,
						$"Requests {permissionName} permission on start up. {permissionName} Capability must be enabled in the project settings.");
					EditorGUILayout.PropertyField(property, label);
				}
			}

			AddPermissionGroup(projectConfig.bodyTrackingSupport != OVRProjectConfig.FeatureSupport.None, "Body Tracking", _requestBodyTrackingPermissionOnStartup);
			AddPermissionGroup(projectConfig.faceTrackingSupport != OVRProjectConfig.FeatureSupport.None, "Face Tracking", _requestFaceTrackingPermissionOnStartup);
			AddPermissionGroup(projectConfig.eyeTrackingSupport != OVRProjectConfig.FeatureSupport.None, "Eye Tracking", _requestEyeTrackingPermissionOnStartup);
		}
		EditorGUILayout.EndFoldoutHeaderGroup();

		#endregion



		if (modified)
		{
			EditorUtility.SetDirty(target);
		}

		serializedObject.ApplyModifiedProperties();
	}
}
