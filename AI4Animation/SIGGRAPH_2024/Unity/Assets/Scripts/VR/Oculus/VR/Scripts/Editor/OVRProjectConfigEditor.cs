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
using Oculus.VR.Editor;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(OVRProjectConfig))]
public class OVRProjectConfigEditor : Editor
{
    override public void OnInspectorGUI()
    {
        OVRProjectConfig projectConfig = (OVRProjectConfig)target;
        DrawTargetDeviceInspector(projectConfig);
        EditorGUILayout.Space();
        DrawProjectConfigInspector(projectConfig);
    }

    public static void DrawTargetDeviceInspector(OVRProjectConfig projectConfig)
    {
        // Target Devices
        EditorGUILayout.LabelField("Target Devices", EditorStyles.boldLabel);
        bool useOculusXRSettings = false;

#if PRIORITIZE_OCULUS_XR_SETTINGS
        EditorGUILayout.LabelField("Configure Target Devices in Oculus XR Plugin Settings:", GUILayout.Width(320));
        if (GUILayout.Button("Open Settings"))
            SettingsService.OpenProjectSettings("Project/XR Plug-in Management/Oculus");
#if !USING_QUEST_PRO_COMPATIBLE_OCULUS_XR_PLUGIN_VERSION
        ++EditorGUI.indentLevel;
        EditorGUILayout.LabelField("Note: The currently installed Oculus XR Plugin version does not support "
               + "configuring Quest Pro as a target device. Please install a compatible version:"
               + "\n2.2.0-preview.1 or newer\n3.2.1 or newer", EditorStyles.wordWrappedMiniLabel);
        --EditorGUI.indentLevel;
#endif
        useOculusXRSettings = true;
#endif

        if (!useOculusXRSettings)
        {
            bool hasModified = false;

            foreach (OVRProjectConfig.DeviceType deviceType in System.Enum.GetValues(typeof(OVRProjectConfig.DeviceType)))
            {
                bool oldSupportsDevice = projectConfig.targetDeviceTypes.Contains(deviceType);
                bool newSupportsDevice = oldSupportsDevice;
                OVREditorUtil.SetupBoolField(projectConfig, ObjectNames.NicifyVariableName(deviceType.ToString()), ref newSupportsDevice, ref hasModified);

                if (newSupportsDevice && !oldSupportsDevice)
                {
                    projectConfig.targetDeviceTypes.Add(deviceType);
                }
                else if (oldSupportsDevice && !newSupportsDevice)
                {
                    projectConfig.targetDeviceTypes.Remove(deviceType);
                }
            }

            if (hasModified)
            {
                OVRProjectConfig.CommitProjectConfig(projectConfig);
            }
        }
    }

    enum eProjectConfigTab
    {
        General = 0,
        BuildSettings,
        Security,
        Experimental,
    }
    static eProjectConfigTab selectedTab = 0;
    static string[] projectConfigTabStrs = null;

    public static void DrawProjectConfigInspector(OVRProjectConfig projectConfig)
    {
        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        EditorGUILayout.LabelField("Quest Features", EditorStyles.boldLabel);

        if (EditorUserBuildSettings.activeBuildTarget != UnityEditor.BuildTarget.Android)
        {
            EditorGUILayout.LabelField($"Your current platform is \"{EditorUserBuildSettings.activeBuildTarget}\". These settings only apply if your active platform is \"Android\".", EditorStyles.wordWrappedMiniLabel);
        }

        if (projectConfigTabStrs == null)
        {
            projectConfigTabStrs = Enum.GetNames(typeof(eProjectConfigTab));
            for (int i = 0; i < projectConfigTabStrs.Length; ++i)
                projectConfigTabStrs[i] = ObjectNames.NicifyVariableName(projectConfigTabStrs[i]);
        }

        selectedTab = (eProjectConfigTab)GUILayout.SelectionGrid((int)selectedTab, projectConfigTabStrs, 3, GUI.skin.button);
        EditorGUILayout.Space(5);
        bool hasModified = false;

        switch (selectedTab)
        {
            case eProjectConfigTab.General:

                // Show overlay support option
                using (new EditorGUI.DisabledScope(true))
                {
                    EditorGUILayout.Toggle(new GUIContent("Focus Aware (Required)",
                        "If checked, the new overlay will be displayed when the user presses the home button. The game will not be paused, but will now receive InputFocusLost and InputFocusAcquired events."), true);
                }

                // Hand Tracking Support
                OVREditorUtil.SetupEnumField(projectConfig, "Hand Tracking Support", ref projectConfig.handTrackingSupport, ref hasModified);

                OVREditorUtil.SetupEnumField(projectConfig, new GUIContent("Hand Tracking Frequency",
                        "Note that a higher tracking frequency will reserve some performance headroom from the application's budget."),
                    ref projectConfig.handTrackingFrequency, ref hasModified, "https://developer.oculus.com/documentation/unity/unity-handtracking/#enable-hand-tracking");

                OVREditorUtil.SetupEnumField(projectConfig, "Hand Tracking Version", ref projectConfig.handTrackingVersion, ref hasModified);

                // Enable Render Model Support
                bool renderModelSupportAvailable = OVRPluginInfo.IsOVRPluginOpenXRActivated();
                EditorGUI.BeginDisabledGroup(!renderModelSupportAvailable);
                if (!renderModelSupportAvailable)
                {
                    projectConfig.renderModelSupport = OVRProjectConfig.RenderModelSupport.Disabled;
                }
                OVREditorUtil.SetupEnumField(projectConfig, new GUIContent("Render Model Support",
                        "If enabled, the application will be able to load render models from the runtime."),
                    ref projectConfig.renderModelSupport, ref hasModified);
                if (hasModified && projectConfig.renderModelSupport == OVRProjectConfig.RenderModelSupport.Disabled)
                {
                    projectConfig.trackedKeyboardSupport = OVRProjectConfig.TrackedKeyboardSupport.None;
                }
                EditorGUI.EndDisabledGroup();

                // System Keyboard Support
                OVREditorUtil.SetupBoolField(projectConfig, new GUIContent("Requires System Keyboard",
                        "If checked, the Oculus System keyboard will be enabled for Unity input fields and any calls to open/close the Unity TouchScreenKeyboard."),
                    ref projectConfig.requiresSystemKeyboard, ref hasModified);

                // Tracked Keyboard Support
                bool trackedKeyboardSupportAvailable = OVRPluginInfo.IsOVRPluginOpenXRActivated();
                EditorGUI.BeginDisabledGroup(!trackedKeyboardSupportAvailable);
                if (!trackedKeyboardSupportAvailable)
                {
                    projectConfig.trackedKeyboardSupport = OVRProjectConfig.TrackedKeyboardSupport.None;
                }
                OVREditorUtil.SetupEnumField(projectConfig, new GUIContent("Tracked Keyboard Support",
                        "Show user's physical keyboard in correct position in VR."),
                    ref projectConfig.trackedKeyboardSupport, ref hasModified);

                // Anchor Support
                using (new EditorGUI.DisabledScope(projectConfig.sharedAnchorSupport !=
                                                   OVRProjectConfig.FeatureSupport.None))
                {
                    var tooltip = projectConfig.sharedAnchorSupport != OVRProjectConfig.FeatureSupport.None
                        ? "Anchor Support is required for Shared Spatial Anchor Support."
                        : "";
                    OVREditorUtil.SetupEnumField(projectConfig, new GUIContent("Anchor Support", tooltip), ref projectConfig.anchorSupport, ref hasModified);
                }

                OVREditorUtil.SetupEnumField(projectConfig,
                    new GUIContent("Shared Spatial Anchor Support", "Enables support for sharing spatial anchors with other users. This requires Anchor Support to be enabled."),
                    ref projectConfig.sharedAnchorSupport, ref hasModified);

                if (projectConfig.sharedAnchorSupport != OVRProjectConfig.FeatureSupport.None &&
                    projectConfig.anchorSupport != OVRProjectConfig.AnchorSupport.Enabled)
                {
                    projectConfig.anchorSupport = OVRProjectConfig.AnchorSupport.Enabled;
                    hasModified = true;
                }


                // Body Tracking Support
                OVREditorUtil.SetupEnumField(projectConfig, "Body Tracking Support", ref projectConfig.bodyTrackingSupport, ref hasModified);

                // Face Tracking Support
                OVREditorUtil.SetupEnumField(projectConfig, "Face Tracking Support", ref projectConfig.faceTrackingSupport, ref hasModified);

                // Eye Tracking Support
                OVREditorUtil.SetupEnumField(projectConfig, "Eye Tracking Support", ref projectConfig.eyeTrackingSupport, ref hasModified);

                if (hasModified && projectConfig.trackedKeyboardSupport != OVRProjectConfig.TrackedKeyboardSupport.None)
                {
                    projectConfig.renderModelSupport = OVRProjectConfig.RenderModelSupport.Enabled;
                }
                if (!OVRPluginInfo.IsOVRPluginOpenXRActivated())
                {
                    EditorGUILayout.HelpBox(
                        "The OpenXR backend must be enabled in the Oculus menu to use the Render Model and Tracked Keyboard features.",
                        MessageType.Info);
                }
                if (projectConfig.trackedKeyboardSupport != OVRProjectConfig.TrackedKeyboardSupport.None && projectConfig.renderModelSupport == OVRProjectConfig.RenderModelSupport.Disabled)
                {
                    EditorGUILayout.HelpBox(
                        "Render model support is required to load keyboard models from the runtime.",
                        MessageType.Error);
                }
                EditorGUI.EndDisabledGroup();

                // System Splash Screen
                OVREditorUtil.SetupTexture2DField(projectConfig, new GUIContent("System Splash Screen",
                        "If set, the Splash Screen will be presented by the Operating System as a high quality composition layer at launch time."),
                    ref projectConfig.systemSplashScreen, ref hasModified,
                    "https://developer.oculus.com/documentation/unity/unity-splash-screen/");

                // Allow optional 3-dof head-tracking
                OVREditorUtil.SetupBoolField(projectConfig, new GUIContent("Allow Optional 3DoF Head Tracking",
                        "If checked, application can work in both 6DoF and 3DoF modes. It's highly recommended to keep it unchecked unless your project strongly needs the 3DoF head tracking."),
                    ref projectConfig.allowOptional3DofHeadTracking, ref hasModified);

                // Passthrough support
                OVREditorUtil.SetupEnumField(projectConfig, new GUIContent("Passthrough Support",
                    "Allows the application to use passthrough functionality. This option must be enabled at build time, otherwise initializing passthrough and creating passthrough layers in application scenes will fail."),
                    ref projectConfig._insightPassthroughSupport, ref hasModified);

                break;

            case eProjectConfigTab.BuildSettings:

                OVREditorUtil.SetupBoolField(projectConfig, new GUIContent("Skip Unneeded Shaders",
                        "If checked, prevent building shaders that are not used by default to reduce time spent when building."),
                    ref projectConfig.skipUnneededShaders, ref hasModified,
                    "https://developer.oculus.com/documentation/unity/unity-strip-shaders/");

                break;

            case eProjectConfigTab.Security:

                OVREditorUtil.SetupBoolField(projectConfig, "Disable Backups", ref projectConfig.disableBackups, ref hasModified,
                    "https://developer.android.com/guide/topics/data/autobackup#EnablingAutoBackup");
                OVREditorUtil.SetupBoolField(projectConfig, "Enable NSC Configuration", ref projectConfig.enableNSCConfig, ref hasModified,
                    "https://developer.android.com/training/articles/security-config");
                EditorGUI.BeginDisabledGroup(!projectConfig.enableNSCConfig);
                ++EditorGUI.indentLevel;
                OVREditorUtil.SetupInputField(projectConfig, "Custom Security XML Path", ref projectConfig.securityXmlPath, ref hasModified);
                --EditorGUI.indentLevel;
                EditorGUI.EndDisabledGroup();

                break;

            case eProjectConfigTab.Experimental:

                // Experimental Features Enabled
                OVREditorUtil.SetupBoolField(projectConfig, new GUIContent("Experimental Features Enabled",
                        "If checked, this application can use experimental features. Note that such features are for developer use only. This option must be disabled when submitting to the Oculus Store."),
                    ref projectConfig.experimentalFeaturesEnabled, ref hasModified);


                break;
        }

        EditorGUILayout.EndVertical();

        // apply any pending changes to project config
        if (hasModified)
        {
            OVRProjectConfig.CommitProjectConfig(projectConfig);
        }
    }
}
