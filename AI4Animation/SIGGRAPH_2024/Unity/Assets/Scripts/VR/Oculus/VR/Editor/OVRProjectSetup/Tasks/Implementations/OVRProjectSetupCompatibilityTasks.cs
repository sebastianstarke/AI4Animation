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
using UnityEditor;

[InitializeOnLoad]
internal static class OVRProjectSetupCompatibilityTasks
{
	public static bool IsTargetingARM64 => (PlayerSettings.Android.targetArchitectures & AndroidArchitecture.ARM64) != 0;
	public static readonly Action<BuildTargetGroup> SetARM64Target = (buildTargetGroup) => PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;

    static OVRProjectSetupCompatibilityTasks()
    {
        var compatibilityTaskGroup = OVRConfigurationTask.TaskGroup.Compatibility;

        // [Required] Platform has to be supported
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Required,
            group: compatibilityTaskGroup,
            isDone: OVRProjectSetup.IsPlatformSupported,
            conditionalMessage: buildTargetGroup =>
                OVRProjectSetup.IsPlatformSupported(buildTargetGroup) ?
                    $"Build Target ({buildTargetGroup}) is supported" :
                    $"Build Target ({buildTargetGroup}) is not supported"
        );

        // [Required] Android minimum level API
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Required,
            group: compatibilityTaskGroup,
            platform: BuildTargetGroup.Android,
            isDone: buildTargetGroup => PlayerSettings.Android.minSdkVersion >= AndroidSdkVersions.AndroidApiLevel29,
            message: "Minimum Android API Level must be at least 29",
            fix: buildTargetGroup => PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel29,
            fixMessage: "PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel29"
        );

        // [Required] Android target level API
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Recommended,
            group: compatibilityTaskGroup,
            platform: BuildTargetGroup.Android,
            isDone: buildTargetGroup => PlayerSettings.Android.targetSdkVersion == AndroidSdkVersions.AndroidApiLevelAuto || PlayerSettings.Android.targetSdkVersion >= AndroidSdkVersions.AndroidApiLevel29,
            message: "Target API should be set to \"Automatic\" as to ensure latest version",
            fix: buildTargetGroup => PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto,
            fixMessage: "PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto"
        );

        // [Required] Install Location
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Recommended,
            group: compatibilityTaskGroup,
            platform: BuildTargetGroup.Android,
            isDone: buildTargetGroup => PlayerSettings.Android.preferredInstallLocation == AndroidPreferredInstallLocation.Auto,
            message: "Install Location should be set to \"Automatic\"",
            fix: buildTargetGroup => PlayerSettings.Android.preferredInstallLocation = AndroidPreferredInstallLocation.Auto,
            fixMessage: "PlayerSettings.Android.preferredInstallLocation = AndroidPreferredInstallLocation.Auto"
        );

        // [Required] Generate Android Manifest
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Optional,
            group: compatibilityTaskGroup,
            platform: BuildTargetGroup.Android,
            isDone: buildTargetGroup => OVRManifestPreprocessor.DoesAndroidManifestExist(),
            message: "An Android Manifest file is required",
            fix: buildTargetGroup => OVRManifestPreprocessor.GenerateManifestForSubmission(),
            fixMessage: "Generates a default Manifest file"
        );

        // ConfigurationTask : IL2CPP when ARM64
        OVRProjectSetup.AddTask(
            conditionalLevel: buildTargetGroup => IsTargetingARM64 ?
                OVRConfigurationTask.TaskLevel.Required :
                OVRConfigurationTask.TaskLevel.Recommended,
            group: compatibilityTaskGroup,
            platform: BuildTargetGroup.Android,
            isDone: buildTargetGroup => PlayerSettings.GetScriptingBackend(buildTargetGroup) == ScriptingImplementation.IL2CPP,
            conditionalMessage: buildTargetGroup => IsTargetingARM64 ?
                "Building the ARM64 architecture requires using IL2CPP as the scripting backend" :
                "Using IL2CPP as the scripting backend is recommended",
            fix: buildTargetGroup => PlayerSettings.SetScriptingBackend(buildTargetGroup, ScriptingImplementation.IL2CPP),
            fixMessage: "PlayerSettings.SetScriptingBackend(buildTargetGroup, ScriptingImplementation.IL2CPP)"
        );

        // ConfigurationTask : ARM64 is recommended
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Required,
            group: compatibilityTaskGroup,
            platform: BuildTargetGroup.Android,
            isDone: buildTargetGroup => IsTargetingARM64,
            message: "Use ARM64 as target architecture",
            fix: SetARM64Target,
            fixMessage: "PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64"
        );

        // ConfigurationTask : No Alpha or Beta for production
        // This is a task that CANNOT BE FIXED
        OVRProjectSetup.AddTask(
	        level: OVRConfigurationTask.TaskLevel.Recommended,
	        group: compatibilityTaskGroup,
	        isDone: group => !OVRManager.IsUnityAlphaOrBetaVersion(),
	        message: "We recommend using a stable version for Oculus Development"
	    );

        // ConfigurationTask : Check that Android TV Compatibility is disabled
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Required,
            platform: BuildTargetGroup.Android,
            group: compatibilityTaskGroup,
            isDone: group => !PlayerSettings.Android.androidTVCompatibility,
            message: "Apps with Android TV Compatibility enabled are not accepted by the Oculus Store",
            fix: group => PlayerSettings.Android.androidTVCompatibility = false,
            fixMessage: "PlayerSettings.Android.androidTVCompatibility = false"
        );
    }
}
