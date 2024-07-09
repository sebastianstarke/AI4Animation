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

using UnityEditor;
using UnityEngine;

[InitializeOnLoad]
internal static class OVRProjectSetupQualityTasks
{
    private static readonly int RecommendedPixelLightCountAndroid = 1;
    private static readonly int RecommendedPixelLightCountStandalone = 3;

    private static int GetRecommendedPixelLightCount(BuildTargetGroup buildTargetGroup)
        => buildTargetGroup == BuildTargetGroup.Standalone
        ? RecommendedPixelLightCountStandalone
        : RecommendedPixelLightCountAndroid;

    static OVRProjectSetupQualityTasks()
    {
        var taskGroup = OVRConfigurationTask.TaskGroup.Quality;

        // [Recommended] Set Pixel Light Count
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Recommended,
            group: taskGroup,
            isDone: buildTargetGroup => QualitySettings.pixelLightCount <= GetRecommendedPixelLightCount(buildTargetGroup),
            conditionalMessage: buildTargetGroup => $"Set maximum pixel lights count to {GetRecommendedPixelLightCount(buildTargetGroup)}",
            fix: buildTargetGroup => QualitySettings.pixelLightCount = GetRecommendedPixelLightCount(buildTargetGroup),
            conditionalFixMessage: buildTargetGroup => $"QualitySettings.pixelLightCount = {GetRecommendedPixelLightCount(buildTargetGroup)}"
            );

        // [Recommended] Set Texture Quality to Full Res
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Recommended,
            group: taskGroup,
            isDone: buildTargetGroup => QualitySettings.globalTextureMipmapLimit == 0,
            message: "Set Texture Quality to Full Res",
            fix: buildTargetGroup => QualitySettings.globalTextureMipmapLimit = 0,
            fixMessage: "QualitySettings.masterTextureLimit = 0"
        );

        // [Recommended] Enable Anisotropic Filtering
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Recommended,
            group: taskGroup,
            isDone: buildTargetGroup => QualitySettings.anisotropicFiltering == AnisotropicFiltering.Enable,
            message: "Enable Anisotropic Filtering on a per-texture basis",
            fix: buildTargetGroup => QualitySettings.anisotropicFiltering = AnisotropicFiltering.Enable,
            fixMessage: "QualitySettings.anisotropicFiltering = AnisotropicFiltering.Enable"
        );

        // Texture compression : Use ASTC
        OVRProjectSetup.AddTask(
	        level: OVRConfigurationTask.TaskLevel.Recommended,
	        group: taskGroup,
	        platform: BuildTargetGroup.Android,
	        isDone: group => EditorUserBuildSettings.androidBuildSubtarget == MobileTextureSubtarget.ASTC || EditorUserBuildSettings.androidBuildSubtarget == MobileTextureSubtarget.ETC2,
	        message: "Optimize Texture Compression : For GPU performance, please use ETC2. In some cases, ASTC may produce better visuals and is also a viable solution.",
	        fix: group => EditorUserBuildSettings.androidBuildSubtarget = MobileTextureSubtarget.ETC2,
	        fixMessage: "EditorUserBuildSettings.androidBuildSubtarget = MobileTextureSubtarget.ETC2"
	    );
    }
}
