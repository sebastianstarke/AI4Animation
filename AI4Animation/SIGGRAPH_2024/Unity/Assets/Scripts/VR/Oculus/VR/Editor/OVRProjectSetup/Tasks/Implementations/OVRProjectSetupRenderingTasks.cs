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
using System.Linq;
using UnityEditor;
using UnityEditor.Rendering;
using UnityEngine;
using UnityEngine.Rendering;

[InitializeOnLoad]
internal static class OVRProjectSetupRenderingTasks
{
#if USING_XR_SDK_OCULUS
    private static Unity.XR.Oculus.OculusSettings OculusSettings
    {
        get
        {
            UnityEditor.EditorBuildSettings.TryGetConfigObject<Unity.XR.Oculus.OculusSettings>("Unity.XR.Oculus.Settings", out var settings);
            return settings;
        }
    }
#endif

	private static GraphicsDeviceType[] GetGraphicsAPIs(BuildTargetGroup buildTargetGroup)
	{
		var buildTarget = buildTargetGroup.GetBuildTarget();
		if (PlayerSettings.GetUseDefaultGraphicsAPIs(buildTarget))
		{
			return Array.Empty<GraphicsDeviceType>();;
		}

		// Recommends OpenGL ES 3 or Vulkan
		return PlayerSettings.GetGraphicsAPIs(buildTarget);
	}

    static OVRProjectSetupRenderingTasks()
    {
        const OVRConfigurationTask.TaskGroup targetGroup = OVRConfigurationTask.TaskGroup.Rendering;

        //[Required] Set the color space to linear
        OVRProjectSetup.AddTask(
            conditionalLevel: buildTargetGroup => OVRProjectSetupUtils.IsPackageInstalled(OVRProjectSetupXRTasks.UnityXRPackage) ? OVRConfigurationTask.TaskLevel.Required : OVRConfigurationTask.TaskLevel.Recommended,
            group: targetGroup,
            isDone: buildTargetGroup => PlayerSettings.colorSpace == ColorSpace.Linear,
            message: "Color Space is required to be Linear",
            fix: buildTargetGroup => PlayerSettings.colorSpace = ColorSpace.Linear,
            fixMessage: "PlayerSettings.colorSpace = ColorSpace.Linear"
        );

        //[Required] Use Graphics Jobs
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Recommended,
            group: targetGroup,
            isDone: buildTargetGroup => !PlayerSettings.graphicsJobs,
            message: "Disable Graphics Jobs",
            fix: buildTargetGroup => PlayerSettings.graphicsJobs = false,
            fixMessage: "PlayerSettings.graphicsJobs = false"
        );

        //[Recommended] Set the Graphics API order
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Recommended,
            group: targetGroup,
            isDone: buildTargetGroup =>
	            GetGraphicsAPIs(buildTargetGroup).Any(item => item == GraphicsDeviceType.OpenGLES3 || item == GraphicsDeviceType.Vulkan),
            message: "Manual selection of Graphic API, favoring Vulkan (or OpenGLES3)",
            fix: buildTargetGroup =>
            {
                var buildTarget = buildTargetGroup.GetBuildTarget();
                PlayerSettings.SetUseDefaultGraphicsAPIs(buildTarget, false);
                PlayerSettings.SetGraphicsAPIs(buildTarget, new [] { GraphicsDeviceType.Vulkan });
            },
            fixMessage: "Set Graphics APIs for this build target to Vulkan"
        );

        //[Recommended] Enable Multithreaded Rendering
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Recommended,
            group: targetGroup,
            isDone: buildTargetGroup => PlayerSettings.MTRendering &&
                                        (buildTargetGroup != BuildTargetGroup.Android
                                         || PlayerSettings.GetMobileMTRendering(buildTargetGroup)),
            message: "Enable Multithreaded Rendering",
            fix: buildTargetGroup =>
            {
                PlayerSettings.MTRendering = true;
                if (buildTargetGroup == BuildTargetGroup.Android)
                {
                    PlayerSettings.SetMobileMTRendering(buildTargetGroup, true);
                }
            },
            conditionalFixMessage: buildTargetGroup =>
	            buildTargetGroup == BuildTargetGroup.Android ?
		            "PlayerSettings.MTRendering = true and PlayerSettings.SetMobileMTRendering(buildTargetGroup, true)"
		            : "PlayerSettings.MTRendering = true"
        );

#if USING_XR_SDK_OCULUS
        //[Recommended] Select Low Overhead Mode
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Recommended,
            conditionalValidity: buildTargetGroup =>
	            GetGraphicsAPIs(buildTargetGroup).Contains(GraphicsDeviceType.OpenGLES3),
            group: targetGroup,
            platform: BuildTargetGroup.Android,
            isDone: buildTargetGroup => OculusSettings?.LowOverheadMode ?? true,
            message: "Use Low Overhead Mode",
            fix: buildTargetGroup =>
            {
                var setting = OculusSettings;
                if (setting != null)
                {
                    setting.LowOverheadMode = true;
                }
            },
            fixMessage: "OculusSettings.LowOverheadMode = true"
        );

        //[Recommended] Enable Dash Support
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Recommended,
            group: targetGroup,
            platform: BuildTargetGroup.Standalone,
            isDone: buildTargetGroup => OculusSettings?.DashSupport ?? true,
            message: "Enable Dash Support",
            fix: buildTargetGroup =>
            {
                var setting = OculusSettings;
                if (setting != null)
                {
                    setting.DashSupport = true;
                }
            },
            fixMessage: "OculusSettings.DashSupport = true"
        );
#endif

        //[Recommended] Set the Display Buffer Format to 32 bit
        OVRProjectSetup.AddTask(
	        level: OVRConfigurationTask.TaskLevel.Recommended,
	        group: targetGroup,
	        isDone: buildTargetGroup =>
		        PlayerSettings.use32BitDisplayBuffer,
	        message: "Use 32Bit Display Buffer",
	        fix: buildTargetGroup => PlayerSettings.use32BitDisplayBuffer = true,
            fixMessage: "PlayerSettings.use32BitDisplayBuffer = true"
        );

        //[Recommended] Set the Rendering Path to Forward
        // TODO : Support Scripted Rendering Pipeline?
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Recommended,
            group: targetGroup,
            isDone: buildTargetGroup =>
                EditorGraphicsSettings.GetTierSettings(buildTargetGroup, Graphics.activeTier).renderingPath == RenderingPath.Forward,
            message: "Use Forward Rendering Path",
            fix: buildTargetGroup =>
            {
                var renderingTier = EditorGraphicsSettings.GetTierSettings(buildTargetGroup, Graphics.activeTier);
                renderingTier.renderingPath =
                    RenderingPath.Forward;
                EditorGraphicsSettings.SetTierSettings(buildTargetGroup, Graphics.activeTier, renderingTier);
            },
            fixMessage: "renderingTier.renderingPath = RenderingPath.Forward"
            );

        //[Recommended] Set the Stereo Rendering to Instancing
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Recommended,
            group: targetGroup,
            isDone: buildTargetGroup =>
                PlayerSettings.stereoRenderingPath == StereoRenderingPath.Instancing,
            message: "Use Stereo Rendering Instancing",
            fix: buildTargetGroup => PlayerSettings.stereoRenderingPath = StereoRenderingPath.Instancing,
            fixMessage: "PlayerSettings.stereoRenderingPath = StereoRenderingPath.Instancing"
        );

        //[Optional] Use Non-Directional Lightmaps
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Optional,
            group: targetGroup,
            isDone: buildTargetGroup =>
            {
                return LightmapSettings.lightmaps.Length == 0 ||
                       LightmapSettings.lightmapsMode == LightmapsMode.NonDirectional;
            },
            message: "Use Non-Directional lightmaps",
            fix: buildTargetGroup => LightmapSettings.lightmapsMode = LightmapsMode.NonDirectional,
            fixMessage: "LightmapSettings.lightmapsMode = LightmapsMode.NonDirectional"
        );

        //[Optional] Disable Realtime GI
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Optional,
            group: targetGroup,
            isDone: buildTargetGroup => !Lightmapping.realtimeGI,
            message: "Disable Realtime Global Illumination",
            fix: buildTargetGroup => Lightmapping.realtimeGI = false,
            fixMessage: "Lightmapping.realtimeGI = false"
        );

        //[Optional] GPU Skinning
        OVRProjectSetup.AddTask(
	        level: OVRConfigurationTask.TaskLevel.Optional,
	        platform:BuildTargetGroup.Android,
	        group: targetGroup,
	        isDone: buildTargetGroup => PlayerSettings.gpuSkinning,
	        message: "Consider using GPU Skinning if your application is CPU bound",
	        fix: buildTargetGroup => PlayerSettings.gpuSkinning = true,
	        fixMessage: "PlayerSettings.gpuSkinning = true"
        );
    }
}
