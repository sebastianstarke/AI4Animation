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

#if USING_XR_SDK_OPENXR

using System;
using System.Runtime.InteropServices;
using AOT;
using UnityEngine;
using UnityEngine.XR.OpenXR;
using UnityEngine.XR.OpenXR.Features;

#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.XR.OpenXR;
using UnityEditor.XR.OpenXR.Features;
#endif

namespace Meta.XR
{
#if UNITY_EDITOR
	public class MetaXRFeatureEditorConfig
	{
		public const string OpenXrExtensionList =
			"XR_KHR_vulkan_enable " +
			"XR_KHR_D3D11_enable " +
			"XR_OCULUS_common_reference_spaces " +
			"XR_FB_display_refresh_rate " +
			"XR_EXT_performance_settings " +
			"XR_FB_composition_layer_image_layout " +
			"XR_KHR_android_surface_swapchain " +
			"XR_FB_android_surface_swapchain_create " +
			"XR_KHR_composition_layer_color_scale_bias " +
			"XR_FB_color_space " +
			"XR_EXT_hand_tracking " +
			"XR_FB_swapchain_update_state " +
			"XR_FB_swapchain_update_state_opengl_es " +
			"XR_FB_swapchain_update_state_vulkan " +
			"XR_FB_foveation " +
			"XR_FB_foveation_configuration " +
			"XR_FB_foveation_vulkan " +
			"XR_FB_composition_layer_alpha_blend " +
			"XR_KHR_composition_layer_depth " +
			"XR_KHR_composition_layer_cylinder " +
			"XR_KHR_composition_layer_cube " +
			"XR_KHR_composition_layer_equirect2 " +
			"XR_KHR_convert_timespec_time " +
			"XR_KHR_visibility_mask " +
			"XR_FB_render_model " +
			"XR_FB_spatial_entity " +
			"XR_FB_spatial_entity_query " +
			"XR_FB_spatial_entity_storage " +
			"XR_META_performance_metrics " +
			"XR_FB_scene " +
			"XR_FB_spatial_entity_container " +
			"XR_FB_scene_capture " +
			"XR_FB_face_tracking " +
			"XR_FB_eye_tracking " +
			"XR_FB_keyboard_tracking " +
			"XR_FB_passthrough " +
			"XR_FB_triangle_mesh " +
			"XR_FB_passthrough_keyboard_hands " +
			"XR_OCULUS_audio_device_guid " +
			"XR_FB_common_events " +
			"XR_FB_space_warp " +
			"XR_FB_hand_tracking_capsules " +
			"XR_FB_hand_tracking_mesh " +
			"XR_FB_hand_tracking_aim " +
			"XR_FB_touch_controller_pro " +
			"XR_FB_touch_controller_proximity " +
			""
			;
	}
#endif

	/// <summary>
	/// MetaXR Feature for OpenXR
	/// </summary>
#if UNITY_EDITOR
	[OpenXRFeature(UiName = "MetaXR Feature",
		BuildTargetGroups = new[] { BuildTargetGroup.Standalone, BuildTargetGroup.Android },
		Company = "Meta",
		Desc = "MetaXR Feature for OpenXR.",
		DocumentationLink = "https://developer.oculus.com/",
		OpenxrExtensionStrings = MetaXRFeatureEditorConfig.OpenXrExtensionList,
		Version = "0.0.1",
		FeatureId = featureId)]
#endif
	public class MetaXRFeature : OpenXRFeature
	{
		/// <summary>
		/// The feature id string. This is used to give the feature a well known id for reference.
		/// </summary>
		public const string featureId = "com.meta.openxr.feature.metaxr";

		/// <inheritdoc />
		protected override IntPtr HookGetInstanceProcAddr(IntPtr func)
		{
			OVRPlugin.UnityOpenXR.Enabled = true;

			Debug.Log($"[MetaXRFeature] HookGetInstanceProcAddr: {func}");

			Debug.Log($"[MetaXRFeature] SetClientVersion");
			OVRPlugin.UnityOpenXR.SetClientVersion();

			return OVRPlugin.UnityOpenXR.HookGetInstanceProcAddr(func);
		}

		/// <inheritdoc />
		protected override bool OnInstanceCreate(ulong xrInstance)
		{
			// here's one way you can grab the instance
			Debug.Log($"[MetaXRFeature] OnInstanceCreate: {xrInstance}");
			return OVRPlugin.UnityOpenXR.OnInstanceCreate(xrInstance);
		}

		/// <inheritdoc />
		protected override void OnInstanceDestroy(ulong xrInstance)
		{
			// here's one way you can grab the instance
			Debug.Log($"[MetaXRFeature] OnInstanceDestroy: {xrInstance}");
			OVRPlugin.UnityOpenXR.OnInstanceDestroy(xrInstance);
		}

		/// <inheritdoc />
		protected override void OnSessionCreate(ulong xrSession)
		{
			// here's one way you can grab the session
			Debug.Log($"[MetaXRFeature] OnSessionCreate: {xrSession}");
			OVRPlugin.UnityOpenXR.OnSessionCreate(xrSession);
		}

		/// <inheritdoc />
		protected override void OnAppSpaceChange(ulong xrSpace)
		{
			Debug.Log($"[MetaXRFeature] OnAppSpaceChange: {xrSpace}");
			OVRPlugin.UnityOpenXR.OnAppSpaceChange(xrSpace);
		}

		/// <inheritdoc />
		protected override void OnSessionStateChange(int oldState, int newState)
		{
			Debug.Log($"[MetaXRFeature] OnSessionStateChange: {oldState} -> {newState}");
			OVRPlugin.UnityOpenXR.OnSessionStateChange(oldState, newState);
		}

		/// <inheritdoc />
		protected override void OnSessionBegin(ulong xrSession)
		{
			Debug.Log($"[MetaXRFeature] OnSessionBegin: {xrSession}");
			OVRPlugin.UnityOpenXR.OnSessionBegin(xrSession);
		}

		/// <inheritdoc />
		protected override void OnSessionEnd(ulong xrSession)
		{
			Debug.Log($"[MetaXRFeature] OnSessionEnd: {xrSession}");
			OVRPlugin.UnityOpenXR.OnSessionEnd(xrSession);
		}

		/// <inheritdoc />
		protected override void OnSessionExiting(ulong xrSession)
		{
			Debug.Log($"[MetaXRFeature] OnSessionExiting: {xrSession}");
			OVRPlugin.UnityOpenXR.OnSessionExiting(xrSession);
		}

		/// <inheritdoc />
		protected override void OnSessionDestroy(ulong xrSession)
		{
			Debug.Log($"[MetaXRFeature] OnSessionDestroy: {xrSession}");
			OVRPlugin.UnityOpenXR.OnSessionDestroy(xrSession);
		}

		// protected override void OnSessionLossPending(ulong xrSession) {}
		// protected override void OnInstanceLossPending (ulong xrInstance) {}
		// protected override void OnSystemChange(ulong xrSystem) {}
		// protected override void OnFormFactorChange (int xrFormFactor) {}
		// protected override void OnViewConfigurationTypeChange (int xrViewConfigurationType) {}
		// protected override void OnEnvironmentBlendModeChange (int xrEnvironmentBlendMode) {}
		// protected override void OnEnabledChange() {}
	}
}

#endif
