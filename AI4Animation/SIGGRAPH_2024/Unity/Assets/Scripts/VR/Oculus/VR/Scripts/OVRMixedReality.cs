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

#if UNITY_ANDROID && !UNITY_EDITOR
#define OVR_ANDROID_MRC
#endif

using System;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || UNITY_ANDROID

/// <summary>
/// Manages mix-reality elements
/// </summary>
internal static class OVRMixedReality
{
	/// <summary>
	/// For Debugging purpose, we can use preset parameters to fake a camera when external camera is not available
	/// </summary>
	public static bool useFakeExternalCamera = false;
	public static Vector3 fakeCameraFloorLevelPosition = new Vector3(0.0f, 2.0f, -0.5f);
	public static Vector3 fakeCameraEyeLevelPosition = fakeCameraFloorLevelPosition - new Vector3(0.0f, 1.8f, 0.0f);
	public static Quaternion fakeCameraRotation = Quaternion.LookRotation((new Vector3(0.0f, fakeCameraFloorLevelPosition.y, 0.0f) - fakeCameraFloorLevelPosition).normalized, Vector3.up);
	public static float fakeCameraFov = 60.0f;
	public static float fakeCameraAspect = 16.0f / 9.0f;

	/// <summary>
	/// Composition object
	/// </summary>
	public static OVRComposition currentComposition = null;

	/// <summary>
	/// Updates the internal state of the Mixed Reality Camera. Called by OVRManager.
	/// </summary>

	public static void Update(GameObject parentObject, Camera mainCamera, OVRMixedRealityCaptureConfiguration configuration, OVRManager.TrackingOrigin trackingOrigin)
	{
		if (!OVRPlugin.initialized)
		{
			Debug.LogError("OVRPlugin not initialized");
			return;
		}

		if (!OVRPlugin.IsMixedRealityInitialized())
		{
			OVRPlugin.InitializeMixedReality();
			if (OVRPlugin.IsMixedRealityInitialized())
			{
				Debug.Log("OVRPlugin_MixedReality initialized");
			}
			else
			{
				Debug.LogError("Unable to initialize OVRPlugin_MixedReality");
				return;
			}
		}

		if (!OVRPlugin.IsMixedRealityInitialized())
		{
			return;
		}

		OVRPlugin.UpdateExternalCamera();
#if !OVR_ANDROID_MRC
		OVRPlugin.UpdateCameraDevices();
#endif

#if OVR_ANDROID_MRC
		useFakeExternalCamera = OVRPlugin.Media.UseMrcDebugCamera();
#endif

		if (currentComposition != null && (currentComposition.CompositionMethod() != configuration.compositionMethod))
		{
			currentComposition.Cleanup();
			currentComposition = null;
		}

		if (configuration.compositionMethod == OVRManager.CompositionMethod.External)
		{
			if (currentComposition == null)
			{
				currentComposition = new OVRExternalComposition(parentObject, mainCamera, configuration);
			}
		}
		else
		{
			Debug.LogError("Unknown/Unsupported CompositionMethod : " + configuration.compositionMethod);
			return;
		}
		currentComposition.Update(parentObject, mainCamera, configuration, trackingOrigin);
	}

	public static void Cleanup()
	{
		if (currentComposition != null)
		{
			currentComposition.Cleanup();
			currentComposition = null;
		}
		if (OVRPlugin.IsMixedRealityInitialized())
		{
			OVRPlugin.ShutdownMixedReality();
		}
	}

	public static void RecenterPose()
	{
		if (currentComposition != null)
		{
			currentComposition.RecenterPose();
		}
	}

}

#endif
