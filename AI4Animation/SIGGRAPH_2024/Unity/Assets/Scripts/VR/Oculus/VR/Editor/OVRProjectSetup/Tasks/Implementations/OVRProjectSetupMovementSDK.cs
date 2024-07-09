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
using UnityEngine;

[InitializeOnLoad]
internal static class OVRProjectSetupMovementSDK
{
	private const OVRConfigurationTask.TaskGroup Group = OVRConfigurationTask.TaskGroup.Features;

	static OVRProjectSetupMovementSDK()
	{
		AddMovementTrackingTasks<OVRBody>(
			"Body Tracking",
			() => OVRProjectConfig.CachedProjectConfig.bodyTrackingSupport,
			ovrManager => ovrManager.requestBodyTrackingPermissionOnStartup,
			projectConfig => projectConfig.bodyTrackingSupport = OVRProjectConfig.FeatureSupport.Supported,
			ovrManager => ovrManager.requestBodyTrackingPermissionOnStartup = true);

		AddMovementTrackingTasks<OVRFaceExpressions>(
			"Face Tracking",
			() => OVRProjectConfig.CachedProjectConfig.faceTrackingSupport,
			ovrManager => ovrManager.requestFaceTrackingPermissionOnStartup,
			projectConfig => projectConfig.faceTrackingSupport = OVRProjectConfig.FeatureSupport.Supported,
			ovrManager => ovrManager.requestFaceTrackingPermissionOnStartup = true);

		AddMovementTrackingTasks<OVREyeGaze>(
			"Eye Tracking",
			() => OVRProjectConfig.CachedProjectConfig.eyeTrackingSupport,
			ovrManager => ovrManager.requestEyeTrackingPermissionOnStartup,
			projectConfig => projectConfig.eyeTrackingSupport = OVRProjectConfig.FeatureSupport.Supported,
			ovrManager => ovrManager.requestEyeTrackingPermissionOnStartup = true);
	}

	private static void AddMovementTrackingTasks<T>(string featureName, Func<OVRProjectConfig.FeatureSupport> supportLevel, Func<OVRManager, bool> permissionRequested, Action<OVRProjectConfig> enableSupport, Action<OVRManager> enablePermissionRequest) where T : Component
	{
		OVRProjectSetup.AddTask(
			level: OVRConfigurationTask.TaskLevel.Required,
			group: Group,
			isDone: buildTargetGroup => OVRProjectSetupUtils.FindComponentInScene<T>() == null || supportLevel() != OVRProjectConfig.FeatureSupport.None,
			message: $"When using {featureName} in your project it's required to enable it's capability in the project config",
			fix: buildTargetGroup =>
			{
				var projectConfig = OVRProjectConfig.CachedProjectConfig;
				enableSupport(projectConfig);
				OVRProjectConfig.CommitProjectConfig(projectConfig);
			},
			fixMessage:$"Enable {featureName} support in the project config"
			);

		OVRProjectSetup.AddTask(
			level: OVRConfigurationTask.TaskLevel.Optional,
			group: Group,
			isDone: buildTargetGroup =>
			{
				if (supportLevel() == OVRProjectConfig.FeatureSupport.None)
				{
					return true;
				}

				var ovrManager = OVRProjectSetupUtils.FindComponentInScene<OVRManager>();
				return !ovrManager || permissionRequested(ovrManager);
			},
			message: $"Automatically request the {featureName} permission on startup",
			fix: buildTargetGroup =>
			{
				var ovrManager = OVRProjectSetupUtils.FindComponentInScene<OVRManager>();
				if (ovrManager != null)
				{
					enablePermissionRequest(ovrManager);
					EditorUtility.SetDirty(ovrManager);
				}
			},
			fixMessage:$"Request {featureName} permission on startup"
		);
	}
}
