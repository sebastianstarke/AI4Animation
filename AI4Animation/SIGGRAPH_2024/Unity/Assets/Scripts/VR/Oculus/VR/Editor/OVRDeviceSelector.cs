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

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
#if PRIORITIZE_OCULUS_XR_SETTINGS
using Unity.XR.Oculus;
#endif

public class OVRDeviceSelector
{
	public static bool isTargetDeviceQuestFamily
	{
		get
		{
			return isTargetDeviceQuest || isTargetDeviceQuest2 || isTargetDeviceQuestPro;
		}
	}
	public static bool isTargetDeviceQuest
	{
		get
		{
#if PRIORITIZE_OCULUS_XR_SETTINGS
			OculusSettings settings;
			UnityEditor.EditorBuildSettings.TryGetConfigObject<OculusSettings>("Unity.XR.Oculus.Settings", out settings);
			return settings.TargetQuest3;
#else
			OVRProjectConfig projectConfig = OVRProjectConfig.GetProjectConfig();
			return projectConfig.targetDeviceTypes.Contains(OVRProjectConfig.DeviceType.Quest);
#endif
		}
	}

	public static bool isTargetDeviceQuest2
	{
		get
		{
#if PRIORITIZE_OCULUS_XR_SETTINGS
			OculusSettings settings;
			UnityEditor.EditorBuildSettings.TryGetConfigObject<OculusSettings>("Unity.XR.Oculus.Settings", out settings);
			return settings.TargetQuest2;
#else
			OVRProjectConfig projectConfig = OVRProjectConfig.GetProjectConfig();
			return projectConfig.targetDeviceTypes.Contains(OVRProjectConfig.DeviceType.Quest2);
#endif
		}
	}

	public static bool isTargetDeviceQuestPro
	{
		get
		{
#if PRIORITIZE_OCULUS_XR_SETTINGS
#if USING_QUEST_PRO_COMPATIBLE_OCULUS_XR_PLUGIN_VERSION
			OculusSettings settings;
			UnityEditor.EditorBuildSettings.TryGetConfigObject<OculusSettings>("Unity.XR.Oculus.Settings", out settings);
			return settings.TargetQuestPro;
#else
			return false;
#endif
#else
			OVRProjectConfig projectConfig = OVRProjectConfig.GetProjectConfig();
			return projectConfig.targetDeviceTypes.Contains(OVRProjectConfig.DeviceType.QuestPro);
#endif
		}
	}
}
