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

[InitializeOnLoad]
internal static  class OVRProjectSetupPassthrough
{
	private const OVRConfigurationTask.TaskGroup Group = OVRConfigurationTask.TaskGroup.Features;

	static OVRProjectSetupPassthrough()
	{
		OVRProjectSetup.AddTask(
			level: OVRConfigurationTask.TaskLevel.Required,
			group: Group,
			isDone: buildTargetGroup => OVRProjectSetupUtils.FindComponentInScene<OVRPassthroughLayer>() == null || OVRProjectConfig.CachedProjectConfig.insightPassthroughSupport != OVRProjectConfig.FeatureSupport.None,
			message: "When using Passthrough in your project it's required to enable it's capability in the project config",
			fix: buildTargetGroup =>
			{
				var projectConfig = OVRProjectConfig.CachedProjectConfig;
				projectConfig.insightPassthroughSupport =
					OVRProjectConfig.FeatureSupport.Supported;
				OVRProjectConfig.CommitProjectConfig(projectConfig);
			},
			fixMessage:"Enable Passthrough support in the project config");

		OVRProjectSetup.AddTask(
			level: OVRConfigurationTask.TaskLevel.Required,
			group: Group,
			isDone: buildTargetGroup => OVRProjectSetupCompatibilityTasks.IsTargetingARM64,
			conditionalValidity: buildTargetGroup => OVRProjectConfig.CachedProjectConfig.insightPassthroughSupport != OVRProjectConfig.FeatureSupport.None,
			message: "When enabling the Passthrough capability in your project it's required to use ARM64 as the target architecture",
			fix: OVRProjectSetupCompatibilityTasks.SetARM64Target,
			fixMessage:"PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64"
		);
	}
}
