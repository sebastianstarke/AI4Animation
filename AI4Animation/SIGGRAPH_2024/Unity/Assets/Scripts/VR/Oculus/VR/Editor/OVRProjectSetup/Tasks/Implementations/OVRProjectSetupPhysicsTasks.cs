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
internal static class OVRProjectSetupPhysicsTasks
{
    static OVRProjectSetupPhysicsTasks()
    {
        // [Recommended] Default Contact Offset >= 0.01f
        OVRProjectSetup.AddTask(
            level: OVRConfigurationTask.TaskLevel.Recommended,
            group: OVRConfigurationTask.TaskGroup.Physics,
            isDone: group => Physics.defaultContactOffset >= 0.01f,
            message: $"Use Default Context Offset above or equal to 0.01",
            fix: group => Physics.defaultContactOffset = 0.01f,
            fixMessage: "Physics.defaultContactOffset = 0.01f"
        );

        // [Recommended] Sleep Threshold >= 0.005f
        OVRProjectSetup.AddTask(
	        level: OVRConfigurationTask.TaskLevel.Recommended,
	        group: OVRConfigurationTask.TaskGroup.Physics,
	        isDone: group => Physics.sleepThreshold >= 0.005f,
	        message: $"Use Sleep Threshold above or equal to 0.005",
	        fix: group => Physics.sleepThreshold = 0.005f,
	        fixMessage: "Physics.sleepThreshold = 0.005f"
        );

        // [Recommended] Default Solver Iterations <= 8
        OVRProjectSetup.AddTask(
	        level: OVRConfigurationTask.TaskLevel.Recommended,
	        group: OVRConfigurationTask.TaskGroup.Physics,
	        isDone: group => Physics.defaultSolverIterations <= 8,
	        message: $"Use Default Solver Iteration below or equal to 8",
	        fix: group => Physics.defaultSolverIterations = 8,
	        fixMessage: "Physics.defaultSolverIterations = 8"
        );
    }
}
