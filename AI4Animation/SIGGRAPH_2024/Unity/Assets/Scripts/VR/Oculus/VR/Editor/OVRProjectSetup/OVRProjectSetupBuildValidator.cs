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
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

internal class OVRProjectSetupBuildValidator : IPreprocessBuildWithReport
{
    public int callbackOrder => 0;
    public void OnPreprocessBuild(BuildReport report)
    {
        PreprocessBuild(report.summary.platformGroup);
    }

    public static void PreprocessBuild(BuildTargetGroup buildTargetGroup)
    {
        if (!OVRProjectSetup.IsPlatformSupported(buildTargetGroup))
        {
            return;
        }

        OVRProjectSetup.UpdateTasks(buildTargetGroup);

        foreach (var task in OVRProjectSetup.GetTasks(buildTargetGroup, false))
        {
            ValidateTask(task, buildTargetGroup);
        }
    }

    private static void ValidateTask(OVRConfigurationTask task, BuildTargetGroup buildTargetGroup)
    {
        if (task.IsIgnored(buildTargetGroup)
            || task.Level.GetValue(buildTargetGroup) != OVRConfigurationTask.TaskLevel.Required
            || task.IsDone(buildTargetGroup))
        {
            return;
        }

        if (OVRProjectSetup.RequiredThrowErrors.Value)
        {
            throw new BuildFailedException(task.GetFullLogMessage(buildTargetGroup));
        }
        else
        {
            Debug.LogWarning(task.GetFullLogMessage(buildTargetGroup));
        }
    }
}
