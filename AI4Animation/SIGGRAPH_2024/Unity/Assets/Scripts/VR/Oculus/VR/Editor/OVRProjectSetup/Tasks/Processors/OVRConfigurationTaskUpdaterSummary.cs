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

using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

internal class OVRConfigurationTaskUpdaterSummary
{
    private readonly List<OVRConfigurationTask> _outstandingTasks;
    private readonly Dictionary<OVRConfigurationTask.TaskLevel, List<OVRConfigurationTask>> _outstandingTasksPerLevel;
    private bool HasChangedState { get; set; }

    public bool HasAvailableFixes => _outstandingTasks.Count > 0;
    public bool HasFixes(OVRConfigurationTask.TaskLevel taskLevel) => _outstandingTasksPerLevel[taskLevel].Count > 0;
    public int GetNumberOfFixes(OVRConfigurationTask.TaskLevel taskLevel) => _outstandingTasksPerLevel[taskLevel].Count;
    public int GetTotalNumberOfFixes() => _outstandingTasks.Count;
    private readonly BuildTargetGroup _buildTargetGroup;

    public BuildTargetGroup BuildTargetGroup => _buildTargetGroup;

    public OVRConfigurationTaskUpdaterSummary(BuildTargetGroup buildTargetGroup)
    {
        _buildTargetGroup = buildTargetGroup;
        _outstandingTasks = new List<OVRConfigurationTask>();
        _outstandingTasksPerLevel = new Dictionary<OVRConfigurationTask.TaskLevel, List<OVRConfigurationTask>>();
        for (var i = OVRConfigurationTask.TaskLevel.Required; i >= OVRConfigurationTask.TaskLevel.Optional; i--)
        {
            _outstandingTasksPerLevel.Add(i, new List<OVRConfigurationTask>());
        }
    }

    public void Reset()
    {
        _outstandingTasks.Clear();
        for (var i = OVRConfigurationTask.TaskLevel.Required; i >= OVRConfigurationTask.TaskLevel.Optional; i--)
        {
            _outstandingTasksPerLevel[i].Clear();
        }

        HasChangedState = false;
    }

    public void AddTask(OVRConfigurationTask task, bool changedState)
    {
        _outstandingTasks.Add(task);
        _outstandingTasksPerLevel[task.Level.GetValue(_buildTargetGroup)].Add(task);
        HasChangedState |= changedState;
    }

    public void Validate()
    {
    }

    public OVRConfigurationTask.TaskLevel? HighestFixLevel
    {
        get
        {
            for (var i = OVRConfigurationTask.TaskLevel.Required; i >= OVRConfigurationTask.TaskLevel.Optional; i--)
            {
                if (HasFixes(i))
                {
                    return i;
                }
            }

            return null;
        }
    }

    public string ComputeNoticeMessage()
    {
        var highestLevel = HighestFixLevel;
        var level = highestLevel ?? OVRConfigurationTask.TaskLevel.Optional;
        var count = GetNumberOfFixes(level);
        if (count == 0)
        {
	        return $"Oculus-Ready for {_buildTargetGroup}";
        }
        else
        {
	        var message = GetLogMessage(level, count);
	        return message;
        }
    }

    public string ComputeLogMessage()
    {
        var highestLevel = HighestFixLevel;
        var level = highestLevel ?? OVRConfigurationTask.TaskLevel.Optional;
        var count = GetNumberOfFixes(level);
        var message = GetFullLogMessage(level, count);
        return message;
    }

    public void Log()
    {
        if (!HasChangedState)
        {
            return;
        }

        var highestLevel = HighestFixLevel;
        var message = ComputeLogMessage();

        switch (highestLevel)
        {
            case OVRConfigurationTask.TaskLevel.Optional:
            {
                Debug.Log(message);
            }
                break;

            case OVRConfigurationTask.TaskLevel.Recommended:
            {
                Debug.LogWarning(message);
            }
                break;

            case OVRConfigurationTask.TaskLevel.Required:
            {
                if (OVRProjectSetup.RequiredThrowErrors.Value)
                {
                    Debug.LogError(message);
                }
                else
                {
                    Debug.LogWarning(message);
                }
            }
            break;
        }
    }

    private static string GetLogMessage(OVRConfigurationTask.TaskLevel level, int count)
    {
        switch (count)
        {
            case 0:
                return $"There are no outstanding {level.ToString()} fixes.";

            case 1:
                return $"There is 1 outstanding {level.ToString()} fix.";

            default:
                return $"There are {count} outstanding {level.ToString()} fixes.";
        }
    }

    internal static string GetFullLogMessage(OVRConfigurationTask.TaskLevel level, int count)
    {
        return
            $"[Oculus Settings] {GetLogMessage(level, count)}\nFor more information, go to <a href=\"{OVRConfigurationTask.ConsoleLinkHref}\">Edit > Project Settings > {OVRProjectSetupSettingsProvider.SettingsName}</a>";
    }
}
