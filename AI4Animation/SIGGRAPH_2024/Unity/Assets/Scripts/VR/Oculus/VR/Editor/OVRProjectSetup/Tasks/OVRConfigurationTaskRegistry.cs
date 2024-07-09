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
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

internal class OVRConfigurationTaskRegistry
{
    private static readonly List<OVRConfigurationTask> EmptyTasksList = new List<OVRConfigurationTask>(0);

    private readonly Dictionary<Hash128, OVRConfigurationTask> _tasksPerUid = new Dictionary<Hash128, OVRConfigurationTask>();
    private readonly List<OVRConfigurationTask> _tasks = new List<OVRConfigurationTask>();

    private List<OVRConfigurationTask> Tasks => _tasks;

    public void AddTask(OVRConfigurationTask task)
    {
        var uid = task.Uid;
        if (_tasksPerUid.ContainsKey(uid))
        {
            // This task is already registered
            throw new ArgumentException(
                $"[{nameof(OVRConfigurationTask)}] Task with same Uid already exists (hash collision)");
        }

        _tasks.Add(task);
        _tasksPerUid.Add(uid, task);
    }

    public void RemoveTask(Hash128 uid)
    {
        var task = GetTask(uid);
        RemoveTask(task);
    }

    public void RemoveTask(OVRConfigurationTask task)
    {
        _tasks.Remove(task);
        _tasksPerUid.Remove(task.Uid);
    }

    public OVRConfigurationTask GetTask(Hash128 uid)
    {
        _tasksPerUid.TryGetValue(uid, out var task);
        return task;
    }

    public void Clear()
    {
        _tasksPerUid.Clear();
        _tasks.Clear();
    }

    internal IEnumerable<OVRConfigurationTask> GetTasks(BuildTargetGroup buildTargetGroup, bool refresh)
    {
        if (refresh)
        {
	        foreach (var task in Tasks)
	        {
		        task.InvalidateCache(buildTargetGroup);
	        }
        }

        return Tasks.Where
        (
			task => (task.Platform == BuildTargetGroup.Unknown || task.Platform == buildTargetGroup)
					&& task.Valid.GetValue(buildTargetGroup)
        );
    }
}
