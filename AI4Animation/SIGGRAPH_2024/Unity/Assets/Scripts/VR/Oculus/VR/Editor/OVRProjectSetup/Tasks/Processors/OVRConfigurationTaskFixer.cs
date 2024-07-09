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
using System;
using System.Linq;
using UnityEngine;

internal class OVRConfigurationTaskFixer : OVRConfigurationTaskProcessor
{
	public override int AllocatedTimeInMs => 10;
	public override ProcessorType Type => ProcessorType.Fixer;
	protected override Func<IEnumerable<OVRConfigurationTask>, List<OVRConfigurationTask>> OpenTasksFilter =>
		(Func<IEnumerable<OVRConfigurationTask>, List<OVRConfigurationTask>>)(tasksToFilter => tasksToFilter
		.Where(task => task.FixAction != null
		               && !task.IsDone(BuildTargetGroup)
		               && !task.IsIgnored(BuildTargetGroup))
		.ToList());

	private const int LoopExitCount = 4;

	private bool _hasFixedSome = false;
	private int _counter = LoopExitCount;

	public OVRConfigurationTaskFixer(
		OVRConfigurationTaskRegistry registry,
		BuildTargetGroup buildTargetGroup,
		Func<IEnumerable<OVRConfigurationTask>, List<OVRConfigurationTask>> filter,
		OVRProjectSetup.LogMessages logMessages,
		bool blocking,
		Action<OVRConfigurationTaskProcessor> onCompleted)
		: base(registry, buildTargetGroup, filter, logMessages, blocking, onCompleted)
	{
	}

	protected override void ProcessTask(OVRConfigurationTask task)
	{
		_hasFixedSome |= task.Fix(BuildTargetGroup);
	}

	protected override void PrepareTasks()
	{
		_hasFixedSome = false;
		base.PrepareTasks();
	}

	protected override void Validate()
	{
		_counter--;

		if (_counter <= 0)
		{
			Debug.LogWarning("[Oculus Settings] Fixing Tasks has exited after too many iterations. (There might be some contradictory rules leading to a loop)");
			return;
		}

		if (!_hasFixedSome)
		{
			return;
		}

		// Preparing a new Run
		PrepareTasks();
		if (Blocking)
		{
			Update();
		}
	}
}
