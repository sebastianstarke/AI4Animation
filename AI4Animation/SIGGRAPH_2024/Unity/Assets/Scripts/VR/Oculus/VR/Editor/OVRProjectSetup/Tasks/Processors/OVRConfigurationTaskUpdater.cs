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

internal class OVRConfigurationTaskUpdater : OVRConfigurationTaskProcessor
{
	public override int AllocatedTimeInMs => 10;
	private readonly OVRConfigurationTaskUpdaterSummary _summary;
	protected override Func<IEnumerable<OVRConfigurationTask>, List<OVRConfigurationTask>> OpenTasksFilter =>
		(Func<IEnumerable<OVRConfigurationTask>, List<OVRConfigurationTask>>)(tasksToFilter => tasksToFilter
		.Where(task => !task.IsIgnored(BuildTargetGroup))
		.ToList());

	public override ProcessorType Type => ProcessorType.Updater;
	public OVRConfigurationTaskUpdaterSummary Summary => _summary;

	public OVRConfigurationTaskUpdater(
		OVRConfigurationTaskRegistry registry,
		BuildTargetGroup buildTargetGroup,
		Func<IEnumerable<OVRConfigurationTask>, List<OVRConfigurationTask>> filter,
		OVRProjectSetup.LogMessages logMessages,
		bool blocking,
		Action<OVRConfigurationTaskProcessor> onCompleted)
		: base(registry, buildTargetGroup, filter, logMessages, blocking, onCompleted)
	{
		_summary = new OVRConfigurationTaskUpdaterSummary(BuildTargetGroup);
	}

	protected override void PrepareTasks()
	{
		_summary.Reset();
		base.PrepareTasks();
	}

	protected override void ProcessTask(OVRConfigurationTask task)
	{
		var changedState = task.UpdateAndGetStateChanged(BuildTargetGroup);

		if (task.IsDone(BuildTargetGroup))
		{
			return;
		}

		if (LogMessages == OVRProjectSetup.LogMessages.All
		    || (LogMessages == OVRProjectSetup.LogMessages.Changed && changedState))
		{
			task.LogMessage(BuildTargetGroup);
		}

		Summary.AddTask(task, changedState);
	}

	public override void Complete()
	{
		Summary.Validate();

		if (LogMessages >= OVRProjectSetup.LogMessages.Summary)
		{
			Summary.Log();
		}

		base.Complete();
	}
}
