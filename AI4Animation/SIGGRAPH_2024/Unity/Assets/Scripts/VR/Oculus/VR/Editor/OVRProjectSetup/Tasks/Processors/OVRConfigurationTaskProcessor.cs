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

internal abstract class OVRConfigurationTaskProcessor
{
	public enum ProcessorType
	{
		Updater,
		Fixer
	}

	public virtual int AllocatedTimeInMs => 10;
	protected abstract void ProcessTask(OVRConfigurationTask task);
	public abstract ProcessorType Type { get; }

	private BuildTargetGroup _buildTargetGroup;
	private OVRProjectSetup.LogMessages _logMessages;
	private readonly OVRConfigurationTaskRegistry _registry;
	private Func<IEnumerable<OVRConfigurationTask>, List<OVRConfigurationTask>> _filter;
	protected abstract Func<IEnumerable<OVRConfigurationTask>, List<OVRConfigurationTask>> OpenTasksFilter { get; }
	private List<OVRConfigurationTask> _tasks;
	private IEnumerator<OVRConfigurationTask> _enumerator;
	private int _startTime;

	public bool Blocking { get; set; }
	public event Action<OVRConfigurationTaskProcessor> OnComplete;

	public BuildTargetGroup BuildTargetGroup => _buildTargetGroup;
	public OVRProjectSetup.LogMessages LogMessages => _logMessages;

	// Status
	public bool Started => _startTime != -1;
	public bool Processing => _enumerator != null;
	public bool Completed => Started && (_enumerator == null || _enumerator.Current == null);
	public List<OVRConfigurationTask> Tasks => _tasks;

	protected OVRConfigurationTaskProcessor(OVRConfigurationTaskRegistry registry, BuildTargetGroup buildTargetGroup,
		Func<IEnumerable<OVRConfigurationTask>, List<OVRConfigurationTask>> filter,
		OVRProjectSetup.LogMessages logMessages,
		bool blocking,
		Action<OVRConfigurationTaskProcessor> onCompleted)
	{
		_registry = registry;
		_enumerator = null;
		_buildTargetGroup = buildTargetGroup;
		_logMessages = logMessages;
		Blocking = blocking;
		_filter = filter;
		_startTime = -1;

		OnComplete += onCompleted;
	}

	protected virtual void PrepareTasks()
	{
		// Get all the tasks from the Setup Tool
		var tasks = _registry.GetTasks(_buildTargetGroup, true);

		// Apply the caller-provided filter
		_tasks = _filter != null ? _filter(tasks) : tasks.ToList();
		// When not forced, apply the OpenTaskFilter as well
		_tasks = (OpenTasksFilter != null) ? OpenTasksFilter(_tasks) : _tasks;

		// Prepare the Enumerator
		_enumerator = _tasks.GetEnumerator();
		_enumerator.MoveNext();
	}

	public virtual void OnRequested()
	{
	}

	public virtual void Start()
	{
		PrepareTasks();
		_startTime = Environment.TickCount;
	}

	public void Update()
	{
		var updateTime = Environment.TickCount;
		var currentTime = updateTime;
		while ((Blocking || (currentTime - updateTime < AllocatedTimeInMs)) && _enumerator?.Current != null)
		{
			ProcessTask(_enumerator.Current);
			currentTime = Environment.TickCount;
			_enumerator.MoveNext();
		}

		if (Completed)
		{
			Validate();
		}
	}

	protected virtual void Validate()
	{
	}

	public virtual void Complete()
	{
		_enumerator = null;
		OnComplete?.Invoke(this);
	}
}
