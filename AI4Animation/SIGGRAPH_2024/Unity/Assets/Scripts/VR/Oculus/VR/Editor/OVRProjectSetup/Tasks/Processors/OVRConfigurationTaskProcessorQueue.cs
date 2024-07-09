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
using UnityEditor;

internal class OVRConfigurationTaskProcessorQueue
{
	public event Action<OVRConfigurationTaskProcessor> OnProcessorCompleted;

	private readonly Queue<OVRConfigurationTaskProcessor> _queue = new Queue<OVRConfigurationTaskProcessor>();

	public bool Busy => _queue.Count > 0;
	public bool Blocked => Busy && _queue.Peek().Blocking;
	public bool BlockedBy(OVRConfigurationTaskProcessor.ProcessorType processorType)
	{
		foreach (var processor in _queue)
		{
			if (processor.Type == processorType && processor.Blocking)
			{
				return true;
			}
		}

		return false;
	}
	public bool BusyWith(OVRConfigurationTaskProcessor.ProcessorType processorType)
	{
		foreach (var processor in _queue)
		{
			if (processor.Type == processorType)
			{
				return true;
			}
		}

		return false;
	}

	public void Request(OVRConfigurationTaskProcessor processor)
	{
		if (!OVRProjectSetup.Enabled.Value)
		{
			return;
		}

		Enqueue(processor);
	}

	private void Enqueue(OVRConfigurationTaskProcessor processor)
	{
		if (!Busy)
		{
			// If was empty, then register to editor update
			EditorApplication.update += Update;
		}

		// Enqueue
		_queue.Enqueue(processor);

		processor.OnRequested();

		if (processor.Blocking)
		{
			// In the case where the newly added processor is blocking
			// we'll make all the previously queued processor blocking as well
			foreach (var otherProcessor in _queue)
			{
				otherProcessor.Blocking = true;
			}

			// Force an update, this will be The blocking update
			Update();
		}
	}

	private void Dequeue(OVRConfigurationTaskProcessor processor)
	{
		// We should only dequeue the current processor
		if (processor != _queue.Peek())
		{
			return;
		}

		// Trigger specific callbacks
		processor.Complete();

		// Trigger global callbacks
		OnProcessorCompleted?.Invoke(processor);

		// Dequeue
		_queue.Dequeue();

		if (!Busy)
		{
			// Now that it is empty, unregister to editor update
			EditorApplication.update -= Update;
		}
	}

	private void Update()
	{
		do
		{
			// Grab the current processor
			var current = _queue.Peek();
			if (!current.Started)
			{
				// If not busy, this implies it hasn't been started yet
				// Start it
				current.Start();
			}

			current.Update();

			if (current.Completed)
			{
				// If it is completed, we can remove it from the queue
				Dequeue(current);
			}
			else
			{
				// If it is not completed, another update call will be necessary
				return;
			}

		} while (_queue.Count > 0 && (_queue.Peek()?.Blocking ?? false));
		// If the queue is blocking, do it until the queue is not empty
		// and the current processor is blocking
	}
}
