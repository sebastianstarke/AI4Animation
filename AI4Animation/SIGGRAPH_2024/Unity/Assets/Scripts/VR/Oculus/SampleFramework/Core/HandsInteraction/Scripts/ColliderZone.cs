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
using UnityEngine;

namespace OculusSampleFramework
{
	/// <summary>
	/// Zone that can be collided with in example code.
	/// </summary>
	public interface ColliderZone
	{
		Collider Collider { get; }
		// Which interactable do we belong to?
		Interactable ParentInteractable { get; }
		InteractableCollisionDepth CollisionDepth { get; }
	}

	/// <summary>
	/// Arguments for object interacting with collider zone.
	/// </summary>
	public class ColliderZoneArgs : EventArgs
	{
		public readonly ColliderZone Collider;
		public readonly float FrameTime;
		public readonly InteractableTool CollidingTool;
		public readonly InteractionType InteractionT;

		public ColliderZoneArgs(ColliderZone collider, float frameTime,
		  InteractableTool collidingTool, InteractionType interactionType)
		{
			Collider = collider;
			FrameTime = frameTime;
			CollidingTool = collidingTool;
			InteractionT = interactionType;
		}
	}

	public enum InteractionType
	{
		Enter = 0,
		Stay,
		Exit
	}
}
