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
using UnityEngine;

namespace OculusSampleFramework
{
	/// <summary>
	/// In case someone wants to know about all interactables in a scene,
	/// this registry is the easiest way to access that information.
	/// </summary>
	public class InteractableRegistry : MonoBehaviour
	{
		public static HashSet<Interactable> _interactables = new HashSet<Interactable>();

		public static HashSet<Interactable> Interactables
		{
			get
			{
				return _interactables;
			}
		}

		public static void RegisterInteractable(Interactable interactable)
		{
			Interactables.Add(interactable);
		}

		public static void UnregisterInteractable(Interactable interactable)
		{
			Interactables.Remove(interactable);
		}
	}
}
