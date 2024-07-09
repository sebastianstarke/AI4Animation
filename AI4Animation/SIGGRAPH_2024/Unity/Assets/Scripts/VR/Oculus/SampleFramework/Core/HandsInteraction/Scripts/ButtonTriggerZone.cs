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


using UnityEngine;
using UnityEngine.Assertions;

namespace OculusSampleFramework
{
	/// <summary>
	/// Trigger zone of button, can be proximity, contact or action.
	/// </summary>
	public class ButtonTriggerZone : MonoBehaviour, ColliderZone
	{
		[SerializeField] private GameObject _parentInteractableObj = null;

		public Collider Collider { get; private set; }
		public Interactable ParentInteractable { get; private set; }

		public InteractableCollisionDepth CollisionDepth
		{
			get
			{
				var myColliderZone = (ColliderZone)this;
				var depth = ParentInteractable.ProximityCollider == myColliderZone ? InteractableCollisionDepth.Proximity :
				  ParentInteractable.ContactCollider == myColliderZone ? InteractableCollisionDepth.Contact :
				  ParentInteractable.ActionCollider == myColliderZone ? InteractableCollisionDepth.Action :
				  InteractableCollisionDepth.None;
				return depth;
			}
		}

		private void Awake()
		{
			Assert.IsNotNull(_parentInteractableObj);

			Collider = GetComponent<Collider>();
			ParentInteractable = _parentInteractableObj.GetComponent<Interactable>();
		}
	}
}
