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

/// <summary>
/// Visual of finger tip poke tool.
/// </summary>
namespace OculusSampleFramework
{
	public class FingerTipPokeToolView : MonoBehaviour, InteractableToolView
	{
		[SerializeField] private MeshRenderer _sphereMeshRenderer = null;

		public InteractableTool InteractableTool { get; set; }

		public bool EnableState
		{
			get
			{
				return _sphereMeshRenderer.enabled;
			}
			set
			{
				_sphereMeshRenderer.enabled = value;
			}
		}

		public bool ToolActivateState { get; set; }

		public float SphereRadius { get; private set; }

		private void Awake()
		{
			Assert.IsNotNull(_sphereMeshRenderer);
			SphereRadius = _sphereMeshRenderer.transform.localScale.z * 0.5f;
		}

		public void SetFocusedInteractable(Interactable interactable)
		{
			// nothing to see here
		}
	}
}
