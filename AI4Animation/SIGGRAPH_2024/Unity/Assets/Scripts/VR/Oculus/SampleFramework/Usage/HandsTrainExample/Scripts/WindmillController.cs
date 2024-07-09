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
	public class WindmillController : MonoBehaviour
	{
		[SerializeField] private GameObject _startStopButton = null;
		[SerializeField] float _maxSpeed = 10f;
		[SerializeField] private SelectionCylinder _selectionCylinder = null;

		private WindmillBladesController _bladesRotation;
		private InteractableTool _toolInteractingWithMe = null;

		private void Awake()
		{
			Assert.IsNotNull(_startStopButton);
			Assert.IsNotNull(_selectionCylinder);

			_bladesRotation = GetComponentInChildren<WindmillBladesController>();

			_bladesRotation.SetMoveState(true, _maxSpeed);
		}

		private void OnEnable()
		{
			_startStopButton.GetComponent<Interactable>().InteractableStateChanged.AddListener(StartStopStateChanged);
		}

		private void OnDisable()
		{
			if (_startStopButton != null)
			{
				_startStopButton.GetComponent<Interactable>().InteractableStateChanged.RemoveListener(StartStopStateChanged);
			}
		}

		private void StartStopStateChanged(InteractableStateArgs obj)
		{
			bool inActionState = obj.NewInteractableState == InteractableState.ActionState;
			if (inActionState)
			{
				if (_bladesRotation.IsMoving)
				{
					_bladesRotation.SetMoveState(false, 0.0f);
				}
				else
				{
					_bladesRotation.SetMoveState(true, _maxSpeed);
				}
			}

			_toolInteractingWithMe = obj.NewInteractableState > InteractableState.Default ?
			  obj.Tool : null;
		}

		private void Update()
		{
			if (_toolInteractingWithMe == null)
			{
				_selectionCylinder.CurrSelectionState = SelectionCylinder.SelectionState.Off;
			}
			else
			{
				_selectionCylinder.CurrSelectionState = (
				  _toolInteractingWithMe.ToolInputState == ToolInputState.PrimaryInputDown ||
				  _toolInteractingWithMe.ToolInputState == ToolInputState.PrimaryInputDownStay)
				  ? SelectionCylinder.SelectionState.Highlighted
				  : SelectionCylinder.SelectionState.Selected;
			}
		}
	}
}
