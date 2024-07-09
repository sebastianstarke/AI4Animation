// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using UnityEngine;

namespace OculusSampleFramework
{
	/// <summary>
	/// Manages pinch state, including if an object is being focused via something
	/// like a ray (or not).
	/// </summary>
	public class PinchStateModule
	{
		private const float PINCH_STRENGTH_THRESHOLD = 1.0f;

		private enum PinchState
		{
			None = 0,
			PinchDown,
			PinchStay,
			PinchUp
		}

		private PinchState _currPinchState;
		private Interactable _firstFocusedInteractable;

		/// <summary>
		/// We want a pinch up and down gesture to be done **while** an object is focused.
		/// We don't want someone to pinch, unfocus an object, then refocus before doing
		/// pinch up. We also want to avoid focusing a different interactable during this process.
		/// While the latter is difficult to do since a person might focus nothing before
		/// focusing on another interactable, it's theoretically possible.
		/// </summary>
		public bool PinchUpAndDownOnFocusedObject
		{
			get
			{
				return _currPinchState == PinchState.PinchUp && _firstFocusedInteractable != null;
			}
		}

		public bool PinchSteadyOnFocusedObject
		{
			get
			{
				return _currPinchState == PinchState.PinchStay && _firstFocusedInteractable != null;
			}
		}

		public bool PinchDownOnFocusedObject
		{
			get
			{
				return _currPinchState == PinchState.PinchDown && _firstFocusedInteractable != null;
			}
		}

		public PinchStateModule()
		{
			_currPinchState = PinchState.None;
			_firstFocusedInteractable = null;
		}

		public void UpdateState(OVRHand hand, Interactable currFocusedInteractable)
		{
			float pinchStrength = hand.GetFingerPinchStrength(OVRHand.HandFinger.Index);
			bool isPinching = Mathf.Abs(PINCH_STRENGTH_THRESHOLD - pinchStrength) < Mathf.Epsilon;
			var oldPinchState = _currPinchState;

			switch (oldPinchState)
			{
				case PinchState.PinchUp:
					// can only be in pinch up for a single frame, so consider
					// next frame carefully
					if (isPinching)
					{
						_currPinchState = PinchState.PinchDown;
						if (currFocusedInteractable != _firstFocusedInteractable)
						{
							_firstFocusedInteractable = null;
						}
					}
					else
					{
						_currPinchState = PinchState.None;
						_firstFocusedInteractable = null;
					}
					break;
				case PinchState.PinchStay:
					if (!isPinching)
					{
						_currPinchState = PinchState.PinchUp;
					}
					// if object is not focused anymore, then forget it
					if (currFocusedInteractable != _firstFocusedInteractable)
					{
						_firstFocusedInteractable = null;
					}
					break;
				// pinch down lasts for a max of 1 frame. either go to pinch stay or up
				case PinchState.PinchDown:
					_currPinchState = isPinching ? PinchState.PinchStay : PinchState.PinchUp;
					// if the focused interactable changes, then the original one is now invalid
					if (_firstFocusedInteractable != currFocusedInteractable)
					{
						_firstFocusedInteractable = null;
					}
					break;
				default:
					if (isPinching)
					{
						_currPinchState = PinchState.PinchDown;
						// this is the interactable that must be focused through out the pinch up and down
						// gesture.
						_firstFocusedInteractable = currFocusedInteractable;
					}
					break;
			}
		}
	}
}
