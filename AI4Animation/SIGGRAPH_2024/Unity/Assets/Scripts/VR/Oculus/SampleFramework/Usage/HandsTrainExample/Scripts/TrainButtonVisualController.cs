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


using System.Collections;
using UnityEngine;
using UnityEngine.Assertions;

namespace OculusSampleFramework
{
	/// <summary>
	/// An example visual controller for a button intended for the train sample scene.
	/// </summary>
	public class TrainButtonVisualController : MonoBehaviour
	{
		private const float LERP_TO_OLD_POS_DURATION = 1.0f;
		private const float LOCAL_SIZE_HALVED = 0.5f;

		[SerializeField] private MeshRenderer _meshRenderer = null;
		[SerializeField] private MeshRenderer _glowRenderer = null;
		[SerializeField] private ButtonController _buttonController = null;
		[SerializeField] private Color _buttonContactColor = new Color(0.51f, 0.78f, 0.92f, 1.0f);
		[SerializeField] private Color _buttonActionColor = new Color(0.24f, 0.72f, 0.98f, 1.0f);

		[SerializeField] private AudioSource _audioSource = null;
		[SerializeField] private AudioClip _actionSoundEffect = null;

		[SerializeField] private Transform _buttonContactTransform = null;
		[SerializeField] private float _contactMaxDisplacementDistance = 0.0141f;

		private Material _buttonMaterial;
		private Color _buttonDefaultColor;
		private int _materialColorId;

		private bool _buttonInContactOrActionStates = false;

		private Coroutine _lerpToOldPositionCr = null;
		private Vector3 _oldPosition;

		private void Awake()
		{
			Assert.IsNotNull(_meshRenderer);
			Assert.IsNotNull(_glowRenderer);
			Assert.IsNotNull(_buttonController);
			Assert.IsNotNull(_audioSource);
			Assert.IsNotNull(_actionSoundEffect);

			Assert.IsNotNull(_buttonContactTransform);
			_materialColorId = Shader.PropertyToID("_Color");
			_buttonMaterial = _meshRenderer.material;
			_buttonDefaultColor = _buttonMaterial.GetColor(_materialColorId);

			_oldPosition = transform.localPosition;
		}

		private void OnDestroy()
		{
			if (_buttonMaterial != null)
			{
				Destroy(_buttonMaterial);
			}
		}

		private void OnEnable()
		{
			_buttonController.InteractableStateChanged.AddListener(InteractableStateChanged);
			_buttonController.ContactZoneEvent += ActionOrInContactZoneStayEvent;
			_buttonController.ActionZoneEvent += ActionOrInContactZoneStayEvent;
			_buttonInContactOrActionStates = false;
		}

		private void OnDisable()
		{
			if (_buttonController != null)
			{
				_buttonController.InteractableStateChanged.RemoveListener(InteractableStateChanged);
				_buttonController.ContactZoneEvent -= ActionOrInContactZoneStayEvent;
				_buttonController.ActionZoneEvent -= ActionOrInContactZoneStayEvent;
			}
		}

		private void ActionOrInContactZoneStayEvent(ColliderZoneArgs collisionArgs)
		{
			if (!_buttonInContactOrActionStates || collisionArgs.CollidingTool.IsFarFieldTool)
			{
				return;
			}

			// calculate how much the button should be pushed inwards. based on contact zone.
			// assume collider is uniform 1x1x1 cube, and all scaling, etc is done on transform component
			// another way to test distances is to measure distance to plane that represents where
			// button translation must stop
			Vector3 buttonScale = _buttonContactTransform.localScale;
			Vector3 interactionPosition = collisionArgs.CollidingTool.InteractionPosition;
			Vector3 localSpacePosition = _buttonContactTransform.InverseTransformPoint(
			  interactionPosition);
			// calculate offset in local space. so bias coordinates from 0.5,-0.5 to 0, -1.
			// 0 is no offset, 1.0 in local space is max offset pushing inwards
			Vector3 offsetVector = localSpacePosition - LOCAL_SIZE_HALVED * Vector3.one;
			// affect offset by button scale. only care about y (since y goes inwards)
			float scaledLocalSpaceOffset = offsetVector.y * buttonScale.y;

			// restrict button movement. can only go so far in negative direction, and cannot
			// be positive (which would cause the button to "stick out")
			if (scaledLocalSpaceOffset > -_contactMaxDisplacementDistance && scaledLocalSpaceOffset
				<= 0.0f)
			{
				transform.localPosition = new Vector3(_oldPosition.x, _oldPosition.y +
				  scaledLocalSpaceOffset, _oldPosition.z);
			}
		}

		private void InteractableStateChanged(InteractableStateArgs obj)
		{
			_buttonInContactOrActionStates = false;
			_glowRenderer.gameObject.SetActive(obj.NewInteractableState >
			  InteractableState.Default);
			switch (obj.NewInteractableState)
			{
				case InteractableState.ContactState:
					StopResetLerping();
					_buttonMaterial.SetColor(_materialColorId, _buttonContactColor);
					_buttonInContactOrActionStates = true;
					break;
				case InteractableState.ProximityState:
					_buttonMaterial.SetColor(_materialColorId, _buttonDefaultColor);
					LerpToOldPosition();
					break;
				case InteractableState.ActionState:
					StopResetLerping();
					_buttonMaterial.SetColor(_materialColorId, _buttonActionColor);
					PlaySound(_actionSoundEffect);
					_buttonInContactOrActionStates = true;
					break;
				default:
					_buttonMaterial.SetColor(_materialColorId, _buttonDefaultColor);
					LerpToOldPosition();
					break;
			}
		}

		private void PlaySound(AudioClip clip)
		{
			_audioSource.timeSamples = 0;
			_audioSource.clip = clip;
			_audioSource.Play();
		}

		private void StopResetLerping()
		{
			if (_lerpToOldPositionCr != null)
			{
				StopCoroutine(_lerpToOldPositionCr);
			}
		}

		private void LerpToOldPosition()
		{
			if ((transform.localPosition - _oldPosition).sqrMagnitude < Mathf.Epsilon)
			{
				return;
			}

			StopResetLerping();
			_lerpToOldPositionCr = StartCoroutine(ResetPosition());
		}

		private IEnumerator ResetPosition()
		{
			var startTime = Time.time;
			var endTime = Time.time + LERP_TO_OLD_POS_DURATION;

			while (Time.time < endTime)
			{
				transform.localPosition = Vector3.Lerp(transform.localPosition, _oldPosition,
				  (Time.time - startTime) / LERP_TO_OLD_POS_DURATION);
				yield return null;
			}

			transform.localPosition = _oldPosition;
			_lerpToOldPositionCr = null;
		}
	}
}
