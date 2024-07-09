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
	public class WindmillBladesController : MonoBehaviour
	{
		private const float MAX_TIME = 1f;

		[SerializeField] private AudioSource _audioSource = null;
		[SerializeField] private AudioClip _windMillRotationSound = null;
		[SerializeField] private AudioClip _windMillStartSound = null;
		[SerializeField] private AudioClip _windMillStopSound = null;

		public bool IsMoving { get; private set; }

		private float _currentSpeed = 0f;
		private Coroutine _lerpSpeedCoroutine;
		private Coroutine _audioChangeCr;
		private Quaternion _originalRotation;
		private float _rotAngle = 0.0f;

		private void Start()
		{
			Assert.IsNotNull(_audioSource);
			Assert.IsNotNull(_windMillRotationSound);
			Assert.IsNotNull(_windMillStartSound);
			Assert.IsNotNull(_windMillStopSound);

			_originalRotation = transform.localRotation;
		}

		private void Update()
		{
			_rotAngle += _currentSpeed * Time.deltaTime;
			if (_rotAngle > 360.0f)
			{
				_rotAngle = 0.0f;
			}

			transform.localRotation = _originalRotation * Quaternion.AngleAxis(_rotAngle, Vector3.forward);
		}

		public void SetMoveState(bool newMoveState, float goalSpeed)
		{
			IsMoving = newMoveState;
			if (_lerpSpeedCoroutine != null)
			{
				StopCoroutine(_lerpSpeedCoroutine);
			}
			_lerpSpeedCoroutine = StartCoroutine(LerpToSpeed(goalSpeed));
		}

		private IEnumerator LerpToSpeed(float goalSpeed)
		{
			var totalTime = 0f;
			var startSpeed = _currentSpeed;

			if (_audioChangeCr != null)
			{
				StopCoroutine(_audioChangeCr);
			}

			// start up
			if (IsMoving)
			{
				_audioChangeCr = StartCoroutine(PlaySoundDelayed(_windMillStartSound,
				  _windMillRotationSound, _windMillStartSound.length * 0.95f));
			} // stop
			else
			{
				PlaySound(_windMillStopSound);
			}

			var diffSpeeds = Mathf.Abs(_currentSpeed - goalSpeed);
			while (diffSpeeds > Mathf.Epsilon)
			{
				_currentSpeed = Mathf.Lerp(startSpeed, goalSpeed, totalTime / MAX_TIME);
				totalTime += Time.deltaTime;
				yield return null;
				diffSpeeds = Mathf.Abs(_currentSpeed - goalSpeed);
			}

			_lerpSpeedCoroutine = null;
		}

		private IEnumerator PlaySoundDelayed(AudioClip initial, AudioClip clip, float timeDelayAfterInitial)
		{
			PlaySound(initial, false);
			yield return new WaitForSeconds(timeDelayAfterInitial);
			PlaySound(clip, true);
		}

		private void PlaySound(AudioClip clip, bool loop = false)
		{
			_audioSource.loop = loop;
			_audioSource.timeSamples = 0;
			_audioSource.clip = clip;
			_audioSource.Play();
		}
	}
}
