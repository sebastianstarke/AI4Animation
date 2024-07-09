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

namespace OculusSampleFramework
{
	public class PanelHMDFollower : MonoBehaviour
	{
		private const float TOTAL_DURATION = 3.0f;
		private const float HMD_MOVEMENT_THRESHOLD = 0.3f;

		[SerializeField] private float _maxDistance = 0.3f;
		[SerializeField] private float _minDistance = 0.05f;
		[SerializeField] private float _minZDistance = 0.05f;

		private OVRCameraRig _cameraRig;
		private Vector3 _panelInitialPosition = Vector3.zero;
		private Coroutine _coroutine = null;
		private Vector3 _prevPos = Vector3.zero;
		private Vector3 _lastMovedToPos = Vector3.zero;

		private void Awake()
		{
			_cameraRig = FindObjectOfType<OVRCameraRig>();
			_panelInitialPosition = transform.position;
		}

		private void Update()
		{
			var centerEyeAnchorPos = _cameraRig.centerEyeAnchor.position;
			var myPosition = transform.position;
			//Distance from centereye since last time we updated panel position.
			float distanceFromLastMovement = Vector3.Distance(centerEyeAnchorPos, _lastMovedToPos);
			float headMovementSpeed = (_cameraRig.centerEyeAnchor.position - _prevPos).magnitude / Time.deltaTime;
			var currDiffFromCenterEye = transform.position - centerEyeAnchorPos;
			var currDistanceFromCenterEye = currDiffFromCenterEye.magnitude;


			// 1) wait for center eye to stabilize after distance gets too large
			// 2) check if center eye is too close to panel
			// 3) check if depth isn't too close
			if (((distanceFromLastMovement > _maxDistance) || (_minZDistance > currDiffFromCenterEye.z) || (_minDistance > currDistanceFromCenterEye)) &&
				headMovementSpeed < HMD_MOVEMENT_THRESHOLD && _coroutine == null)
			{
				if (_coroutine == null)
				{
					_coroutine = StartCoroutine(LerpToHMD());
				}
			}

			_prevPos = _cameraRig.centerEyeAnchor.position;
		}

		private Vector3 CalculateIdealAnchorPosition()
		{
			return _cameraRig.centerEyeAnchor.position + _panelInitialPosition;
		}

		private IEnumerator LerpToHMD()
		{
			Vector3 newPanelPosition = CalculateIdealAnchorPosition();
			_lastMovedToPos = _cameraRig.centerEyeAnchor.position;
			float startTime = Time.time;
			float endTime = Time.time + TOTAL_DURATION;

			while (Time.time < endTime)
			{
				transform.position =
				  Vector3.Lerp(transform.position, newPanelPosition, (Time.time - startTime) / TOTAL_DURATION);
				yield return null;
			}

			transform.position = newPanelPosition;
			_coroutine = null;
		}
	}
}
