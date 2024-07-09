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
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class HandsActiveChecker : MonoBehaviour
{
	[SerializeField]
	private GameObject _notificationPrefab = null;

	private GameObject _notification = null;
	private OVRCameraRig _cameraRig = null;
	private Transform _centerEye = null;

	private void Awake()
	{
		Assert.IsNotNull(_notificationPrefab);
		_notification = Instantiate(_notificationPrefab);
		StartCoroutine(GetCenterEye());
	}

	private void Update()
	{
		if (OVRPlugin.GetHandTrackingEnabled())
		{
			_notification.SetActive(false);
		}
		else
		{
			_notification.SetActive(true);
			if (_centerEye) {
				_notification.transform.position = _centerEye.position + _centerEye.forward * 0.5f;
				_notification.transform.rotation = _centerEye.rotation;
			}
			
		}

	}

	private IEnumerator GetCenterEye()
	{
		if ((_cameraRig = FindObjectOfType<OVRCameraRig>()) != null)
		{
			while (!_centerEye)
			{
				_centerEye = _cameraRig.centerEyeAnchor;
				yield return null;
			}
		}
	}
}
