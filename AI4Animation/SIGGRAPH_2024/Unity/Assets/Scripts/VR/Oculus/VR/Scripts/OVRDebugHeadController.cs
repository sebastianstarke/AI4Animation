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

#if USING_XR_MANAGEMENT && (USING_XR_SDK_OCULUS || USING_XR_SDK_OPENXR)
#define USING_XR_SDK
#endif

#if UNITY_2020_1_OR_NEWER
#define REQUIRES_XR_SDK
#endif

using UnityEngine;
using System.Collections;

#if USING_XR_SDK
using UnityEngine.XR;
using UnityEngine.Experimental.XR;
#endif

/// <summary>
/// This is a simple behavior that can be attached to a parent of the CameraRig in order
/// to provide movement via the gamepad. This is useful when testing an application in
/// the Unity editor without the HMD.
/// To use it, create a game object in your scene and drag your CameraRig to be a child
/// of the game object. Then, add the OVRDebugHeadController behavior to the game object.
/// Alternatively, this behavior can be placed directly on the OVRCameraRig object, but
/// that is not guaranteed to work if OVRCameraRig functionality changes in the future.
/// In the parent case, the object with OVRDebugHeadController can be thougt of as a
/// platform that your camera is attached to. When the platform moves or rotates, the
/// camera moves or rotates, but the camera can still move independently while "on" the
/// platform.
/// In general, this behavior should be disabled when not debugging.
/// </summary>
public class OVRDebugHeadController : MonoBehaviour
{
	[SerializeField]
	public bool AllowPitchLook = false;
	[SerializeField]
	public bool AllowYawLook = true;
	[SerializeField]
	public bool InvertPitch = false;
	[SerializeField]
	public float GamePad_PitchDegreesPerSec = 90.0f;
	[SerializeField]
	public float GamePad_YawDegreesPerSec = 90.0f;
	[SerializeField]
	public bool AllowMovement = false;
	[SerializeField]
	public float ForwardSpeed = 2.0f;
	[SerializeField]
	public float StrafeSpeed = 2.0f;

	protected OVRCameraRig CameraRig = null;

	void Awake()
	{
		// locate the camera rig so we can use it to get the current camera transform each frame
		OVRCameraRig[] CameraRigs = gameObject.GetComponentsInChildren<OVRCameraRig>();

		if( CameraRigs.Length == 0 )
			Debug.LogWarning("OVRCamParent: No OVRCameraRig attached.");
		else if (CameraRigs.Length > 1)
			Debug.LogWarning("OVRCamParent: More then 1 OVRCameraRig attached.");
		else
			CameraRig = CameraRigs[0];
	}

	// Use this for initialization
	void Start ()
	{

	}

	// Update is called once per frame
	void Update ()
	{
		if ( AllowMovement )
		{
			float gamePad_FwdAxis = OVRInput.Get(OVRInput.RawAxis2D.LThumbstick).y;
			float gamePad_StrafeAxis = OVRInput.Get(OVRInput.RawAxis2D.LThumbstick).x;

			Vector3 fwdMove = ( CameraRig.centerEyeAnchor.rotation * Vector3.forward ) * gamePad_FwdAxis * Time.deltaTime * ForwardSpeed;
			Vector3 strafeMove = ( CameraRig.centerEyeAnchor.rotation * Vector3.right ) * gamePad_StrafeAxis * Time.deltaTime * StrafeSpeed;
			transform.position += fwdMove + strafeMove;
		}

		bool hasDevice = false;
#if USING_XR_SDK
		XRDisplaySubsystem currentDisplaySubsystem = OVRManager.GetCurrentDisplaySubsystem();
		if (currentDisplaySubsystem != null)
			hasDevice = currentDisplaySubsystem.running;
#elif REQUIRES_XR_SDK
		hasDevice = false;
#else
		hasDevice = UnityEngine.XR.XRDevice.isPresent;
#endif

		if ( !hasDevice && ( AllowYawLook || AllowPitchLook ) )
		{
			Quaternion r = transform.rotation;
			if ( AllowYawLook )
			{
				float gamePadYaw = OVRInput.Get(OVRInput.RawAxis2D.RThumbstick).x;
				float yawAmount = gamePadYaw * Time.deltaTime * GamePad_YawDegreesPerSec;
				Quaternion yawRot = Quaternion.AngleAxis( yawAmount, Vector3.up );
				r = yawRot * r;
			}
			if ( AllowPitchLook )
			{
				float gamePadPitch = OVRInput.Get(OVRInput.RawAxis2D.RThumbstick).y;
				if ( Mathf.Abs( gamePadPitch ) > 0.0001f )
				{
					if ( InvertPitch )
					{
						gamePadPitch *= -1.0f;
					}
					float pitchAmount = gamePadPitch * Time.deltaTime * GamePad_PitchDegreesPerSec;
					Quaternion pitchRot = Quaternion.AngleAxis( pitchAmount, Vector3.left );
					r = r * pitchRot;
				}
			}

			transform.rotation = r;
		}
	}
}
