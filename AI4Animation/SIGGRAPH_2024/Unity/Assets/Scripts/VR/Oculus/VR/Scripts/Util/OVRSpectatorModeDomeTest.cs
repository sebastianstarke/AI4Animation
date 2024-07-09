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

#if UNITY_ANDROID && !UNITY_EDITOR
#define OVR_ANDROID_MRC
#endif

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OVRSpectatorModeDomeTest : MonoBehaviour {

	bool inited = false;

	public Camera defaultExternalCamera;
	OVRPlugin.Fovf defaultFov;
	public Transform SpectatorAnchor;
	public Transform Head;
#if OVR_ANDROID_MRC
	private OVRPlugin.Media.PlatformCameraMode camMode = OVRPlugin.Media.PlatformCameraMode.Disabled;
	private bool readyToSwitch = false;
	private Transform SpectatorCamera;

	// Dome sphere representation
	private float distance = 0.8f;
	private float elevation = 0.0f;
	private float polar = 90.0f;
	private const float distance_near = 0.5f;
	private const float distance_far = 1.2f;
	private const float elevationLimit = 30.0f;
#endif

	// Start is called before the first frame update
	void Awake()
	{
#if OVR_ANDROID_MRC
		OVRPlugin.Media.SetPlatformInitialized();
		SpectatorCamera = defaultExternalCamera.transform.parent;
#endif
	}

	// Use this for initialization
	void Start ()
	{
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || OVR_ANDROID_MRC
		if (!defaultExternalCamera)
		{
			Debug.LogWarning("defaultExternalCamera undefined");
		}

#if !OVR_ANDROID_MRC
		// On Quest, we enable MRC automatically through the configuration
		if (!OVRManager.instance.enableMixedReality)
		{
			OVRManager.instance.enableMixedReality = true;
		}
#endif
#endif
	}

	void Initialize()
	{
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || OVR_ANDROID_MRC
		if (inited)
			return;

#if OVR_ANDROID_MRC
		if (!OVRPlugin.Media.GetInitialized())
			return;
#else
		if (!OVRPlugin.IsMixedRealityInitialized())
			return;
#endif

		OVRPlugin.ResetDefaultExternalCamera();
		Debug.LogFormat("GetExternalCameraCount before adding manual external camera {0}", OVRPlugin.GetExternalCameraCount());
		UpdateDefaultExternalCamera();
		Debug.LogFormat("GetExternalCameraCount after adding manual external camera {0}", OVRPlugin.GetExternalCameraCount());

		// obtain default FOV
		{
			OVRPlugin.CameraIntrinsics cameraIntrinsics;
			OVRPlugin.CameraExtrinsics cameraExtrinsics;
			OVRPlugin.GetMixedRealityCameraInfo(0, out cameraExtrinsics, out cameraIntrinsics);
			defaultFov = cameraIntrinsics.FOVPort;
		}

		inited = true;
#if OVR_ANDROID_MRC
		readyToSwitch = true;
#endif
#endif
	}

	void UpdateDefaultExternalCamera()
	{
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || OVR_ANDROID_MRC
		// always build a 1080p external camera
		const int cameraPixelWidth = 1920;
		const int cameraPixelHeight = 1080;
		const float cameraAspect = (float)cameraPixelWidth / cameraPixelHeight;


		string cameraName = "UnityExternalCamera";
		OVRPlugin.CameraIntrinsics cameraIntrinsics = new OVRPlugin.CameraIntrinsics();
		OVRPlugin.CameraExtrinsics cameraExtrinsics = new OVRPlugin.CameraExtrinsics();

		// intrinsics

		cameraIntrinsics.IsValid = OVRPlugin.Bool.True;
		cameraIntrinsics.LastChangedTimeSeconds = Time.time;

		float vFov = defaultExternalCamera.fieldOfView * Mathf.Deg2Rad;
		float hFov = Mathf.Atan(Mathf.Tan(vFov * 0.5f) * cameraAspect) * 2.0f;
		OVRPlugin.Fovf fov = new OVRPlugin.Fovf();
		fov.UpTan = fov.DownTan = Mathf.Tan(vFov * 0.5f);
		fov.LeftTan = fov.RightTan = Mathf.Tan(hFov * 0.5f);

		cameraIntrinsics.FOVPort = fov;
		cameraIntrinsics.VirtualNearPlaneDistanceMeters = defaultExternalCamera.nearClipPlane;
		cameraIntrinsics.VirtualFarPlaneDistanceMeters = defaultExternalCamera.farClipPlane;
		cameraIntrinsics.ImageSensorPixelResolution.w = cameraPixelWidth;
		cameraIntrinsics.ImageSensorPixelResolution.h = cameraPixelHeight;

		// extrinsics

		cameraExtrinsics.IsValid = OVRPlugin.Bool.True;
		cameraExtrinsics.LastChangedTimeSeconds = Time.time;
		cameraExtrinsics.CameraStatusData = OVRPlugin.CameraStatus.CameraStatus_Calibrated;
		cameraExtrinsics.AttachedToNode = OVRPlugin.Node.None;

		Camera mainCamera = Camera.main;
		OVRCameraRig cameraRig = mainCamera.GetComponentInParent<OVRCameraRig>();
		if (cameraRig)
		{
			Transform trackingSpace = cameraRig.trackingSpace;
			OVRPose trackingSpacePose = trackingSpace.ToOVRPose(false);
			OVRPose cameraPose = defaultExternalCamera.transform.ToOVRPose(false);
			OVRPose relativePose = trackingSpacePose.Inverse() * cameraPose;
#if OVR_ANDROID_MRC
			OVRPose stageToLocalPose = OVRPlugin.GetTrackingTransformRelativePose(OVRPlugin.TrackingOrigin.Stage).ToOVRPose();
			OVRPose localToStagePose = stageToLocalPose.Inverse();
			relativePose = localToStagePose * relativePose;
#endif
			cameraExtrinsics.RelativePose = relativePose.ToPosef();
		}
		else
		{
			cameraExtrinsics.RelativePose = OVRPlugin.Posef.identity;
		}

		if (!OVRPlugin.SetDefaultExternalCamera(cameraName, ref cameraIntrinsics, ref cameraExtrinsics))
		{
			Debug.LogError("SetDefaultExternalCamera() failed");
		}
#endif
	}

	private void UpdateSpectatorCameraStatus()
	{
#if OVR_ANDROID_MRC
		// Trigger to switch between 1st person and spectator mode during casting to phone
		if (OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger) || OVRInput.GetDown(OVRInput.Button.SecondaryIndexTrigger))
		{
			camMode = OVRPlugin.Media.GetPlatformCameraMode();

			if (camMode == OVRPlugin.Media.PlatformCameraMode.Disabled && readyToSwitch)
			{
				OVRPlugin.Media.SetMrcFrameImageFlipped(false);
				OVRPlugin.Media.SetPlatformCameraMode(OVRPlugin.Media.PlatformCameraMode.Initialized);
				StartCoroutine(TimerCoroutine());
			}

			if (camMode == OVRPlugin.Media.PlatformCameraMode.Initialized && readyToSwitch)
			{
				OVRPlugin.Media.SetMrcFrameImageFlipped(true);
				OVRPlugin.Media.SetPlatformCameraMode(OVRPlugin.Media.PlatformCameraMode.Disabled);
				StartCoroutine(TimerCoroutine());
			}
		}

		// Keep spectator camera on dome surface 
		Vector2 axis = OVRInput.Get(OVRInput.Axis2D.SecondaryThumbstick);
		if (Mathf.Abs(axis.x) > 0.2f)
		{
			polar = polar - axis.x * 0.5f;
		}

		if (Mathf.Abs(axis.y) > 0.2f)
		{
			elevation = elevation + axis.y * 0.5f;
			if (elevation < -90.0f + elevationLimit) elevation = -90.0f + elevationLimit;
			if (elevation > 90.0f) elevation = 90.0f;
		}

		axis = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick);
		if (Mathf.Abs(axis.y) > 0.1f)
		{
			distance = axis.y * 0.05f + distance;
			if (distance > distance_far) distance = distance_far;
			if (distance < distance_near) distance = distance_near;
		}

		SpectatorCamera.position = SpectatorCameraDomePosition(SpectatorAnchor.position, distance, elevation, polar);
		SpectatorCamera.rotation = Quaternion.LookRotation(SpectatorCamera.position - SpectatorAnchor.position);
		Head.position = SpectatorAnchor.position;
		Head.rotation = SpectatorAnchor.rotation;
#endif
	}

	Vector3 SpectatorCameraDomePosition(Vector3 spectatorAnchorPosition, float d, float e, float p)
	{
		float x = d * Mathf.Cos(Mathf.Deg2Rad * e) * Mathf.Cos(Mathf.Deg2Rad * p);
		float y = d * Mathf.Sin(Mathf.Deg2Rad * e);
		float z = d * Mathf.Cos(Mathf.Deg2Rad * e) * Mathf.Sin(Mathf.Deg2Rad * p);

		return new Vector3(x + spectatorAnchorPosition.x, y + spectatorAnchorPosition.y, z + spectatorAnchorPosition.z);
	}

	IEnumerator TimerCoroutine()
	{
#if OVR_ANDROID_MRC
		readyToSwitch = false;
#endif
		yield return new WaitForSeconds(2);
#if OVR_ANDROID_MRC
		readyToSwitch = true;
#endif
	}

	// Update is called once per frame
	void Update () {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || OVR_ANDROID_MRC
		if (!inited)
		{
			Initialize();
			return;
		}

		if (!defaultExternalCamera)
		{
			return;
		}

#if OVR_ANDROID_MRC
		if (!OVRPlugin.Media.GetInitialized())
		{
			return;
		}
#else
		if (!OVRPlugin.IsMixedRealityInitialized())
		{
			return;
		}
#endif
		UpdateSpectatorCameraStatus();

		UpdateDefaultExternalCamera();
		OVRPlugin.OverrideExternalCameraFov(0, false, new OVRPlugin.Fovf());
		OVRPlugin.OverrideExternalCameraStaticPose(0, false, OVRPlugin.Posef.identity);

#endif
	}

	void OnApplicationPause()
	{
#if OVR_ANDROID_MRC
		OVRPlugin.Media.SetMrcFrameImageFlipped(true);
		OVRPlugin.Media.SetPlatformCameraMode(OVRPlugin.Media.PlatformCameraMode.Disabled);
#endif
	}

	void OnApplicationQuit()
	{
#if OVR_ANDROID_MRC
		OVRPlugin.Media.SetMrcFrameImageFlipped(true);
		OVRPlugin.Media.SetPlatformCameraMode(OVRPlugin.Media.PlatformCameraMode.Disabled);
#endif
	}
}
