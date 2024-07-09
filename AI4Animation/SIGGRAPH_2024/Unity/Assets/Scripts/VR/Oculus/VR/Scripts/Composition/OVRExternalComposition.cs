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

using System;
using UnityEngine;
using System.Collections.Generic;
using System.Threading;
using UnityEngine.Rendering;
using Object = UnityEngine.Object;

#if USING_URP
using UnityEngine.Rendering.Universal;
#endif

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || UNITY_ANDROID

public class OVRExternalComposition : OVRComposition
{
	private GameObject previousMainCameraObject = null;
	public GameObject foregroundCameraGameObject = null;
	public Camera foregroundCamera = null;
	public GameObject backgroundCameraGameObject = null;
	public Camera backgroundCamera = null;
#if OVR_ANDROID_MRC
	private bool skipFrame = false;
	private float fpsThreshold = 80.0f;
	private bool isFrameSkipped = true;
	public bool renderCombinedFrame = false;
	public AudioListener audioListener;
	public OVRMRAudioFilter audioFilter;
	public RenderTexture[] mrcRenderTextureArray = new RenderTexture[2];
	public int frameIndex;
	public int lastMrcEncodeFrameSyncId;

	// when rendererSupportsCameraRect is false, mrcRenderTextureArray would only store the background frame (regular width)
	public RenderTexture[] mrcForegroundRenderTextureArray = new RenderTexture[2];

	// this is used for moving MRC camera where we would need to be able to synchronize the camera position from the game with that on the client for composition
	public double[] cameraPoseTimeArray = new double[2];
#endif

	public override OVRManager.CompositionMethod CompositionMethod() { return OVRManager.CompositionMethod.External; }

	public OVRExternalComposition(GameObject parentObject, Camera mainCamera, OVRMixedRealityCaptureConfiguration configuration)
		: base(parentObject, mainCamera, configuration)
	{

#if OVR_ANDROID_MRC
		renderCombinedFrame = false;

		int frameWidth;
		int frameHeight;
		OVRPlugin.Media.GetMrcFrameSize(out frameWidth, out frameHeight);
		Debug.LogFormat("[OVRExternalComposition] Create render texture {0}, {1}", renderCombinedFrame ? frameWidth : frameWidth/2, frameHeight);
		for (int i=0; i<2; ++i)
		{
			mrcRenderTextureArray[i] = new RenderTexture(renderCombinedFrame ? frameWidth : frameWidth/2, frameHeight, 24, RenderTextureFormat.ARGB32);
			mrcRenderTextureArray[i].Create();
			cameraPoseTimeArray[i] = 0.0;
		}

		skipFrame = OVRManager.display.displayFrequency > fpsThreshold;
		OVRManager.DisplayRefreshRateChanged += DisplayRefreshRateChanged;
		frameIndex = 0;
		lastMrcEncodeFrameSyncId = -1;

		if (!renderCombinedFrame)
		{
			Debug.LogFormat("[OVRExternalComposition] Create extra render textures for foreground");
			for (int i = 0; i < 2; ++i)
			{
				mrcForegroundRenderTextureArray[i] = new RenderTexture(frameWidth / 2, frameHeight, 24, RenderTextureFormat.ARGB32);
				mrcForegroundRenderTextureArray[i].Create();
			}
		}
#endif
		RefreshCameraObjects(parentObject, mainCamera, configuration);
	}

	private void RefreshCameraObjects(GameObject parentObject, Camera mainCamera, OVRMixedRealityCaptureConfiguration configuration)
	{
		if (mainCamera.gameObject != previousMainCameraObject)
		{
			Debug.LogFormat("[OVRExternalComposition] Camera refreshed. Rebind camera to {0}", mainCamera.gameObject.name);

			OVRCompositionUtil.SafeDestroy(ref backgroundCameraGameObject);
			backgroundCamera = null;
			OVRCompositionUtil.SafeDestroy(ref foregroundCameraGameObject);
			foregroundCamera = null;

			RefreshCameraRig(parentObject, mainCamera);

			Debug.Assert(backgroundCameraGameObject == null);
			if (configuration.instantiateMixedRealityCameraGameObject != null)
			{
				backgroundCameraGameObject = configuration.instantiateMixedRealityCameraGameObject(mainCamera.gameObject, OVRManager.MrcCameraType.Background);
			}
			else
			{
				backgroundCameraGameObject = Object.Instantiate(mainCamera.gameObject);
			}

			backgroundCameraGameObject.name = "OculusMRC_BackgroundCamera";
			backgroundCameraGameObject.transform.parent =
				cameraInTrackingSpace ? cameraRig.trackingSpace : parentObject.transform;
			if (backgroundCameraGameObject.GetComponent<AudioListener>()) {
				Object.Destroy(backgroundCameraGameObject.GetComponent<AudioListener>());
			}

			if (backgroundCameraGameObject.GetComponent<OVRManager>()) {
				Object.Destroy(backgroundCameraGameObject.GetComponent<OVRManager>());
			}
			backgroundCamera = backgroundCameraGameObject.GetComponent<Camera>();
			backgroundCamera.tag = "Untagged";
#if USING_MRC_COMPATIBLE_URP_VERSION
			var backgroundCamData = backgroundCamera.GetUniversalAdditionalCameraData();
			if (backgroundCamData != null)
			{
				backgroundCamData.allowXRRendering = false;
			}
#elif USING_URP
			Debug.LogError("Using URP with MRC is only supported with URP version 10.0.0 or higher. Consider using Unity 2020 or higher.");
#else
			backgroundCamera.stereoTargetEye = StereoTargetEyeMask.None;
#endif
			backgroundCamera.depth = 99990.0f;
			backgroundCamera.rect = new Rect(0.0f, 0.0f, 0.5f, 1.0f);
			backgroundCamera.cullingMask = (backgroundCamera.cullingMask & ~configuration.extraHiddenLayers) | configuration.extraVisibleLayers;
#if OVR_ANDROID_MRC
			backgroundCamera.targetTexture = mrcRenderTextureArray[0];
			if (!renderCombinedFrame)
			{
				backgroundCamera.rect = new Rect(0.0f, 0.0f, 1.0f, 1.0f);
			}
#endif

			Debug.Assert(foregroundCameraGameObject == null);
			if (configuration.instantiateMixedRealityCameraGameObject != null)
			{
				foregroundCameraGameObject = configuration.instantiateMixedRealityCameraGameObject(mainCamera.gameObject, OVRManager.MrcCameraType.Foreground);
			}
			else
			{
				foregroundCameraGameObject = Object.Instantiate(mainCamera.gameObject);
			}

			foregroundCameraGameObject.name = "OculusMRC_ForgroundCamera";
			foregroundCameraGameObject.transform.parent = cameraInTrackingSpace ? cameraRig.trackingSpace : parentObject.transform;
			if (foregroundCameraGameObject.GetComponent<AudioListener>())
			{
				Object.Destroy(foregroundCameraGameObject.GetComponent<AudioListener>());
			}
			if (foregroundCameraGameObject.GetComponent<OVRManager>())
			{
				Object.Destroy(foregroundCameraGameObject.GetComponent<OVRManager>());
			}
			foregroundCamera = foregroundCameraGameObject.GetComponent<Camera>();
			foregroundCamera.tag = "Untagged";
#if USING_MRC_COMPATIBLE_URP_VERSION
			var foregroundCamData = foregroundCamera.GetUniversalAdditionalCameraData();
			if (foregroundCamData != null)
			{
				foregroundCamData.allowXRRendering = false;
			}
#elif USING_URP
			Debug.LogError("Using URP with MRC is only supported with URP version 10.0.0 or higher. Consider using Unity 2020 or higher.");
#else
			foregroundCamera.stereoTargetEye = StereoTargetEyeMask.None;
#endif
			foregroundCamera.depth = backgroundCamera.depth + 1.0f;     // enforce the forground be rendered after the background
			foregroundCamera.rect = new Rect(0.5f, 0.0f, 0.5f, 1.0f);
			foregroundCamera.clearFlags = CameraClearFlags.Color;
#if OVR_ANDROID_MRC
			foregroundCamera.backgroundColor = configuration.externalCompositionBackdropColorQuest;
#else
			foregroundCamera.backgroundColor = configuration.externalCompositionBackdropColorRift;
#endif
			foregroundCamera.cullingMask = (foregroundCamera.cullingMask & ~configuration.extraHiddenLayers) | configuration.extraVisibleLayers;

#if OVR_ANDROID_MRC
			if (renderCombinedFrame)
			{
				foregroundCamera.targetTexture = mrcRenderTextureArray[0];
			}
			else
			{
				foregroundCamera.targetTexture = mrcForegroundRenderTextureArray[0];
				foregroundCamera.rect = new Rect(0.0f, 0.0f, 1.0f, 1.0f);
			}
#endif

			// [Debug] Uncommenting the following code will put a cube on the external camera location for visualization
			//GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
			//cube.transform.parent = foregroundCameraGameObject.transform;
			//cube.transform.localPosition = Vector3.zero;
			//cube.transform.localRotation = Quaternion.identity;
			//cube.transform.localScale = Vector3.one * 0.1f;

			previousMainCameraObject = mainCamera.gameObject;
		}
	}

#if OVR_ANDROID_MRC
	private void RefreshAudioFilter(Camera mainCamera)
	{
		if (audioListener == null || !audioListener.enabled || !audioListener.gameObject.activeInHierarchy)
		{
			CleanupAudioFilter();

			// first try cameraRig
			AudioListener tmpAudioListener = cameraRig != null && cameraRig.centerEyeAnchor.gameObject.activeInHierarchy
				? cameraRig.centerEyeAnchor.GetComponent<AudioListener>()
				: null;

			// second try mainCamera
			if (tmpAudioListener == null || !tmpAudioListener.enabled)
			{
				tmpAudioListener = mainCamera != null && mainCamera.gameObject.activeInHierarchy
					? mainCamera.GetComponent<AudioListener>()
					: null;
			}

			// third try Camera.main (expensive)
			if (tmpAudioListener == null || !tmpAudioListener.enabled)
			{
				mainCamera = Camera.main;
				tmpAudioListener = mainCamera != null && mainCamera.gameObject.activeInHierarchy
					? mainCamera.GetComponent<AudioListener>()
					: null;
			}

			// fourth, search for all AudioListeners (very expensive)
			if (tmpAudioListener == null || !tmpAudioListener.enabled)
			{
				Object[] allListeners = Object.FindObjectsOfType<AudioListener>();
				foreach (var l in allListeners)
				{
					AudioListener al = l as AudioListener;
					if (al != null && al.enabled && al.gameObject.activeInHierarchy)
					{
						tmpAudioListener = al;
						break;
					}
				}
			}
			if (tmpAudioListener == null || !tmpAudioListener.enabled)
			{
				Debug.LogWarning("[OVRExternalComposition] No AudioListener in scene");
			}
			else
			{
				Debug.LogFormat("[OVRExternalComposition] AudioListener found, obj {0}", tmpAudioListener.gameObject.name);
				audioListener = tmpAudioListener;
				audioFilter = audioListener.gameObject.AddComponent<OVRMRAudioFilter>();
				audioFilter.composition = this;
				Debug.LogFormat("OVRMRAudioFilter added");
			}
		}
	}

	private float[] cachedAudioDataArray = null;

	private int CastMrcFrame(int castTextureIndex)
	{
		int audioFrames;
		int audioChannels;
		GetAndResetAudioData(ref cachedAudioDataArray, out audioFrames, out audioChannels);

		int syncId = -1;
		//Debug.Log("EncodeFrameThreadObject EncodeMrcFrame");
		bool ret = false;
		if (OVRPlugin.Media.GetMrcInputVideoBufferType() == OVRPlugin.Media.InputVideoBufferType.TextureHandle)
		{
			ret = OVRPlugin.Media.EncodeMrcFrame(mrcRenderTextureArray[castTextureIndex].GetNativeTexturePtr(),
				renderCombinedFrame ? System.IntPtr.Zero : mrcForegroundRenderTextureArray[castTextureIndex].GetNativeTexturePtr(),
				cachedAudioDataArray, audioFrames, audioChannels, AudioSettings.dspTime, cameraPoseTimeArray[castTextureIndex], ref syncId);
		}
		else
		{
			ret = OVRPlugin.Media.EncodeMrcFrame(mrcRenderTextureArray[castTextureIndex], cachedAudioDataArray, audioFrames, audioChannels, AudioSettings.dspTime, cameraPoseTimeArray[castTextureIndex], ref syncId);
		}

		if (!ret)
		{
			Debug.LogWarning("EncodeMrcFrame failed. Likely caused by OBS plugin disconnection");
			return -1;
		}

		return syncId;
	}

	private void SetCameraTargetTexture(int drawTextureIndex)
	{
		if (renderCombinedFrame)
		{
			RenderTexture texture = mrcRenderTextureArray[drawTextureIndex];
			if (backgroundCamera.targetTexture != texture)
			{
				backgroundCamera.targetTexture = texture;
			}
			if (foregroundCamera.targetTexture != texture)
			{
				foregroundCamera.targetTexture = texture;
			}
		}
		else
		{
			RenderTexture bgTexture = mrcRenderTextureArray[drawTextureIndex];
			RenderTexture fgTexture = mrcForegroundRenderTextureArray[drawTextureIndex];
			if (backgroundCamera.targetTexture != bgTexture)
			{
				backgroundCamera.targetTexture = bgTexture;
			}
			if (foregroundCamera.targetTexture != fgTexture)
			{
				foregroundCamera.targetTexture = fgTexture;
			}
		}
	}
#endif


	public override void Update(GameObject gameObject, Camera mainCamera, OVRMixedRealityCaptureConfiguration configuration, OVRManager.TrackingOrigin trackingOrigin)
	{
#if OVR_ANDROID_MRC
		if (skipFrame && OVRPlugin.Media.IsCastingToRemoteClient()) {
			isFrameSkipped = !isFrameSkipped;
			if(isFrameSkipped) { return; }
		}
#endif

		RefreshCameraObjects(gameObject, mainCamera, configuration);

		OVRPlugin.SetHandNodePoseStateLatency(0.0);     // the HandNodePoseStateLatency doesn't apply to the external composition. Always enforce it to 0.0

		// For third-person camera to use for calculating camera position with different anchors
		OVRPose stageToLocalPose = OVRPlugin.GetTrackingTransformRelativePose(OVRPlugin.TrackingOrigin.Stage).ToOVRPose();
		OVRPose localToStagePose = stageToLocalPose.Inverse();
		OVRPose head = localToStagePose * OVRPlugin.GetNodePose(OVRPlugin.Node.Head, OVRPlugin.Step.Render).ToOVRPose();
		OVRPose leftC = localToStagePose * OVRPlugin.GetNodePose(OVRPlugin.Node.HandLeft, OVRPlugin.Step.Render).ToOVRPose();
		OVRPose rightC = localToStagePose * OVRPlugin.GetNodePose(OVRPlugin.Node.HandRight, OVRPlugin.Step.Render).ToOVRPose();
		OVRPlugin.Media.SetMrcHeadsetControllerPose(head.ToPosef(), leftC.ToPosef(), rightC.ToPosef());

#if OVR_ANDROID_MRC
		RefreshAudioFilter(mainCamera);

		int drawTextureIndex = (frameIndex / 2) % 2;
		int castTextureIndex = 1 - drawTextureIndex;

		backgroundCamera.enabled = (frameIndex % 2) == 0;
		foregroundCamera.enabled = (frameIndex % 2) == 1;

		if (frameIndex % 2 == 0)
		{
			if (lastMrcEncodeFrameSyncId != -1)
			{
				OVRPlugin.Media.SyncMrcFrame(lastMrcEncodeFrameSyncId);
				lastMrcEncodeFrameSyncId = -1;
			}
			lastMrcEncodeFrameSyncId = CastMrcFrame(castTextureIndex);
			SetCameraTargetTexture(drawTextureIndex);
		}

		++ frameIndex;
#endif

		backgroundCamera.clearFlags = mainCamera.clearFlags;
		backgroundCamera.backgroundColor = mainCamera.backgroundColor;
		if (configuration.dynamicCullingMask)
		{
			backgroundCamera.cullingMask = (mainCamera.cullingMask & ~configuration.extraHiddenLayers) | configuration.extraVisibleLayers;
		}
		backgroundCamera.nearClipPlane = mainCamera.nearClipPlane;
		backgroundCamera.farClipPlane = mainCamera.farClipPlane;

		if (configuration.dynamicCullingMask)
		{
			foregroundCamera.cullingMask = (mainCamera.cullingMask & ~configuration.extraHiddenLayers) | configuration.extraVisibleLayers;
		}
		foregroundCamera.nearClipPlane = mainCamera.nearClipPlane;
		foregroundCamera.farClipPlane = mainCamera.farClipPlane;

		if (OVRMixedReality.useFakeExternalCamera || OVRPlugin.GetExternalCameraCount() == 0)
		{
			OVRPose worldSpacePose = new OVRPose();
			OVRPose trackingSpacePose = new OVRPose();
			trackingSpacePose.position = trackingOrigin == OVRManager.TrackingOrigin.EyeLevel ?
				OVRMixedReality.fakeCameraEyeLevelPosition :
				OVRMixedReality.fakeCameraFloorLevelPosition;
			trackingSpacePose.orientation = OVRMixedReality.fakeCameraRotation;
			worldSpacePose = OVRExtensions.ToWorldSpacePose(trackingSpacePose, mainCamera);

			backgroundCamera.fieldOfView = OVRMixedReality.fakeCameraFov;
			backgroundCamera.aspect = OVRMixedReality.fakeCameraAspect;
			foregroundCamera.fieldOfView = OVRMixedReality.fakeCameraFov;
			foregroundCamera.aspect = OVRMixedReality.fakeCameraAspect;

			if (cameraInTrackingSpace)
			{
				backgroundCamera.transform.FromOVRPose(trackingSpacePose, true);
				foregroundCamera.transform.FromOVRPose(trackingSpacePose, true);
			}
			else
			{
				backgroundCamera.transform.FromOVRPose(worldSpacePose);
				foregroundCamera.transform.FromOVRPose(worldSpacePose);
			}
		}
		else
		{
			OVRPlugin.CameraExtrinsics extrinsics;
			OVRPlugin.CameraIntrinsics intrinsics;

			// So far, only support 1 camera for MR and always use camera index 0
			if (OVRPlugin.GetMixedRealityCameraInfo(0, out extrinsics, out intrinsics))
			{
				float fovY = Mathf.Atan(intrinsics.FOVPort.UpTan) * Mathf.Rad2Deg * 2;
				float aspect = intrinsics.FOVPort.LeftTan / intrinsics.FOVPort.UpTan;
				backgroundCamera.fieldOfView = fovY;
				backgroundCamera.aspect = aspect;
				foregroundCamera.fieldOfView = fovY;
				foregroundCamera.aspect = intrinsics.FOVPort.LeftTan / intrinsics.FOVPort.UpTan;

				if (cameraInTrackingSpace)
				{
					OVRPose trackingSpacePose = ComputeCameraTrackingSpacePose(extrinsics);
					backgroundCamera.transform.FromOVRPose(trackingSpacePose, true);
					foregroundCamera.transform.FromOVRPose(trackingSpacePose, true);
				}
				else
				{
					OVRPose worldSpacePose = ComputeCameraWorldSpacePose(extrinsics, mainCamera);
					backgroundCamera.transform.FromOVRPose(worldSpacePose);
					foregroundCamera.transform.FromOVRPose(worldSpacePose);
				}
#if OVR_ANDROID_MRC
				cameraPoseTimeArray[drawTextureIndex] = extrinsics.LastChangedTimeSeconds;
#endif
			}
			else
			{
				Debug.LogError("Failed to get external camera information");
				return;
			}
		}

		Vector3 headToExternalCameraVec = mainCamera.transform.position - foregroundCamera.transform.position;
		float clipDistance = Vector3.Dot(headToExternalCameraVec, foregroundCamera.transform.forward);
		foregroundCamera.farClipPlane = Mathf.Max(foregroundCamera.nearClipPlane + 0.001f, clipDistance);
	}

#if OVR_ANDROID_MRC
	private void CleanupAudioFilter()
	{
		if (audioFilter)
		{
			audioFilter.composition = null;
			Object.Destroy(audioFilter);
			Debug.LogFormat("OVRMRAudioFilter destroyed");
			audioFilter = null;
		}

	}
#endif

	public override void Cleanup()
	{
		OVRCompositionUtil.SafeDestroy(ref backgroundCameraGameObject);
		backgroundCamera = null;
		OVRCompositionUtil.SafeDestroy(ref foregroundCameraGameObject);
		foregroundCamera = null;
		Debug.Log("ExternalComposition deactivated");

#if OVR_ANDROID_MRC
		if (lastMrcEncodeFrameSyncId != -1)
		{
			OVRPlugin.Media.SyncMrcFrame(lastMrcEncodeFrameSyncId);
			lastMrcEncodeFrameSyncId = -1;
		}

		CleanupAudioFilter();

		for (int i=0; i<2; ++i)
		{
			mrcRenderTextureArray[i].Release();
			mrcRenderTextureArray[i] = null;

			if (!renderCombinedFrame)
			{
				mrcForegroundRenderTextureArray[i].Release();
				mrcForegroundRenderTextureArray[i] = null;
			}
		}

		OVRManager.DisplayRefreshRateChanged -= DisplayRefreshRateChanged;
		frameIndex = 0;
#endif
	}

	private readonly object audioDataLock = new object();
	private List<float> cachedAudioData = new List<float>(16384);
	private int cachedChannels = 0;

	public void CacheAudioData(float[] data, int channels)
	{
		lock(audioDataLock)
		{
			if (channels != cachedChannels)
			{
				cachedAudioData.Clear();
			}
			cachedChannels = channels;
			cachedAudioData.AddRange(data);
			//Debug.LogFormat("[CacheAudioData] dspTime {0} indata {1} channels {2} accu_len {3}", AudioSettings.dspTime, data.Length, channels, cachedAudioData.Count);
		}
	}

	public void GetAndResetAudioData(ref float[] audioData, out int audioFrames, out int channels)
	{
		lock(audioDataLock)
		{
			//Debug.LogFormat("[GetAndResetAudioData] dspTime {0} accu_len {1}", AudioSettings.dspTime, cachedAudioData.Count);
			if (audioData == null || audioData.Length < cachedAudioData.Count)
			{
				audioData = new float[cachedAudioData.Capacity];
			}
			cachedAudioData.CopyTo(audioData);
			audioFrames = cachedAudioData.Count;
			channels = cachedChannels;
			cachedAudioData.Clear();
		}
	}

#if OVR_ANDROID_MRC

	private void DisplayRefreshRateChanged(float fromRefreshRate, float toRefreshRate)
	{
		skipFrame = toRefreshRate > fpsThreshold;
	}
#endif

}

#if OVR_ANDROID_MRC

public class OVRMRAudioFilter : MonoBehaviour
{
	private bool running = false;

	public OVRExternalComposition composition;

	void Start()
	{
		running = true;
	}

	void OnAudioFilterRead(float[] data, int channels)
	{
		if (!running)
			return;

		if (composition != null)
		{
			composition.CacheAudioData(data, channels);
		}
	}
}
#endif

#endif
