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

using System;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using UnityEngine;
using System.Collections.Generic;

#if USING_XR_SDK
using UnityEngine.XR;
using UnityEngine.Experimental.XR;
#endif

using InputTracking = UnityEngine.XR.InputTracking;
using Node = UnityEngine.XR.XRNode;
using Settings = UnityEngine.XR.XRSettings;

/// <summary>
/// Manages an Oculus Rift head-mounted display (HMD).
/// </summary>
public class OVRDisplay
{
	/// <summary>
	/// Contains full fov information per eye
	/// Under Symmetric Fov mode, UpFov == DownFov and LeftFov == RightFov.
	/// </summary>
	public struct EyeFov
	{
		public float UpFov;
		public float DownFov;
		public float LeftFov;
		public float RightFov;
	}

	/// <summary>
	/// Specifies the size and field-of-view for one eye texture.
	/// </summary>
	public struct EyeRenderDesc
	{
		/// <summary>
		/// The horizontal and vertical size of the texture.
		/// </summary>
		public Vector2 resolution;

		/// <summary>
		/// The angle of the horizontal and vertical field of view in degrees.
		/// For Symmetric FOV interface compatibility
		/// Note this includes the fov angle from both sides
		/// </summary>
		public Vector2 fov;

		/// <summary>
		/// The full information of field of view in degrees.
		/// When Asymmetric FOV isn't enabled, this returns the maximum fov angle
		/// </summary>
		public EyeFov fullFov;
	}

	/// <summary>
	/// Contains latency measurements for a single frame of rendering.
	/// </summary>
	public struct LatencyData
	{
		/// <summary>
		/// The time it took to render both eyes in seconds.
		/// </summary>
		public float render;

		/// <summary>
		/// The time it took to perform TimeWarp in seconds.
		/// </summary>
		public float timeWarp;

		/// <summary>
		/// The time between the end of TimeWarp and scan-out in seconds.
		/// </summary>
		public float postPresent;
		public float renderError;
		public float timeWarpError;
	}

	private bool needsConfigureTexture;
	private EyeRenderDesc[] eyeDescs = new EyeRenderDesc[2];
	private bool recenterRequested = false;
	private int recenterRequestedFrameCount = int.MaxValue;
	private int localTrackingSpaceRecenterCount = 0;

	/// <summary>
	/// Creates an instance of OVRDisplay. Called by OVRManager.
	/// </summary>
	public OVRDisplay()
	{
		UpdateTextures();
	}

	/// <summary>
	/// Updates the internal state of the OVRDisplay. Called by OVRManager.
	/// </summary>
	public void Update()
	{
		UpdateTextures();

		if (recenterRequested && Time.frameCount > recenterRequestedFrameCount)
		{
			Debug.Log("Recenter event detected");
			if (RecenteredPose != null)
			{
				RecenteredPose();
			}
			recenterRequested = false;
			recenterRequestedFrameCount = int.MaxValue;
		}

		if (OVRPlugin.GetSystemHeadsetType() >= OVRPlugin.SystemHeadset.Oculus_Quest &&
			OVRPlugin.GetSystemHeadsetType() < OVRPlugin.SystemHeadset.Rift_DK1) // all Oculus Standalone headsets
		{
			int recenterCount = OVRPlugin.GetLocalTrackingSpaceRecenterCount();
			if (localTrackingSpaceRecenterCount != recenterCount)
			{
				Debug.Log("Recenter event detected");
				if (RecenteredPose != null)
				{
					RecenteredPose();
				}
				localTrackingSpaceRecenterCount = recenterCount;
			}
		}
	}

	/// <summary>
	/// Occurs when the head pose is reset.
	/// </summary>
	public event System.Action RecenteredPose;

	/// <summary>
	/// Recenters the head pose.
	/// </summary>
	public void RecenterPose()
	{
#if USING_XR_SDK
		XRInputSubsystem currentInputSubsystem = OVRManager.GetCurrentInputSubsystem();
		if (currentInputSubsystem != null)
		{
			currentInputSubsystem.TryRecenter();
		}
#elif !REQUIRES_XR_SDK
#pragma warning disable 618
		InputTracking.Recenter();
#pragma warning restore 618
#endif

		// The current poses are cached for the current frame and won't be updated immediately
		// after UnityEngine.VR.InputTracking.Recenter(). So we need to wait until next frame
		// to trigger the RecenteredPose delegate. The application could expect the correct pose
		// when the RecenteredPose delegate get called.
		recenterRequested = true;
		recenterRequestedFrameCount = Time.frameCount;

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
		OVRMixedReality.RecenterPose();
#endif
	}

	/// <summary>
	/// Gets the current linear acceleration of the head.
	/// </summary>
	public Vector3 acceleration
	{
		get {
			if (!OVRManager.isHmdPresent)
				return Vector3.zero;

			Vector3 retVec = Vector3.zero;
			if (OVRNodeStateProperties.GetNodeStatePropertyVector3(Node.Head, NodeStatePropertyType.Acceleration, OVRPlugin.Node.Head, OVRPlugin.Step.Render, out retVec))
				return retVec;
			return Vector3.zero;
		}
	}

	/// <summary>
	/// Gets the current angular acceleration of the head in radians per second per second about each axis.
	/// </summary>
	public Vector3 angularAcceleration
	{
		get
		{
			if (!OVRManager.isHmdPresent)
				return Vector3.zero;

			Vector3 retVec = Vector3.zero;
			if (OVRNodeStateProperties.GetNodeStatePropertyVector3(Node.Head, NodeStatePropertyType.AngularAcceleration, OVRPlugin.Node.Head, OVRPlugin.Step.Render, out retVec))
				return retVec;
			return Vector3.zero;

		}
	}

	/// <summary>
	/// Gets the current linear velocity of the head in meters per second.
	/// </summary>
	public Vector3 velocity
	{
		get
		{
			if (!OVRManager.isHmdPresent)
				return Vector3.zero;

			Vector3 retVec = Vector3.zero;
			if (OVRNodeStateProperties.GetNodeStatePropertyVector3(Node.Head, NodeStatePropertyType.Velocity, OVRPlugin.Node.Head, OVRPlugin.Step.Render, out retVec))
				return retVec;
			return Vector3.zero;
		}
	}

	/// <summary>
	/// Gets the current angular velocity of the head in radians per second about each axis.
	/// </summary>
	public Vector3 angularVelocity
	{
		get {
			if (!OVRManager.isHmdPresent)
				return Vector3.zero;

			Vector3 retVec = Vector3.zero;
			if (OVRNodeStateProperties.GetNodeStatePropertyVector3(Node.Head, NodeStatePropertyType.AngularVelocity, OVRPlugin.Node.Head, OVRPlugin.Step.Render, out retVec))
				return retVec;
			return Vector3.zero;
		}
	}

	/// <summary>
	/// Gets the resolution and field of view for the given eye.
	/// </summary>
	public EyeRenderDesc GetEyeRenderDesc(UnityEngine.XR.XRNode eye)
	{
		return eyeDescs[(int)eye];
	}

	/// <summary>
	/// Gets the current measured latency values.
	/// </summary>
	public LatencyData latency
	{
		get {
			if (!OVRManager.isHmdPresent)
				return new LatencyData();

			string latency = OVRPlugin.latency;

			var r = new Regex("Render: ([0-9]+[.][0-9]+)ms, TimeWarp: ([0-9]+[.][0-9]+)ms, PostPresent: ([0-9]+[.][0-9]+)ms", RegexOptions.None);

			var ret = new LatencyData();

			Match match = r.Match(latency);
			if (match.Success)
			{
				ret.render = float.Parse(match.Groups[1].Value);
				ret.timeWarp = float.Parse(match.Groups[2].Value);
				ret.postPresent = float.Parse(match.Groups[3].Value);
			}

			return ret;
		}
	}

	/// <summary>
	/// Gets application's frame rate reported by oculus plugin
	/// </summary>
	public float appFramerate
	{
		get
		{
			if (!OVRManager.isHmdPresent)
				return 0;

			return OVRPlugin.GetAppFramerate();
		}
	}

	/// <summary>
	/// Gets the recommended MSAA level for optimal quality/performance the current device.
	/// </summary>
	public int recommendedMSAALevel
	{
		get
		{
			int result = OVRPlugin.recommendedMSAALevel;

			if (result == 1)
				result = 0;

			return result;
		}
	}

	/// <summary>
	/// Gets the list of available display frequencies supported by this hardware.
	/// </summary>
	public float[] displayFrequenciesAvailable
	{
		get { return OVRPlugin.systemDisplayFrequenciesAvailable; }
	}

	/// <summary>
	/// Gets and sets the current display frequency.
	/// </summary>
	public float displayFrequency
	{
		get
		{
			return OVRPlugin.systemDisplayFrequency;
		}
		set
		{
			OVRPlugin.systemDisplayFrequency = value;
		}
	}

	private void UpdateTextures()
	{
		ConfigureEyeDesc(Node.LeftEye);
		ConfigureEyeDesc(Node.RightEye);
	}

	private void ConfigureEyeDesc(Node eye)
	{
		if (!OVRManager.isHmdPresent)
			return;

		int eyeTextureWidth = Settings.eyeTextureWidth;
		int eyeTextureHeight = Settings.eyeTextureHeight;

		eyeDescs[(int)eye] = new EyeRenderDesc();
		eyeDescs[(int)eye].resolution = new Vector2(eyeTextureWidth, eyeTextureHeight);

		OVRPlugin.Frustumf2 frust;
		if (OVRPlugin.GetNodeFrustum2((OVRPlugin.Node)eye, out frust))
		{
			eyeDescs[(int)eye].fullFov.LeftFov = Mathf.Rad2Deg * Mathf.Atan(frust.Fov.LeftTan);
			eyeDescs[(int)eye].fullFov.RightFov = Mathf.Rad2Deg * Mathf.Atan(frust.Fov.RightTan);
			eyeDescs[(int)eye].fullFov.UpFov = Mathf.Rad2Deg * Mathf.Atan(frust.Fov.UpTan);
			eyeDescs[(int)eye].fullFov.DownFov = Mathf.Rad2Deg * Mathf.Atan(frust.Fov.DownTan);
		}
		else
		{
			OVRPlugin.Frustumf frustOld = OVRPlugin.GetEyeFrustum((OVRPlugin.Eye)eye);
			eyeDescs[(int)eye].fullFov.LeftFov = Mathf.Rad2Deg * frustOld.fovX * 0.5f;
			eyeDescs[(int)eye].fullFov.RightFov = Mathf.Rad2Deg * frustOld.fovX * 0.5f;
			eyeDescs[(int)eye].fullFov.UpFov = Mathf.Rad2Deg * frustOld.fovY * 0.5f;
			eyeDescs[(int)eye].fullFov.DownFov = Mathf.Rad2Deg * frustOld.fovY * 0.5f;
		}

		// Symmetric Fov uses the maximum fov angle
		float maxFovX = Mathf.Max(eyeDescs[(int)eye].fullFov.LeftFov, eyeDescs[(int)eye].fullFov.RightFov);
		float maxFovY = Mathf.Max(eyeDescs[(int)eye].fullFov.UpFov, eyeDescs[(int)eye].fullFov.DownFov);
		eyeDescs[(int)eye].fov.x = maxFovX * 2.0f;
		eyeDescs[(int)eye].fov.y = maxFovY * 2.0f;

		if (!OVRPlugin.AsymmetricFovEnabled)
		{
			eyeDescs[(int)eye].fullFov.LeftFov = maxFovX;
			eyeDescs[(int)eye].fullFov.RightFov = maxFovX;

			eyeDescs[(int)eye].fullFov.UpFov = maxFovY;
			eyeDescs[(int)eye].fullFov.DownFov = maxFovY;
		}
	}
}
