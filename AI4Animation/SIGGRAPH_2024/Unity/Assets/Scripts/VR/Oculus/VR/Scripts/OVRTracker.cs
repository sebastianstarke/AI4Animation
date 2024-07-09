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

using System;
using System.Runtime.InteropServices;
using UnityEngine;

/// <summary>
/// An infrared camera that tracks the position of a head-mounted display.
/// </summary>
public class OVRTracker
{
	/// <summary>
	/// The (symmetric) visible area in front of the sensor.
	/// </summary>
	public struct Frustum
	{
		/// <summary>
		/// The sensor's minimum supported distance to the HMD.
		/// </summary>
		public float nearZ;
		/// <summary>
		/// The sensor's maximum supported distance to the HMD.
		/// </summary>
		public float farZ;
		/// <summary>
		/// The sensor's horizontal and vertical fields of view in degrees.
		/// </summary>
		public Vector2 fov;
	}

	/// <summary>
	/// If true, a sensor is attached to the system.
	/// </summary>
	public bool isPresent
	{
		get {
			if (!OVRManager.isHmdPresent)
				return false;

			return OVRPlugin.positionSupported;
		}
	}

	/// <summary>
	/// If true, the sensor is actively tracking the HMD's position. Otherwise the HMD may be temporarily occluded, the system may not support position tracking, etc.
	/// </summary>
	public bool isPositionTracked
	{
		get {
			return OVRPlugin.positionTracked;
		}
	}

	/// <summary>
	/// If this is true and a sensor is available, the system will use position tracking when isPositionTracked is also true.
	/// </summary>
	public bool isEnabled
	{
		get {
			if (!OVRManager.isHmdPresent)
				return false;

			return OVRPlugin.position;
		}

		set {
			if (!OVRManager.isHmdPresent)
				return;

			OVRPlugin.position = value;
		}
	}

	/// <summary>
	/// Returns the number of sensors currently connected to the system.
	/// </summary>
	public int count
	{
		get {
			int count = 0;

			for (int i = 0; i < (int)OVRPlugin.Tracker.Count; ++i)
			{
				if (GetPresent(i))
					count++;
			}

			return count;
		}
	}

	/// <summary>
	/// Gets the sensor's viewing frustum.
	/// </summary>
	public Frustum GetFrustum(int tracker = 0)
	{
		if (!OVRManager.isHmdPresent)
			return new Frustum();

		return OVRPlugin.GetTrackerFrustum((OVRPlugin.Tracker)tracker).ToFrustum();
	}

	/// <summary>
	/// Gets the sensor's pose, relative to the head's pose at the time of the last pose recentering.
	/// </summary>
	public OVRPose GetPose(int tracker = 0)
	{
		if (!OVRManager.isHmdPresent)
			return OVRPose.identity;

		OVRPose p;
		switch (tracker)
		{
			case 0:
				p = OVRPlugin.GetNodePose(OVRPlugin.Node.TrackerZero, OVRPlugin.Step.Render).ToOVRPose();
				break;
			case 1:
				p = OVRPlugin.GetNodePose(OVRPlugin.Node.TrackerOne, OVRPlugin.Step.Render).ToOVRPose();
				break;
			case 2:
				p = OVRPlugin.GetNodePose(OVRPlugin.Node.TrackerTwo, OVRPlugin.Step.Render).ToOVRPose();
				break;
			case 3:
				p = OVRPlugin.GetNodePose(OVRPlugin.Node.TrackerThree, OVRPlugin.Step.Render).ToOVRPose();
				break;
			default:
				return OVRPose.identity;
		}

		return new OVRPose()
		{
			position = p.position,
			orientation = p.orientation * Quaternion.Euler(0, 180, 0)
		};
	}

	/// <summary>
	/// If true, the pose of the sensor is valid and is ready to be queried.
	/// </summary>
	public bool GetPoseValid(int tracker = 0)
	{
		if (!OVRManager.isHmdPresent)
			return false;

		switch (tracker)
		{
			case 0:
				return OVRPlugin.GetNodePositionTracked(OVRPlugin.Node.TrackerZero);
			case 1:
				return OVRPlugin.GetNodePositionTracked(OVRPlugin.Node.TrackerOne);
			case 2:
				return OVRPlugin.GetNodePositionTracked(OVRPlugin.Node.TrackerTwo);
			case 3:
				return OVRPlugin.GetNodePositionTracked(OVRPlugin.Node.TrackerThree);
			default:
				return false;
		}
	}

	public bool GetPresent(int tracker = 0)
	{
		if (!OVRManager.isHmdPresent)
			return false;

		switch (tracker)
		{
			case 0:
				return OVRPlugin.GetNodePresent(OVRPlugin.Node.TrackerZero);
			case 1:
				return OVRPlugin.GetNodePresent(OVRPlugin.Node.TrackerOne);
			case 2:
				return OVRPlugin.GetNodePresent(OVRPlugin.Node.TrackerTwo);
			case 3:
				return OVRPlugin.GetNodePresent(OVRPlugin.Node.TrackerThree);
			default:
				return false;
		}
	}
}
