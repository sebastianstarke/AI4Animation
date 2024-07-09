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
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

public static class OVRKeyboard
{
	public struct TrackedKeyboardState
	{
		public bool isPositionValid;
		public bool isPositionTracked;
		public bool isOrientationValid;
		public bool isOrientationTracked;
		public Vector3 position;
		public Quaternion rotation;
		public double timeInSeconds;
	}

	public struct TrackedKeyboardInfo
	{
		public string Name;
		public UInt64 Identifier;
		public Vector3 Dimensions;
		public OVRPlugin.TrackedKeyboardFlags KeyboardFlags;
		public OVRPlugin.TrackedKeyboardPresentationStyles SupportedPresentationStyles;
	}

	public static TrackedKeyboardState GetKeyboardState()
	{
		TrackedKeyboardState keyboardState;

		OVRPlugin.KeyboardState keyboardStatePlugin;
		OVRPlugin.GetKeyboardState(OVRPlugin.Step.Render, out keyboardStatePlugin);
		keyboardState.timeInSeconds = keyboardStatePlugin.PoseState.Time;

		OVRPose nodePose = keyboardStatePlugin.PoseState.Pose.ToOVRPose();
		keyboardState.position = nodePose.position;
		keyboardState.rotation = nodePose.orientation;

		keyboardState.isPositionValid      = (keyboardStatePlugin.PositionValid      == OVRPlugin.Bool.True);
		keyboardState.isPositionTracked    = (keyboardStatePlugin.PositionTracked    == OVRPlugin.Bool.True);
		keyboardState.isOrientationValid   = (keyboardStatePlugin.OrientationValid   == OVRPlugin.Bool.True);
		keyboardState.isOrientationTracked = (keyboardStatePlugin.OrientationTracked == OVRPlugin.Bool.True);

		return keyboardState;
	}

	// Query for information about the system keyboards.
	public static bool GetSystemKeyboardInfo(OVRPlugin.TrackedKeyboardQueryFlags keyboardQueryFlags, out TrackedKeyboardInfo keyboardInfo)
	{
		keyboardInfo = default(TrackedKeyboardInfo);

		OVRPlugin.KeyboardDescription keyboardDescription;
		if(OVRPlugin.GetSystemKeyboardDescription(keyboardQueryFlags, out keyboardDescription))
		{
			keyboardInfo.Name = Encoding.UTF8.GetString(keyboardDescription.Name).TrimEnd('\0');
			keyboardInfo.Identifier = keyboardDescription.TrackedKeyboardId;
			keyboardInfo.Dimensions = new Vector3(keyboardDescription.Dimensions.x, keyboardDescription.Dimensions.y, keyboardDescription.Dimensions.z);
			keyboardInfo.KeyboardFlags = keyboardDescription.KeyboardFlags;
			keyboardInfo.SupportedPresentationStyles = keyboardDescription.SupportedPresentationStyles;
			return true;
		}
		else
		{
			return false;
		}
	}

	public static bool StopKeyboardTracking(TrackedKeyboardInfo keyboardInfo)
	{
		return OVRPlugin.StopKeyboardTracking();
	}
}
