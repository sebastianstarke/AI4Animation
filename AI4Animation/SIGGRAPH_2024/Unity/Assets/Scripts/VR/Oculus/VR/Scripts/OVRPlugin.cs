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

#if !(UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || (UNITY_ANDROID && !UNITY_EDITOR))
#define OVRPLUGIN_UNSUPPORTED_PLATFORM
#endif


#if UNITY_ANDROID && !UNITY_EDITOR
#define OVRPLUGIN_INCLUDE_MRC_ANDROID
#endif

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

// Internal C# wrapper for OVRPlugin.

public static partial class OVRPlugin
{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
	public const bool isSupportedPlatform = false;
#else
	public const bool isSupportedPlatform = true;
#endif

#if OVRPLUGIN_UNSUPPORTED_PLATFORM
	public static readonly System.Version wrapperVersion = _versionZero;
#else
	public static readonly System.Version wrapperVersion = OVRP_1_81_0.version;
#endif

#if !OVRPLUGIN_UNSUPPORTED_PLATFORM
	private static System.Version _version;
#endif
	public static System.Version version
	{
		get {
#if OVRPLUGIN_EDITOR_MOCK_ENABLED
			return wrapperVersion;
#elif OVRPLUGIN_UNSUPPORTED_PLATFORM
			Debug.LogWarning("Platform is not currently supported by OVRPlugin");
			return _versionZero;
#else
			if (_version == null)
			{
				try
				{
					string pluginVersion = OVRP_1_1_0.ovrp_GetVersion();

					if (pluginVersion != null)
					{
						// Truncate unsupported trailing version info for System.Version. Original string is returned if not present.
						pluginVersion = pluginVersion.Split('-')[0];
						_version = new System.Version(pluginVersion);
					}
					else
					{
						_version = _versionZero;
					}
				}
				catch
				{
					_version = _versionZero;
				}

				// Unity 5.1.1f3-p3 have OVRPlugin version "0.5.0", which isn't accurate.
				if (_version == OVRP_0_5_0.version)
					_version = OVRP_0_1_0.version;

				if (_version > _versionZero && _version < OVRP_1_3_0.version)
					throw new PlatformNotSupportedException("Oculus Utilities version " + wrapperVersion + " is too new for OVRPlugin version " + _version.ToString() + ". Update to the latest version of Unity.");
			}

			return _version;
#endif
		}
	}

#if !OVRPLUGIN_UNSUPPORTED_PLATFORM
	private static System.Version _nativeSDKVersion;
#endif
	public static System.Version nativeSDKVersion
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return _versionZero;
#else
			if (_nativeSDKVersion == null)
			{
				try
				{
					string sdkVersion = string.Empty;

					if (version >= OVRP_1_1_0.version)
						sdkVersion = OVRP_1_1_0.ovrp_GetNativeSDKVersion();
					else
						sdkVersion = _versionZero.ToString();

					if (sdkVersion != null)
					{
						// Truncate unsupported trailing version info for System.Version. Original string is returned if not present.
						sdkVersion = sdkVersion.Split('-')[0];
						_nativeSDKVersion = new System.Version(sdkVersion);
					}
					else
					{
						_nativeSDKVersion = _versionZero;
					}
				}
				catch
				{
					_nativeSDKVersion = _versionZero;
				}
			}

			return _nativeSDKVersion;
#endif
		}
	}

	[StructLayout(LayoutKind.Sequential)]
	private class GUID
	{
		public int a;
		public short b;
		public short c;
		public byte d0;
		public byte d1;
		public byte d2;
		public byte d3;
		public byte d4;
		public byte d5;
		public byte d6;
		public byte d7;
	}

	public enum Bool
	{
		False = 0,
		True
	}

	public enum Result
	{
		/// Success
		Success = 0,
		Success_EventUnavailable = 1,
		Success_Pending = 2,

		/// Failure
		Failure = -1000,
		Failure_InvalidParameter = -1001,
		Failure_NotInitialized = -1002,
		Failure_InvalidOperation = -1003,
		Failure_Unsupported = -1004,
		Failure_NotYetImplemented = -1005,
		Failure_OperationFailed = -1006,
		Failure_InsufficientSize = -1007,
		Failure_DataIsInvalid = -1008,
		Failure_DeprecatedOperation = -1009,
		Failure_ErrorLimitReached = -1010,
		Failure_ErrorInitializationFailed = -1011,

		/// Space error cases
		Failure_SpaceCloudStorageDisabled = -2000,
		Failure_SpaceMappingInsufficient = -2001,
		Failure_SpaceLocalizationFailed = -2002,
		Failure_SpaceNetworkTimeout = -2003,
		Failure_SpaceNetworkRequestFailed = -2004,
	}

	public static bool IsSuccess(this Result result) => result >= 0;

	public enum LogLevel
	{
		Debug = 0,
		Info = 1,
		Error = 2
	}

	public delegate void LogCallback2DelegateType(LogLevel logLevel, IntPtr message, int size);

	public static void SetLogCallback2(LogCallback2DelegateType logCallback)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			// do nothing
#else
		if (version >= OVRP_1_70_0.version)
		{
			Result result = OVRP_1_70_0.ovrp_SetLogCallback2(logCallback);
			if (result != Result.Success)
			{
				Debug.LogWarning("OVRPlugin.SetLogCallback2() failed");
			}
		}
#endif
	}


	public enum CameraStatus
	{
		CameraStatus_None,
		CameraStatus_Connected,
		CameraStatus_Calibrating,
		CameraStatus_CalibrationFailed,
		CameraStatus_Calibrated,
		CameraStatus_ThirdPerson,
		CameraStatus_EnumSize = 0x7fffffff
	}

	public enum CameraAnchorType
	{
		CameraAnchorType_PreDefined = 0,
		CameraAnchorType_Custom = 1,
		CameraAnchorType_Count,
		CameraAnchorType_EnumSize = 0x7fffffff
	}

	public enum XrApi
	{
		Unknown = 0,
		CAPI = 1,
		VRAPI = 2,
		OpenXR = 3,
		EnumSize = 0x7fffffff
	}

	public enum Eye
	{
		None = -1,
		Left = 0,
		Right = 1,
		Count = 2
	}


	public enum Tracker
	{
		None = -1,
		Zero = 0,
		One = 1,
		Two = 2,
		Three = 3,
		Count,
	}

	public enum Node
	{
		None = -1,
		EyeLeft = 0,
		EyeRight = 1,
		EyeCenter = 2,
		HandLeft = 3,
		HandRight = 4,
		TrackerZero = 5,
		TrackerOne = 6,
		TrackerTwo = 7,
		TrackerThree = 8,
		Head = 9,
		DeviceObjectZero = 10,
		TrackedKeyboard = 11,
		Count,
	}

	public enum Controller
	{
		None = 0,
		LTouch = 0x00000001,
		RTouch = 0x00000002,
		Touch = LTouch | RTouch,
		Remote = 0x00000004,
		Gamepad = 0x00000010,
		LHand = 0x00000020,
		RHand = 0x00000040,
		Hands = LHand | RHand,
		Active = unchecked((int)0x80000000),
		All = ~None,
	}

	public enum InteractionProfile
	{
		None = 0,
		Touch = 1,
		TouchPro = 2,
	}

	public enum Handedness
	{
		Unsupported = 0,
		LeftHanded = 1,
		RightHanded = 2,
	}

	public enum TrackingOrigin
	{
		EyeLevel = 0,
		FloorLevel = 1,
		Stage = 2,
		View = 4,
		Count,
	}

	public enum RecenterFlags
	{
		Default = 0,
		IgnoreAll = unchecked((int)0x80000000),
		Count,
	}

	public enum BatteryStatus
	{
		Charging = 0,
		Discharging,
		Full,
		NotCharging,
		Unknown,
	}

	public enum EyeTextureFormat
	{
		Default = 0,
		R8G8B8A8_sRGB = 0,
		R8G8B8A8 = 1,
		R16G16B16A16_FP = 2,
		R11G11B10_FP = 3,
		B8G8R8A8_sRGB = 4,
		B8G8R8A8 = 5,
		R5G6B5 = 11,
		EnumSize = 0x7fffffff
	}

	public enum PlatformUI
	{
		None = -1,
		ConfirmQuit = 1,
		GlobalMenuTutorial, // Deprecated
	}

	public enum SystemRegion
	{
		Unspecified = 0,
		Japan,
		China,
	}

	public enum SystemHeadset
	{
		None = 0,

		// Standalone headsets
		Oculus_Quest = 8,
		Oculus_Quest_2 = 9,
		Meta_Quest_Pro = 10,
		Placeholder_11,
		Placeholder_12,
		Placeholder_13,
		Placeholder_14,

		// PC headsets
		Rift_DK1 = 0x1000,
		Rift_DK2,
		Rift_CV1,
		Rift_CB,
		Rift_S,
		Oculus_Link_Quest,
		Oculus_Link_Quest_2,
		Meta_Link_Quest_Pro,
		PC_Placeholder_4104,
		PC_Placeholder_4105,
		PC_Placeholder_4106,
		PC_Placeholder_4107
	}

	public enum OverlayShape
	{
		Quad = 0,
		Cylinder = 1,
		Cubemap = 2,
		OffcenterCubemap = 4,
		Equirect = 5,
		ReconstructionPassthrough = 7,
		SurfaceProjectedPassthrough = 8,
		Fisheye = 9,
		KeyboardHandsPassthrough = 10,
		KeyboardMaskedHandsPassthrough = 11,
	}

	public static bool IsPassthroughShape(OverlayShape shape)
	{
		return shape == OverlayShape.ReconstructionPassthrough
		       || shape == OverlayShape.KeyboardHandsPassthrough
		       || shape == OverlayShape.KeyboardMaskedHandsPassthrough
		       || shape == OverlayShape.SurfaceProjectedPassthrough;
	}

	public enum Step
	{
		Render = -1,
		Physics = 0, // will be deprecated when using OpenXR
	}

	public enum CameraDevice
	{
		None = 0,
		WebCamera0 = 100,
		WebCamera1 = 101,
		ZEDCamera = 300,
	}

	public enum CameraDeviceDepthSensingMode
	{
		Standard = 0,
		Fill = 1,
	}

	public enum CameraDeviceDepthQuality
	{
		Low = 0,
		Medium = 1,
		High = 2,
	}

	public enum FoveatedRenderingLevel
	{
		Off = 0,
		Low = 1,
		Medium = 2,
		High = 3,
		HighTop = 4,
		EnumSize = 0x7FFFFFFF
	}

	[Obsolete("Please use FoveatedRenderingLevel instead", false)]
	public enum FixedFoveatedRenderingLevel
	{
		Off = 0,
		Low = FoveatedRenderingLevel.Low,
		Medium = FoveatedRenderingLevel.Medium,
		High = FoveatedRenderingLevel.High,
		// High foveation setting with more detail toward the bottom of the view and more foveation near the top (Same as High on Oculus Go)
		HighTop = FoveatedRenderingLevel.HighTop,
		EnumSize = 0x7FFFFFFF
	}

	[Obsolete("Please use FixedFoveatedRenderingLevel instead", false)]
	public enum TiledMultiResLevel
	{
		Off = 0,
		LMSLow = FixedFoveatedRenderingLevel.Low,
		LMSMedium = FixedFoveatedRenderingLevel.Medium,
		LMSHigh = FixedFoveatedRenderingLevel.High,
		// High foveation setting with more detail toward the bottom of the view and more foveation near the top (Same as High on Oculus Go)
		LMSHighTop = FixedFoveatedRenderingLevel.HighTop,
		EnumSize = 0x7FFFFFFF
	}

	public static int MAX_CPU_CORES = 8;
	public enum PerfMetrics
	{
		App_CpuTime_Float = 0,
		App_GpuTime_Float = 1,

		Compositor_CpuTime_Float = 3,
		Compositor_GpuTime_Float = 4,
		Compositor_DroppedFrameCount_Int = 5,

		System_GpuUtilPercentage_Float = 7,
		System_CpuUtilAveragePercentage_Float = 8,
		System_CpuUtilWorstPercentage_Float = 9,

		// Added 1.32.0
		Device_CpuClockFrequencyInMHz_Float = 10, // Deprecated 1.68.0
		Device_GpuClockFrequencyInMHz_Float = 11, // Deprecated 1.68.0
		Device_CpuClockLevel_Int = 12, // Deprecated 1.68.0
		Device_GpuClockLevel_Int = 13, // Deprecated 1.68.0

		Compositor_SpaceWarp_Mode_Int = 14,

		Device_CpuCore0UtilPercentage_Float = 32,
		Device_CpuCore1UtilPercentage_Float = Device_CpuCore0UtilPercentage_Float + 1,
		Device_CpuCore2UtilPercentage_Float = Device_CpuCore0UtilPercentage_Float + 2,
		Device_CpuCore3UtilPercentage_Float = Device_CpuCore0UtilPercentage_Float + 3,
		Device_CpuCore4UtilPercentage_Float = Device_CpuCore0UtilPercentage_Float + 4,
		Device_CpuCore5UtilPercentage_Float = Device_CpuCore0UtilPercentage_Float + 5,
		Device_CpuCore6UtilPercentage_Float = Device_CpuCore0UtilPercentage_Float + 6,
		Device_CpuCore7UtilPercentage_Float = Device_CpuCore0UtilPercentage_Float + 7,
		// Enum value 32~63 are reserved for CPU Cores' utilization (assuming at most 32 cores).

		Count,
		EnumSize = 0x7FFFFFFF
	}

	public enum ProcessorPerformanceLevel
	{
		PowerSavings = 0,
		SustainedLow = 1,
		SustainedHigh = 2,
		Boost = 3,
		EnumSize = 0x7FFFFFFF
	}

	public enum FeatureType
	{
		HandTracking = 0,
		KeyboardTracking = 1,
		EyeTracking = 2,
		FaceTracking = 3,
		BodyTracking = 4,
		Passthrough = 5,
		GazeBasedFoveatedRendering = 6,
		Count,
		EnumSize = 0x7FFFFFFF
	}


	[StructLayout(LayoutKind.Sequential)]
	public struct CameraDeviceIntrinsicsParameters
	{
		float fx; /* Focal length in pixels along x axis. */
		float fy; /* Focal length in pixels along y axis. */
		float cx; /* Optical center along x axis, defined in pixels (usually close to width/2). */
		float cy; /* Optical center along y axis, defined in pixels (usually close to height/2). */
		double disto0; /* Distortion factor : [ k1, k2, p1, p2, k3 ]. Radial (k1,k2,k3) and Tangential (p1,p2) distortion.*/
		double disto1;
		double disto2;
		double disto3;
		double disto4;
		float v_fov; /* Vertical field of view after stereo rectification, in degrees. */
		float h_fov; /* Horizontal field of view after stereo rectification, in degrees.*/
		float d_fov; /* Diagonal field of view after stereo rectification, in degrees.*/
		int w; /* Resolution width */
		int h; /* Resolution height */
	}

	private const int OverlayShapeFlagShift = 4;
	private enum OverlayFlag
	{
		None = unchecked((int)0x00000000),
		OnTop = unchecked((int)0x00000001),
		HeadLocked = unchecked((int)0x00000002),
		NoDepth = unchecked((int)0x00000004),
		ExpensiveSuperSample = unchecked((int)0x00000008),
		EfficientSuperSample = unchecked((int)0x00000010),
		EfficientSharpen = unchecked((int)0x00000020),
		BicubicFiltering = unchecked((int)0x00000040),
		ExpensiveSharpen = unchecked((int)0x00000080),
		SecureContent = unchecked((int)0x00000100),

		// Using the 5-8 bits for shapes, total 16 potential shapes can be supported 0x000000[0]0 ->  0x000000[F]0
		ShapeFlag_Quad = unchecked((int)OverlayShape.Quad << OverlayShapeFlagShift),
		ShapeFlag_Cylinder = unchecked((int)OverlayShape.Cylinder << OverlayShapeFlagShift),
		ShapeFlag_Cubemap = unchecked((int)OverlayShape.Cubemap << OverlayShapeFlagShift),
		ShapeFlag_OffcenterCubemap = unchecked((int)OverlayShape.OffcenterCubemap << OverlayShapeFlagShift),
		ShapeFlagRangeMask = unchecked((int)0xF << OverlayShapeFlagShift),

		Hidden = unchecked((int)0x000000200),
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Vector2f
	{
		public float x;
		public float y;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Vector3f
	{
		public float x;
		public float y;
		public float z;
		public static readonly Vector3f zero = new Vector3f { x = 0.0f, y = 0.0f, z = 0.0f };
		public override string ToString()
		{
			return string.Format(System.Globalization.CultureInfo.InvariantCulture, "{0}, {1}, {2}", x, y, z);
		}
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Vector4f
	{
		public float x;
		public float y;
		public float z;
		public float w;
		public static readonly Vector4f zero = new Vector4f { x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f };
		public override string ToString()
		{
			return string.Format(System.Globalization.CultureInfo.InvariantCulture, "{0}, {1}, {2}, {3}", x, y, z, w);
		}
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Vector4s
	{
		public short x;
		public short y;
		public short z;
		public short w;
		public static readonly Vector4s zero = new Vector4s { x = 0, y = 0, z = 0, w = 0 };
		public override string ToString()
		{
			return string.Format(System.Globalization.CultureInfo.InvariantCulture, "{0}, {1}, {2}, {3}", x, y, z, w);
		}
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Quatf
	{
		public float x;
		public float y;
		public float z;
		public float w;
		public static readonly Quatf identity = new Quatf { x = 0.0f, y = 0.0f, z = 0.0f, w = 1.0f };
		public override string ToString()
		{
			return string.Format(System.Globalization.CultureInfo.InvariantCulture, "{0}, {1}, {2}, {3}", x, y, z, w);
		}
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Posef
	{
		public Quatf Orientation;
		public Vector3f Position;
		public static readonly Posef identity = new Posef { Orientation = Quatf.identity, Position = Vector3f.zero };
		public override string ToString()
		{
			return string.Format(System.Globalization.CultureInfo.InvariantCulture, "Position ({0}), Orientation({1})", Position, Orientation);
		}
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct TextureRectMatrixf
	{
		public Rect leftRect;
		public Rect rightRect;
		public Vector4 leftScaleBias;
		public Vector4 rightScaleBias;
		public static readonly TextureRectMatrixf zero = new TextureRectMatrixf { leftRect = new Rect(0, 0, 1, 1), rightRect = new Rect(0, 0, 1, 1), leftScaleBias = new Vector4(1, 1, 0, 0), rightScaleBias = new Vector4(1, 1, 0, 0) };

		public override string ToString()
		{
			return string.Format(System.Globalization.CultureInfo.InvariantCulture, "Rect Left ({0}), Rect Right({1}), Scale Bias Left ({2}), Scale Bias Right({3})", leftRect, rightRect, leftScaleBias, rightScaleBias);
		}
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct PoseStatef
	{
		public Posef Pose;
		public Vector3f Velocity;
		public Vector3f Acceleration;
		public Vector3f AngularVelocity;
		public Vector3f AngularAcceleration;
		public double Time;

		public static readonly PoseStatef identity = new PoseStatef
		{
			Pose = Posef.identity,
			Velocity = Vector3f.zero,
			Acceleration = Vector3f.zero,
			AngularVelocity = Vector3f.zero,
			AngularAcceleration = Vector3f.zero
		};
	}

	public enum HapticsLocation
	{
		None = 0x00,
		Hand = 0x01,
		Thumb = 0x02,
		Index = 0x04,
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct ControllerState5
	{
		public uint ConnectedControllers;
		public uint Buttons;
		public uint Touches;
		public uint NearTouches;
		public float LIndexTrigger;
		public float RIndexTrigger;
		public float LHandTrigger;
		public float RHandTrigger;
		public Vector2f LThumbstick;
		public Vector2f RThumbstick;
		public Vector2f LTouchpad;
		public Vector2f RTouchpad;
		public byte LBatteryPercentRemaining;
		public byte RBatteryPercentRemaining;
		public byte LRecenterCount;
		public byte RRecenterCount;
		public float LThumbRestForce;
		public float RThumbRestForce;
		public float LStylusForce;
		public float RStylusForce;
		public float LIndexTriggerCurl;
		public float RIndexTriggerCurl;
		public float LIndexTriggerSlide;
		public float RIndexTriggerSlide;

		public ControllerState5(ControllerState4 cs)
		{
			ConnectedControllers = cs.ConnectedControllers;
			Buttons = cs.Buttons;
			Touches = cs.Touches;
			NearTouches = cs.NearTouches;
			LIndexTrigger = cs.LIndexTrigger;
			RIndexTrigger = cs.RIndexTrigger;
			LHandTrigger = cs.LHandTrigger;
			RHandTrigger = cs.RHandTrigger;
			LThumbstick = cs.LThumbstick;
			RThumbstick = cs.RThumbstick;
			LTouchpad = cs.LTouchpad;
			RTouchpad = cs.RTouchpad;
			LBatteryPercentRemaining = cs.LBatteryPercentRemaining;
			RBatteryPercentRemaining = cs.RBatteryPercentRemaining;
			LRecenterCount = cs.LRecenterCount;
			RRecenterCount = cs.RRecenterCount;
			LThumbRestForce = 0.0f;
			RThumbRestForce = 0.0f;
			LStylusForce = 0.0f;
			RStylusForce = 0.0f;
			LIndexTriggerCurl = 0.0f;
			RIndexTriggerCurl = 0.0f;
			LIndexTriggerSlide = 0.0f;
			RIndexTriggerSlide = 0.0f;
		}
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct ControllerState4
	{
		public uint ConnectedControllers;
		public uint Buttons;
		public uint Touches;
		public uint NearTouches;
		public float LIndexTrigger;
		public float RIndexTrigger;
		public float LHandTrigger;
		public float RHandTrigger;
		public Vector2f LThumbstick;
		public Vector2f RThumbstick;
		public Vector2f LTouchpad;
		public Vector2f RTouchpad;
		public byte LBatteryPercentRemaining;
		public byte RBatteryPercentRemaining;
		public byte LRecenterCount;
		public byte RRecenterCount;
		public byte Reserved_27;
		public byte Reserved_26;
		public byte Reserved_25;
		public byte Reserved_24;
		public byte Reserved_23;
		public byte Reserved_22;
		public byte Reserved_21;
		public byte Reserved_20;
		public byte Reserved_19;
		public byte Reserved_18;
		public byte Reserved_17;
		public byte Reserved_16;
		public byte Reserved_15;
		public byte Reserved_14;
		public byte Reserved_13;
		public byte Reserved_12;
		public byte Reserved_11;
		public byte Reserved_10;
		public byte Reserved_09;
		public byte Reserved_08;
		public byte Reserved_07;
		public byte Reserved_06;
		public byte Reserved_05;
		public byte Reserved_04;
		public byte Reserved_03;
		public byte Reserved_02;
		public byte Reserved_01;
		public byte Reserved_00;

		public ControllerState4(ControllerState2 cs)
		{
			ConnectedControllers = cs.ConnectedControllers;
			Buttons = cs.Buttons;
			Touches = cs.Touches;
			NearTouches = cs.NearTouches;
			LIndexTrigger = cs.LIndexTrigger;
			RIndexTrigger = cs.RIndexTrigger;
			LHandTrigger = cs.LHandTrigger;
			RHandTrigger = cs.RHandTrigger;
			LThumbstick = cs.LThumbstick;
			RThumbstick = cs.RThumbstick;
			LTouchpad = cs.LTouchpad;
			RTouchpad = cs.RTouchpad;
			LBatteryPercentRemaining = 0;
			RBatteryPercentRemaining = 0;
			LRecenterCount = 0;
			RRecenterCount = 0;
			Reserved_27 = 0;
			Reserved_26 = 0;
			Reserved_25 = 0;
			Reserved_24 = 0;
			Reserved_23 = 0;
			Reserved_22 = 0;
			Reserved_21 = 0;
			Reserved_20 = 0;
			Reserved_19 = 0;
			Reserved_18 = 0;
			Reserved_17 = 0;
			Reserved_16 = 0;
			Reserved_15 = 0;
			Reserved_14 = 0;
			Reserved_13 = 0;
			Reserved_12 = 0;
			Reserved_11 = 0;
			Reserved_10 = 0;
			Reserved_09 = 0;
			Reserved_08 = 0;
			Reserved_07 = 0;
			Reserved_06 = 0;
			Reserved_05 = 0;
			Reserved_04 = 0;
			Reserved_03 = 0;
			Reserved_02 = 0;
			Reserved_01 = 0;
			Reserved_00 = 0;
		}
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct ControllerState2
	{
		public uint ConnectedControllers;
		public uint Buttons;
		public uint Touches;
		public uint NearTouches;
		public float LIndexTrigger;
		public float RIndexTrigger;
		public float LHandTrigger;
		public float RHandTrigger;
		public Vector2f LThumbstick;
		public Vector2f RThumbstick;
		public Vector2f LTouchpad;
		public Vector2f RTouchpad;

		public ControllerState2(ControllerState cs)
		{
			ConnectedControllers = cs.ConnectedControllers;
			Buttons = cs.Buttons;
			Touches = cs.Touches;
			NearTouches = cs.NearTouches;
			LIndexTrigger = cs.LIndexTrigger;
			RIndexTrigger = cs.RIndexTrigger;
			LHandTrigger = cs.LHandTrigger;
			RHandTrigger = cs.RHandTrigger;
			LThumbstick = cs.LThumbstick;
			RThumbstick = cs.RThumbstick;
			LTouchpad = new Vector2f() { x = 0.0f, y = 0.0f };
			RTouchpad = new Vector2f() { x = 0.0f, y = 0.0f };
		}
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct ControllerState
	{
		public uint ConnectedControllers;
		public uint Buttons;
		public uint Touches;
		public uint NearTouches;
		public float LIndexTrigger;
		public float RIndexTrigger;
		public float LHandTrigger;
		public float RHandTrigger;
		public Vector2f LThumbstick;
		public Vector2f RThumbstick;
	}


	[StructLayout(LayoutKind.Sequential)]
	public struct HapticsBuffer
	{
		public IntPtr Samples;
		public int SamplesCount;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct HapticsState
	{
		public int SamplesAvailable;
		public int SamplesQueued;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct HapticsDesc
	{
		public int SampleRateHz;
		public int SampleSizeInBytes;
		public int MinimumSafeSamplesQueued;
		public int MinimumBufferSamplesCount;
		public int OptimalBufferSamplesCount;
		public int MaximumBufferSamplesCount;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct HapticsAmplitudeEnvelopeVibration
	{
		public float Duration;
		public UInt32 AmplitudeCount;
		public IntPtr Amplitudes;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct HapticsPcmVibration
	{
		public UInt32 BufferSize;
		public IntPtr Buffer;
		public float SampleRateHz;
		public Bool Append;
		public IntPtr SamplesConsumed;
	}

	public enum HapticsConstants
	{
		MaxSamples = 4000,
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct AppPerfFrameStats
	{
		public int HmdVsyncIndex;
		public int AppFrameIndex;
		public int AppDroppedFrameCount;
		public float AppMotionToPhotonLatency;
		public float AppQueueAheadTime;
		public float AppCpuElapsedTime;
		public float AppGpuElapsedTime;
		public int CompositorFrameIndex;
		public int CompositorDroppedFrameCount;
		public float CompositorLatency;
		public float CompositorCpuElapsedTime;
		public float CompositorGpuElapsedTime;
		public float CompositorCpuStartToGpuEndElapsedTime;
		public float CompositorGpuEndToVsyncElapsedTime;
	}

	public const int AppPerfFrameStatsMaxCount = 5;

	[StructLayout(LayoutKind.Sequential)]
	public struct AppPerfStats
	{
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = AppPerfFrameStatsMaxCount)]
		public AppPerfFrameStats[] FrameStats;
		public int FrameStatsCount;
		public Bool AnyFrameStatsDropped;
		public float AdaptiveGpuPerformanceScale;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Sizei : IEquatable<Sizei>
	{
		public int w;
		public int h;

		public static readonly Sizei zero = new Sizei { w = 0, h = 0 };

		public bool Equals(Sizei other)
		{
			return w == other.w && h == other.h;
		}

		public override bool Equals(object obj)
		{
			return obj is Sizei other && Equals(other);
		}

		public override int GetHashCode()
		{
			unchecked
			{
				return (w * 397) ^ h;
			}
		}
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Sizef
	{
		public float w;
		public float h;

		public static readonly Sizef zero = new Sizef { w = 0, h = 0 };
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Size3f
	{
		public float w;
		public float h;
		public float d;

		public static readonly Size3f zero = new Size3f { w = 0, h = 0, d = 0 };
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Vector2i
	{
		public int x;
		public int y;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Recti {
		public Vector2i Pos;
		public Sizei Size;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Rectf {
		public Vector2f Pos;
		public Sizef Size;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Boundsf {
		public Vector3f Pos;
		public Size3f Size;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Frustumf
	{
		public float zNear;
		public float zFar;
		public float fovX;
		public float fovY;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Frustumf2
	{
		public float zNear;
		public float zFar;
		public Fovf Fov;
	}

	public enum BoundaryType
	{
		[System.Obsolete("Deprecated. This enum value will not be supported in OpenXR", false)]
		OuterBoundary = 0x0001,
		PlayArea = 0x0100,
	}

	[System.Obsolete("Deprecated. This struct will not be supported in OpenXR", false)]
	[StructLayout(LayoutKind.Sequential)]
	public struct BoundaryTestResult
	{
		public Bool IsTriggering;
		public float ClosestDistance;
		public Vector3f ClosestPoint;
		public Vector3f ClosestPointNormal;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct BoundaryGeometry
	{
		public BoundaryType BoundaryType;
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
		public Vector3f[] Points;
		public int PointsCount;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Colorf
	{
		public float r;
		public float g;
		public float b;
		public float a;

		public override string ToString()
		{
			return string.Format(System.Globalization.CultureInfo.InvariantCulture,
				"R:{0:F3} G:{1:F3} B:{2:F3} A:{3:F3}", r, g, b, a);
		}
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Fovf
	{
		public float UpTan;
		public float DownTan;
		public float LeftTan;
		public float RightTan;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct CameraIntrinsics
	{
		public Bool IsValid;
		public double LastChangedTimeSeconds;
		public Fovf FOVPort;
		public float VirtualNearPlaneDistanceMeters;
		public float VirtualFarPlaneDistanceMeters;
		public Sizei ImageSensorPixelResolution;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct CameraExtrinsics
	{
		public Bool IsValid;
		public double LastChangedTimeSeconds;
		public CameraStatus CameraStatusData;
		public Node AttachedToNode;
		public Posef RelativePose;
	}

	public enum LayerLayout
	{
		Stereo = 0,
		Mono = 1,
		DoubleWide = 2,
		Array = 3,
		EnumSize = 0xF
	}

	public enum LayerFlags
	{
		Static = (1 << 0),
		LoadingScreen = (1 << 1),
		SymmetricFov = (1 << 2),
		TextureOriginAtBottomLeft = (1 << 3),
		ChromaticAberrationCorrection = (1 << 4),
		NoAllocation = (1 << 5),
		ProtectedContent = (1 << 6),
		AndroidSurfaceSwapChain = (1 << 7),
		BicubicFiltering = (1 << 14),
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct LayerDesc
	{
		public OverlayShape Shape;
		public LayerLayout Layout;
		public Sizei TextureSize;
		public int MipLevels;
		public int SampleCount;
		public EyeTextureFormat Format;
		public int LayerFlags;

		//Eye FOV-only members.
		public Fovf[] Fov;
		public Rectf[] VisibleRect;
		public Sizei MaxViewportSize;
		public EyeTextureFormat DepthFormat;

		public EyeTextureFormat MotionVectorFormat;
		public EyeTextureFormat MotionVectorDepthFormat;
		public Sizei MotionVectorTextureSize;

		public override string ToString()
		{
			string delim = ", ";
			return Shape.ToString()
			       + delim + Layout.ToString()
			       + delim + TextureSize.w.ToString() + "x" + TextureSize.h.ToString()
			       + delim + MipLevels.ToString()
			       + delim + SampleCount.ToString()
			       + delim + Format.ToString()
			       + delim + LayerFlags.ToString();
		}
	}

	[StructLayout(LayoutKind.Sequential)]
	private struct LayerDescInternal
	{
		public OverlayShape Shape;
		public LayerLayout Layout;
		public Sizei TextureSize;
		public int MipLevels;
		public int SampleCount;
		public EyeTextureFormat Format;
		public int LayerFlags;

		public Fovf Fov0;
		public Fovf Fov1;
		public Rectf VisibleRect0;
		public Rectf VisibleRect1;
		public Sizei MaxViewportSize;
		public EyeTextureFormat DepthFormat;

		public EyeTextureFormat MotionVectorFormat;
		public EyeTextureFormat MotionVectorDepthFormat;
		public Sizei MotionVectorTextureSize;

		public LayerDescInternal(LayerDesc layerDesc)
		{
			Shape = layerDesc.Shape;
			Layout = layerDesc.Layout;
			TextureSize = layerDesc.TextureSize;
			MipLevels = layerDesc.MipLevels;
			SampleCount = layerDesc.SampleCount;
			Format = layerDesc.Format;
			LayerFlags = layerDesc.LayerFlags;

			Fov0 = layerDesc.Fov[0];
			Fov1 = layerDesc.Fov[1];
			VisibleRect0 = layerDesc.VisibleRect[0];
			VisibleRect1 = layerDesc.VisibleRect[1];

			MaxViewportSize = layerDesc.MaxViewportSize;
			DepthFormat = layerDesc.DepthFormat;
			MotionVectorFormat = layerDesc.MotionVectorFormat;
			MotionVectorDepthFormat = layerDesc.MotionVectorDepthFormat;
			MotionVectorTextureSize = layerDesc.MotionVectorTextureSize;
		}

		public LayerDesc ToLayerDesc()
		{
			LayerDesc layerDesc = new LayerDesc();
			layerDesc.Shape = Shape;
			layerDesc.Layout = Layout;
			layerDesc.TextureSize = TextureSize;
			layerDesc.MipLevels = MipLevels;
			layerDesc.SampleCount = SampleCount;
			layerDesc.Format = Format;
			layerDesc.LayerFlags = LayerFlags;

			Array.Resize(ref layerDesc.Fov, 2);
			layerDesc.Fov[0] = Fov0;
			layerDesc.Fov[1] = Fov1;


			Array.Resize(ref layerDesc.VisibleRect, 2);
			layerDesc.VisibleRect[0] = VisibleRect0;
			layerDesc.VisibleRect[1] = VisibleRect1;

			layerDesc.MaxViewportSize = MaxViewportSize;
			layerDesc.DepthFormat = DepthFormat;
			layerDesc.MotionVectorFormat = MotionVectorFormat;
			layerDesc.MotionVectorDepthFormat = MotionVectorDepthFormat;
			layerDesc.MotionVectorTextureSize = MotionVectorTextureSize;

			return layerDesc;
		}
	}


	public enum BlendFactor {
		Zero = 0,
		One = 1,
		SrcAlpha = 2,
		OneMinusSrcAlpha = 3,
		DstAlpha = 4,
		OneMinusDstAlpha = 5
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct LayerSubmit
	{
		int LayerId;
		int TextureStage;
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
		Recti[] ViewportRect;
		Posef Pose;
		int LayerSubmitFlags;
	}

	public enum TrackingConfidence
	{
		Low = 0,
		High = 0x3f800000,
	}

	public enum Hand
	{
		None = -1,
		HandLeft = 0,
		HandRight = 1,
	}

	[Flags]
	public enum HandStatus
	{
		HandTracked = (1 << 0), // if this is set the hand pose and bone rotations data is usable
		InputStateValid = (1 << 1), // if this is set the pointer pose and pinch data is usable
		SystemGestureInProgress = (1 << 6), // if this is set the hand is currently processing a system gesture
		DominantHand = (1 << 7), // if this is set the hand is currently the dominant hand
		MenuPressed = (1 << 8) // if this is set the hand performed a menu press
	}

	public enum BoneId
	{
		Invalid                 = -1,

		// hand bones
		Hand_Start              = 0,
		Hand_WristRoot          = Hand_Start + 0, // root frame of the hand, where the wrist is located
		Hand_ForearmStub        = Hand_Start + 1, // frame for user's forearm
		Hand_Thumb0             = Hand_Start + 2, // thumb trapezium bone
		Hand_Thumb1             = Hand_Start + 3, // thumb metacarpal bone
		Hand_Thumb2             = Hand_Start + 4, // thumb proximal phalange bone
		Hand_Thumb3             = Hand_Start + 5, // thumb distal phalange bone
		Hand_Index1             = Hand_Start + 6, // index proximal phalange bone
		Hand_Index2             = Hand_Start + 7, // index intermediate phalange bone
		Hand_Index3             = Hand_Start + 8, // index distal phalange bone
		Hand_Middle1            = Hand_Start + 9, // middle proximal phalange bone
		Hand_Middle2            = Hand_Start + 10, // middle intermediate phalange bone
		Hand_Middle3            = Hand_Start + 11, // middle distal phalange bone
		Hand_Ring1              = Hand_Start + 12, // ring proximal phalange bone
		Hand_Ring2              = Hand_Start + 13, // ring intermediate phalange bone
		Hand_Ring3              = Hand_Start + 14, // ring distal phalange bone
		Hand_Pinky0             = Hand_Start + 15, // pinky metacarpal bone
		Hand_Pinky1             = Hand_Start + 16, // pinky proximal phalange bone
		Hand_Pinky2             = Hand_Start + 17, // pinky intermediate phalange bone
		Hand_Pinky3             = Hand_Start + 18, // pinky distal phalange bone
		Hand_MaxSkinnable       = Hand_Start + 19,
		// Bone tips are position only. They are not used for skinning but are useful for hit-testing.
		// NOTE: Hand_ThumbTip == Hand_MaxSkinnable since the extended tips need to be contiguous
		Hand_ThumbTip           = Hand_MaxSkinnable + 0, // tip of the thumb
		Hand_IndexTip           = Hand_MaxSkinnable + 1, // tip of the index finger
		Hand_MiddleTip          = Hand_MaxSkinnable + 2, // tip of the middle finger
		Hand_RingTip            = Hand_MaxSkinnable + 3, // tip of the ring finger
		Hand_PinkyTip           = Hand_MaxSkinnable + 4, // tip of the pinky
		Hand_End                = Hand_MaxSkinnable + 5,

		// body bones
		Body_Start                       = 0,
		Body_Root                        = Body_Start + 0,
		Body_Hips                        = Body_Start + 1,
		Body_SpineLower                  = Body_Start + 2,
		Body_SpineMiddle                 = Body_Start + 3,
		Body_SpineUpper                  = Body_Start + 4,
		Body_Chest                       = Body_Start + 5,
		Body_Neck                        = Body_Start + 6,
		Body_Head                        = Body_Start + 7,
		Body_LeftShoulder                = Body_Start + 8,
		Body_LeftScapula                 = Body_Start + 9,
		Body_LeftArmUpper                = Body_Start + 10,
		Body_LeftArmLower                = Body_Start + 11,
		Body_LeftHandWristTwist          = Body_Start + 12,
		Body_RightShoulder               = Body_Start + 13,
		Body_RightScapula                = Body_Start + 14,
		Body_RightArmUpper               = Body_Start + 15,
		Body_RightArmLower               = Body_Start + 16,
		Body_RightHandWristTwist         = Body_Start + 17,
		Body_LeftHandPalm                = Body_Start + 18,
		Body_LeftHandWrist               = Body_Start + 19,
		Body_LeftHandThumbMetacarpal     = Body_Start + 20,
		Body_LeftHandThumbProximal       = Body_Start + 21,
		Body_LeftHandThumbDistal         = Body_Start + 22,
		Body_LeftHandThumbTip            = Body_Start + 23,
		Body_LeftHandIndexMetacarpal     = Body_Start + 24,
		Body_LeftHandIndexProximal       = Body_Start + 25,
		Body_LeftHandIndexIntermediate   = Body_Start + 26,
		Body_LeftHandIndexDistal         = Body_Start + 27,
		Body_LeftHandIndexTip            = Body_Start + 28,
		Body_LeftHandMiddleMetacarpal    = Body_Start + 29,
		Body_LeftHandMiddleProximal      = Body_Start + 30,
		Body_LeftHandMiddleIntermediate  = Body_Start + 31,
		Body_LeftHandMiddleDistal        = Body_Start + 32,
		Body_LeftHandMiddleTip           = Body_Start + 33,
		Body_LeftHandRingMetacarpal      = Body_Start + 34,
		Body_LeftHandRingProximal        = Body_Start + 35,
		Body_LeftHandRingIntermediate    = Body_Start + 36,
		Body_LeftHandRingDistal          = Body_Start + 37,
		Body_LeftHandRingTip             = Body_Start + 38,
		Body_LeftHandLittleMetacarpal    = Body_Start + 39,
		Body_LeftHandLittleProximal      = Body_Start + 40,
		Body_LeftHandLittleIntermediate  = Body_Start + 41,
		Body_LeftHandLittleDistal        = Body_Start + 42,
		Body_LeftHandLittleTip           = Body_Start + 43,
		Body_RightHandPalm               = Body_Start + 44,
		Body_RightHandWrist              = Body_Start + 45,
		Body_RightHandThumbMetacarpal    = Body_Start + 46,
		Body_RightHandThumbProximal      = Body_Start + 47,
		Body_RightHandThumbDistal        = Body_Start + 48,
		Body_RightHandThumbTip           = Body_Start + 49,
		Body_RightHandIndexMetacarpal    = Body_Start + 50,
		Body_RightHandIndexProximal      = Body_Start + 51,
		Body_RightHandIndexIntermediate  = Body_Start + 52,
		Body_RightHandIndexDistal        = Body_Start + 53,
		Body_RightHandIndexTip           = Body_Start + 54,
		Body_RightHandMiddleMetacarpal   = Body_Start + 55,
		Body_RightHandMiddleProximal     = Body_Start + 56,
		Body_RightHandMiddleIntermediate = Body_Start + 57,
		Body_RightHandMiddleDistal       = Body_Start + 58,
		Body_RightHandMiddleTip          = Body_Start + 59,
		Body_RightHandRingMetacarpal     = Body_Start + 60,
		Body_RightHandRingProximal       = Body_Start + 61,
		Body_RightHandRingIntermediate   = Body_Start + 62,
		Body_RightHandRingDistal         = Body_Start + 63,
		Body_RightHandRingTip            = Body_Start + 64,
		Body_RightHandLittleMetacarpal   = Body_Start + 65,
		Body_RightHandLittleProximal     = Body_Start + 66,
		Body_RightHandLittleIntermediate = Body_Start + 67,
		Body_RightHandLittleDistal       = Body_Start + 68,
		Body_RightHandLittleTip          = Body_Start + 69,
		Body_End                         = Body_Start + 70,

		// add new bones here
		Max = ((int)Hand_End > (int)Body_End) ? (int)Hand_End : (int)Body_End,
	}

	public enum HandFinger
	{
		Thumb = 0,
		Index = 1,
		Middle = 2,
		Ring = 3,
		Pinky = 4,
		Max = 5,
	}

	[Flags]
	public enum HandFingerPinch
	{
		Thumb  = (1 << HandFinger.Thumb),
		Index  = (1 << HandFinger.Index),
		Middle = (1 << HandFinger.Middle),
		Ring   = (1 << HandFinger.Ring),
		Pinky  = (1 << HandFinger.Pinky),
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct HandState
	{
		public HandStatus Status;
		public Posef RootPose;
		public Quatf[] BoneRotations;
		public HandFingerPinch Pinches;
		public float[] PinchStrength;
		public Posef PointerPose;
		public float HandScale;
		public TrackingConfidence HandConfidence;
		public TrackingConfidence[] FingerConfidences;
		public double RequestedTimeStamp;
		public double SampleTimeStamp;
	}


	[StructLayout(LayoutKind.Sequential)]
	private struct HandStateInternal
	{
		public HandStatus Status;
		public Posef RootPose;
		public Quatf BoneRotations_0;
		public Quatf BoneRotations_1;
		public Quatf BoneRotations_2;
		public Quatf BoneRotations_3;
		public Quatf BoneRotations_4;
		public Quatf BoneRotations_5;
		public Quatf BoneRotations_6;
		public Quatf BoneRotations_7;
		public Quatf BoneRotations_8;
		public Quatf BoneRotations_9;
		public Quatf BoneRotations_10;
		public Quatf BoneRotations_11;
		public Quatf BoneRotations_12;
		public Quatf BoneRotations_13;
		public Quatf BoneRotations_14;
		public Quatf BoneRotations_15;
		public Quatf BoneRotations_16;
		public Quatf BoneRotations_17;
		public Quatf BoneRotations_18;
		public Quatf BoneRotations_19;
		public Quatf BoneRotations_20;
		public Quatf BoneRotations_21;
		public Quatf BoneRotations_22;
		public Quatf BoneRotations_23;
		public HandFingerPinch Pinches;
		public float PinchStrength_0;
		public float PinchStrength_1;
		public float PinchStrength_2;
		public float PinchStrength_3;
		public float PinchStrength_4;
		public Posef PointerPose;
		public float HandScale;
		public TrackingConfidence HandConfidence;
		public TrackingConfidence FingerConfidences_0;
		public TrackingConfidence FingerConfidences_1;
		public TrackingConfidence FingerConfidences_2;
		public TrackingConfidence FingerConfidences_3;
		public TrackingConfidence FingerConfidences_4;
		public double RequestedTimeStamp;
		public double SampleTimeStamp;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct BoneCapsule
	{
		public short BoneIndex;
		public Vector3f StartPoint;
		public Vector3f EndPoint;
		public float Radius;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Bone
	{
		public BoneId Id;
		public short ParentBoneIndex;
		public Posef Pose;
	}

	public enum SkeletonConstants
	{
		MaxHandBones = BoneId.Hand_End,
		MaxBodyBones = BoneId.Body_End,
		MaxBones = BoneId.Max,
		MaxBoneCapsules = 19,
	}

	public enum SkeletonType
	{
		None = -1,
		HandLeft = 0,
		HandRight = 1,
		Body = 2,
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Skeleton
	{
		public SkeletonType Type;
		public uint NumBones;
		public uint NumBoneCapsules;
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = (int)SkeletonConstants.MaxHandBones)]
		public Bone[] Bones;
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = (int)SkeletonConstants.MaxBoneCapsules)]
		public BoneCapsule[] BoneCapsules;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct Skeleton2
	{
		public SkeletonType Type;
		public uint NumBones;
		public uint NumBoneCapsules;
		public Bone[] Bones;
		public BoneCapsule[] BoneCapsules;
	}

	[StructLayout(LayoutKind.Sequential)]
	private struct Skeleton2Internal
	{
		public SkeletonType Type;
		public uint NumBones;
		public uint NumBoneCapsules;
		public Bone Bones_0;
		public Bone Bones_1;
		public Bone Bones_2;
		public Bone Bones_3;
		public Bone Bones_4;
		public Bone Bones_5;
		public Bone Bones_6;
		public Bone Bones_7;
		public Bone Bones_8;
		public Bone Bones_9;
		public Bone Bones_10;
		public Bone Bones_11;
		public Bone Bones_12;
		public Bone Bones_13;
		public Bone Bones_14;
		public Bone Bones_15;
		public Bone Bones_16;
		public Bone Bones_17;
		public Bone Bones_18;
		public Bone Bones_19;
		public Bone Bones_20;
		public Bone Bones_21;
		public Bone Bones_22;
		public Bone Bones_23;
		public Bone Bones_24;
		public Bone Bones_25;
		public Bone Bones_26;
		public Bone Bones_27;
		public Bone Bones_28;
		public Bone Bones_29;
		public Bone Bones_30;
		public Bone Bones_31;
		public Bone Bones_32;
		public Bone Bones_33;
		public Bone Bones_34;
		public Bone Bones_35;
		public Bone Bones_36;
		public Bone Bones_37;
		public Bone Bones_38;
		public Bone Bones_39;
		public Bone Bones_40;
		public Bone Bones_41;
		public Bone Bones_42;
		public Bone Bones_43;
		public Bone Bones_44;
		public Bone Bones_45;
		public Bone Bones_46;
		public Bone Bones_47;
		public Bone Bones_48;
		public Bone Bones_49;
		public Bone Bones_50;
		public Bone Bones_51;
		public Bone Bones_52;
		public Bone Bones_53;
		public Bone Bones_54;
		public Bone Bones_55;
		public Bone Bones_56;
		public Bone Bones_57;
		public Bone Bones_58;
		public Bone Bones_59;
		public Bone Bones_60;
		public Bone Bones_61;
		public Bone Bones_62;
		public Bone Bones_63;
		public Bone Bones_64;
		public Bone Bones_65;
		public Bone Bones_66;
		public Bone Bones_67;
		public Bone Bones_68;
		public Bone Bones_69;
		public BoneCapsule BoneCapsules_0;
		public BoneCapsule BoneCapsules_1;
		public BoneCapsule BoneCapsules_2;
		public BoneCapsule BoneCapsules_3;
		public BoneCapsule BoneCapsules_4;
		public BoneCapsule BoneCapsules_5;
		public BoneCapsule BoneCapsules_6;
		public BoneCapsule BoneCapsules_7;
		public BoneCapsule BoneCapsules_8;
		public BoneCapsule BoneCapsules_9;
		public BoneCapsule BoneCapsules_10;
		public BoneCapsule BoneCapsules_11;
		public BoneCapsule BoneCapsules_12;
		public BoneCapsule BoneCapsules_13;
		public BoneCapsule BoneCapsules_14;
		public BoneCapsule BoneCapsules_15;
		public BoneCapsule BoneCapsules_16;
		public BoneCapsule BoneCapsules_17;
		public BoneCapsule BoneCapsules_18;
	}

	public enum MeshConstants
	{
		MaxVertices = 3000,
		MaxIndices = MaxVertices * 6,
	}

	public enum MeshType
	{
		None = -1,
		HandLeft = 0,
		HandRight = 1,
	}

	[StructLayout(LayoutKind.Sequential)]
	public class Mesh
	{
		public MeshType Type;
		public uint NumVertices;
		public uint NumIndices;

		[MarshalAs(UnmanagedType.ByValArray, SizeConst = (int)MeshConstants.MaxVertices)]
		public Vector3f[] VertexPositions;
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = (int)MeshConstants.MaxIndices)]
		public short[] Indices;
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = (int)MeshConstants.MaxVertices)]
		public Vector3f[] VertexNormals;
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = (int)MeshConstants.MaxVertices)]
		public Vector2f[] VertexUV0;
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = (int)MeshConstants.MaxVertices)]
		public Vector4s[] BlendIndices;
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = (int)MeshConstants.MaxVertices)]
		public Vector4f[] BlendWeights;
	}

	/// <summary>
	/// Space location flags
	/// </summary>
	/// <remarks>
	/// See the [OpenXR spc](https://www.khronos.org/registry/OpenXR/specs/1.0/man/html/XrSpaceLocationFlags.html) for
	/// more information.
	/// </remarks>
	[Flags]
	public enum SpaceLocationFlags : ulong
	{
		/// <summary>
		/// Indicates that the pose field's orientation field contains valid data.
		/// </summary>
		/// <remarks>
		/// Applications must not read a pose field's orientation if this flag is unset.
		/// </remarks>
		OrientationValid = 0x00000001,

		/// <summary>
		/// Indicates that the pose field's position field contains valid data.
		/// </summary>
		/// <remarks>
		/// Applications must not read a pose field's position if this flag is unset.
		/// </remarks>
		PositionValid = 0x00000002,

		/// <summary>
		/// Indicates that a pose field's orientation field represents an actively tracked orientation.
		/// </summary>
		/// <remarks>
		/// When a space location tracking an object whose orientation is no longer known during tracking loss
		/// (e.g. an observed QR code), the orientation will be a valid but untracked orientation and will be
		/// meaningful to use.
		/// </remarks>
		OrientationTracked = 0x00000004,

		/// <summary>
		/// Indicates that a pose field's position field represents an actively tracked position.
		/// </summary>
		/// <remarks>
		/// When a space location loses tracking, the position will be a valid but untracked value that is inferred or
		/// last-known, e.g. based on neck model updates, inertial dead reckoning, or a last-known position, and will be
		/// meaningful to use.
		/// </remarks>
		PositionTracked = 0x00000008,
	}

	public static bool IsPositionValid(this SpaceLocationFlags value) =>
		(value & SpaceLocationFlags.PositionValid) != 0;

	public static bool IsOrientationValid(this SpaceLocationFlags value) =>
		(value & SpaceLocationFlags.OrientationValid) != 0;

	public static bool IsPositionTracked(this SpaceLocationFlags value) =>
		(value & SpaceLocationFlags.PositionTracked) != 0;

	public static bool IsOrientationTracked(this SpaceLocationFlags value) =>
		(value & SpaceLocationFlags.OrientationTracked) != 0;

	[StructLayout(LayoutKind.Sequential)]
	public struct SpaceLocationf
	{
		public SpaceLocationFlags locationFlags;
		public Posef pose;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct BodyJointLocation
	{
		/// <summary>
		/// The <see cref="SpaceLocationFlags"/> for this <see cref="BodyJointLocation"/>.
		/// </summary>
		public SpaceLocationFlags LocationFlags;

		/// <summary>
		/// The pose of this <see cref="BodyJointLocation"/>.
		/// </summary>
		public Posef Pose;

		/// <summary>
		/// Indicates that the <see cref="Pose"/>'s <see cref="Posef.Orientation"/> contains valid data.
		/// </summary>
		public bool OrientationValid => (LocationFlags & SpaceLocationFlags.OrientationValid) != 0;

		/// <summary>
		/// Indicates that the <see cref="Pose"/>'s <see cref="Posef.Position"/> contains valid data.
		/// </summary>
		public bool PositionValid => (LocationFlags & SpaceLocationFlags.PositionValid) != 0;

		/// <summary>
		/// Indicates that the <see cref="Pose"/>'s <see cref="Posef.Orientation"/> represents an actively tracked
		/// orientation.
		/// </summary>
		public bool OrientationTracked => (LocationFlags & SpaceLocationFlags.OrientationTracked) != 0;

		/// <summary>
		/// Indicates that the <see cref="Pose"/>'s <see cref="Posef.Position"/> represents an actively tracked
		/// position.
		/// </summary>
		public bool PositionTracked => (LocationFlags & SpaceLocationFlags.PositionTracked) != 0;
	}

	/// <summary>
	/// Represents the state of a tracked body.
	/// </summary>
	public struct BodyState
	{
		/// <summary>
		/// The <see cref="BodyJointLocation"/>s for each joint in the tracked body.
		/// </summary>
		public BodyJointLocation[] JointLocations;

		/// <summary>
		/// The confidence of the <see cref="JointLocations"/>.
		/// </summary>
		/// <remarks>
		/// This value ranges from 0 to 1, inclusive. 0 means no confidence while 1 means full confidence.
		/// </remarks>
		public float Confidence;

		/// <summary>
		/// The number of times the skeleton has changed.
		/// </summary>
		public uint SkeletonChangedCount;

		/// <summary>
		/// The time, in seconds, corresponding to this state.
		/// </summary>
		public double Time;
	}

	[StructLayout(LayoutKind.Sequential)]
	private struct BodyStateInternal
	{
		public Bool IsActive;
		public float Confidence;
		public uint SkeletonChangedCount;
		public double Time;
		public BodyJointLocation JointLocation_0;
		public BodyJointLocation JointLocation_1;
		public BodyJointLocation JointLocation_2;
		public BodyJointLocation JointLocation_3;
		public BodyJointLocation JointLocation_4;
		public BodyJointLocation JointLocation_5;
		public BodyJointLocation JointLocation_6;
		public BodyJointLocation JointLocation_7;
		public BodyJointLocation JointLocation_8;
		public BodyJointLocation JointLocation_9;
		public BodyJointLocation JointLocation_10;
		public BodyJointLocation JointLocation_11;
		public BodyJointLocation JointLocation_12;
		public BodyJointLocation JointLocation_13;
		public BodyJointLocation JointLocation_14;
		public BodyJointLocation JointLocation_15;
		public BodyJointLocation JointLocation_16;
		public BodyJointLocation JointLocation_17;
		public BodyJointLocation JointLocation_18;
		public BodyJointLocation JointLocation_19;
		public BodyJointLocation JointLocation_20;
		public BodyJointLocation JointLocation_21;
		public BodyJointLocation JointLocation_22;
		public BodyJointLocation JointLocation_23;
		public BodyJointLocation JointLocation_24;
		public BodyJointLocation JointLocation_25;
		public BodyJointLocation JointLocation_26;
		public BodyJointLocation JointLocation_27;
		public BodyJointLocation JointLocation_28;
		public BodyJointLocation JointLocation_29;
		public BodyJointLocation JointLocation_30;
		public BodyJointLocation JointLocation_31;
		public BodyJointLocation JointLocation_32;
		public BodyJointLocation JointLocation_33;
		public BodyJointLocation JointLocation_34;
		public BodyJointLocation JointLocation_35;
		public BodyJointLocation JointLocation_36;
		public BodyJointLocation JointLocation_37;
		public BodyJointLocation JointLocation_38;
		public BodyJointLocation JointLocation_39;
		public BodyJointLocation JointLocation_40;
		public BodyJointLocation JointLocation_41;
		public BodyJointLocation JointLocation_42;
		public BodyJointLocation JointLocation_43;
		public BodyJointLocation JointLocation_44;
		public BodyJointLocation JointLocation_45;
		public BodyJointLocation JointLocation_46;
		public BodyJointLocation JointLocation_47;
		public BodyJointLocation JointLocation_48;
		public BodyJointLocation JointLocation_49;
		public BodyJointLocation JointLocation_50;
		public BodyJointLocation JointLocation_51;
		public BodyJointLocation JointLocation_52;
		public BodyJointLocation JointLocation_53;
		public BodyJointLocation JointLocation_54;
		public BodyJointLocation JointLocation_55;
		public BodyJointLocation JointLocation_56;
		public BodyJointLocation JointLocation_57;
		public BodyJointLocation JointLocation_58;
		public BodyJointLocation JointLocation_59;
		public BodyJointLocation JointLocation_60;
		public BodyJointLocation JointLocation_61;
		public BodyJointLocation JointLocation_62;
		public BodyJointLocation JointLocation_63;
		public BodyJointLocation JointLocation_64;
		public BodyJointLocation JointLocation_65;
		public BodyJointLocation JointLocation_66;
		public BodyJointLocation JointLocation_67;
		public BodyJointLocation JointLocation_68;
		public BodyJointLocation JointLocation_69;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct KeyboardState
	{
		public Bool IsActive;
		public Bool OrientationValid;
		public Bool PositionValid;
		public Bool OrientationTracked;
		public Bool PositionTracked;
		public PoseStatef PoseState;
		public Vector4f ContrastParameters;
	}

	public enum KeyboardDescriptionConstants
	{
		NameMaxLength = 128,
	}

	// Enum defining the type of the keyboard model, effect render parameters and passthrough configuration.
	public enum TrackedKeyboardPresentationStyles {
		Unknown = 0,
		Opaque = 1,
		KeyLabel = 2,
	}

	// Enum defining the type of the keyboard returned
	public enum TrackedKeyboardFlags {
		Exists = 1,
		Local = 2,
		Remote = 4,
		Connected = 8,
	}

	// Enum defining the type of the keyboard requested
	public enum TrackedKeyboardQueryFlags {
		Local = 2,
		Remote = 4,
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct KeyboardDescription
	{
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = (int)KeyboardDescriptionConstants.NameMaxLength)]
		public byte[] Name;

		public UInt64 TrackedKeyboardId;
		public Vector3f Dimensions;
		public TrackedKeyboardFlags KeyboardFlags;
		public TrackedKeyboardPresentationStyles SupportedPresentationStyles;
	}


	public struct FaceExpressionStatus
	{
		public bool IsValid;
		public bool IsEyeFollowingBlendshapesValid;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct FaceState
	{
		public float[] ExpressionWeights;
		public float[] ExpressionWeightConfidences;
		public FaceExpressionStatus Status;
		public double Time;
	}

	[StructLayout(LayoutKind.Sequential)]
#if OVRPLUGIN_TESTING
	public struct FaceExpressionStatusInternal
#else
	private struct FaceExpressionStatusInternal
#endif // OVRPLUGIN_TESTING
	{
		public Bool IsValid;
		public Bool IsEyeFollowingBlendshapesValid;
		public FaceExpressionStatus ToFaceExpressionStatus() => new FaceExpressionStatus
		{
			IsValid = IsValid == Bool.True,
			IsEyeFollowingBlendshapesValid = IsEyeFollowingBlendshapesValid == Bool.True,
		};
	}

	[StructLayout(LayoutKind.Sequential)]
	private struct FaceStateInternal
	{
		public float ExpressionWeights_0;
		public float ExpressionWeights_1;
		public float ExpressionWeights_2;
		public float ExpressionWeights_3;
		public float ExpressionWeights_4;
		public float ExpressionWeights_5;
		public float ExpressionWeights_6;
		public float ExpressionWeights_7;
		public float ExpressionWeights_8;
		public float ExpressionWeights_9;
		public float ExpressionWeights_10;
		public float ExpressionWeights_11;
		public float ExpressionWeights_12;
		public float ExpressionWeights_13;
		public float ExpressionWeights_14;
		public float ExpressionWeights_15;
		public float ExpressionWeights_16;
		public float ExpressionWeights_17;
		public float ExpressionWeights_18;
		public float ExpressionWeights_19;
		public float ExpressionWeights_20;
		public float ExpressionWeights_21;
		public float ExpressionWeights_22;
		public float ExpressionWeights_23;
		public float ExpressionWeights_24;
		public float ExpressionWeights_25;
		public float ExpressionWeights_26;
		public float ExpressionWeights_27;
		public float ExpressionWeights_28;
		public float ExpressionWeights_29;
		public float ExpressionWeights_30;
		public float ExpressionWeights_31;
		public float ExpressionWeights_32;
		public float ExpressionWeights_33;
		public float ExpressionWeights_34;
		public float ExpressionWeights_35;
		public float ExpressionWeights_36;
		public float ExpressionWeights_37;
		public float ExpressionWeights_38;
		public float ExpressionWeights_39;
		public float ExpressionWeights_40;
		public float ExpressionWeights_41;
		public float ExpressionWeights_42;
		public float ExpressionWeights_43;
		public float ExpressionWeights_44;
		public float ExpressionWeights_45;
		public float ExpressionWeights_46;
		public float ExpressionWeights_47;
		public float ExpressionWeights_48;
		public float ExpressionWeights_49;
		public float ExpressionWeights_50;
		public float ExpressionWeights_51;
		public float ExpressionWeights_52;
		public float ExpressionWeights_53;
		public float ExpressionWeights_54;
		public float ExpressionWeights_55;
		public float ExpressionWeights_56;
		public float ExpressionWeights_57;
		public float ExpressionWeights_58;
		public float ExpressionWeights_59;
		public float ExpressionWeights_60;
		public float ExpressionWeights_61;
		public float ExpressionWeights_62;
		public float ExpressionWeightConfidences_0;
		public float ExpressionWeightConfidences_1;
		public FaceExpressionStatusInternal Status;
		public double Time;
	}

	public enum FaceRegionConfidence
	{
		Lower = 0,
		Upper = 1,
		Max   = 2,
	}

	public enum FaceExpression
	{
		Invalid                = -1,
		Brow_Lowerer_L         = 0,
		Brow_Lowerer_R         = 1,
		Cheek_Puff_L           = 2,
		Cheek_Puff_R           = 3,
		Cheek_Raiser_L         = 4,
		Cheek_Raiser_R         = 5,
		Cheek_Suck_L           = 6,
		Cheek_Suck_R           = 7,
		Chin_Raiser_B          = 8,
		Chin_Raiser_T          = 9,
		Dimpler_L              = 10,
		Dimpler_R              = 11,
		Eyes_Closed_L          = 12,
		Eyes_Closed_R          = 13,
		Eyes_Look_Down_L       = 14,
		Eyes_Look_Down_R       = 15,
		Eyes_Look_Left_L       = 16,
		Eyes_Look_Left_R       = 17,
		Eyes_Look_Right_L      = 18,
		Eyes_Look_Right_R      = 19,
		Eyes_Look_Up_L         = 20,
		Eyes_Look_Up_R         = 21,
		Inner_Brow_Raiser_L    = 22,
		Inner_Brow_Raiser_R    = 23,
		Jaw_Drop               = 24,
		Jaw_Sideways_Left      = 25,
		Jaw_Sideways_Right     = 26,
		Jaw_Thrust             = 27,
		Lid_Tightener_L        = 28,
		Lid_Tightener_R        = 29,
		Lip_Corner_Depressor_L = 30,
		Lip_Corner_Depressor_R = 31,
		Lip_Corner_Puller_L    = 32,
		Lip_Corner_Puller_R    = 33,
		Lip_Funneler_LB        = 34,
		Lip_Funneler_LT        = 35,
		Lip_Funneler_RB        = 36,
		Lip_Funneler_RT        = 37,
		Lip_Pressor_L          = 38,
		Lip_Pressor_R          = 39,
		Lip_Pucker_L           = 40,
		Lip_Pucker_R           = 41,
		Lip_Stretcher_L        = 42,
		Lip_Stretcher_R        = 43,
		Lip_Suck_LB            = 44,
		Lip_Suck_LT            = 45,
		Lip_Suck_RB            = 46,
		Lip_Suck_RT            = 47,
		Lip_Tightener_L        = 48,
		Lip_Tightener_R        = 49,
		Lips_Toward            = 50,
		Lower_Lip_Depressor_L  = 51,
		Lower_Lip_Depressor_R  = 52,
		Mouth_Left             = 53,
		Mouth_Right            = 54,
		Nose_Wrinkler_L        = 55,
		Nose_Wrinkler_R        = 56,
		Outer_Brow_Raiser_L    = 57,
		Outer_Brow_Raiser_R    = 58,
		Upper_Lid_Raiser_L     = 59,
		Upper_Lid_Raiser_R     = 60,
		Upper_Lip_Raiser_L     = 61,
		Upper_Lip_Raiser_R     = 62,
		Max                    = 63,
	}

	public enum FaceConstants
	{
		MaxFaceExpressions = FaceExpression.Max,
		MaxFaceRegionConfidences = FaceRegionConfidence.Max
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct EyeGazeState
	{
		public Posef Pose;
		public float Confidence;
		internal Bool _isValid;
		public bool IsValid => _isValid == Bool.True;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct EyeGazesState
	{
		public EyeGazeState[] EyeGazes;
		public double Time;
	}

	[StructLayout(LayoutKind.Sequential)]
	private struct EyeGazesStateInternal
	{
		public EyeGazeState EyeGazes_0;
		public EyeGazeState EyeGazes_1;
		public double Time;
	}



	public enum ColorSpace
	{
		/// The default value from GetHmdColorSpace until SetClientColorDesc is called. Only valid on PC, and will be remapped to Quest on Mobile
		Unknown = 0,
		/// No color correction, not recommended for production use. See documentation for more info
		Unmanaged = 1,
		/// Preferred color space for standardized color across all Oculus HMDs with D65 white point
		Rec_2020 = 2,
		/// Rec. 709 is used on Oculus Go and shares the same primary color coordinates as sRGB
		Rec_709 = 3,
		/// Oculus Rift CV1 uses a unique color space, see documentation for more info
		Rift_CV1 = 4,
		/// Oculus Rift S uses a unique color space, see documentation for more info
		Rift_S = 5,
		/// Oculus Quest's native color space is slightly different than Rift CV1
		Quest = 6,
		/// Similar to DCI-P3. See documentation for more details on P3
		P3 = 7,
		/// Similar to sRGB but with deeper greens using D65 white point
		Adobe_RGB = 8,
	}

	public enum EventType
	{
		Unknown = 0,
		DisplayRefreshRateChanged = 1,

		SpatialAnchorCreateComplete = 49,
		SpaceSetComponentStatusComplete = 50,
		SpaceQueryResults = 51,
		SpaceQueryComplete = 52,
		SpaceSaveComplete = 53,
		SpaceEraseComplete = 54,
		SpaceShareResult = 56,
		SpaceListSaveResult = 57,

		SceneCaptureComplete = 100,



	}

	private const int EventDataBufferSize = 4000;

	[StructLayout(LayoutKind.Sequential)]
	public struct EventDataBuffer
	{
		public EventType EventType;
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = EventDataBufferSize)]
		public byte[] EventData;
	}

	public const int RENDER_MODEL_NULL_KEY = 0;

	[StructLayout(LayoutKind.Sequential)]
	public struct RenderModelProperties
	{
		public string ModelName;
		public UInt64 ModelKey;
		public uint VendorId;
		public uint ModelVersion;
	}

	[StructLayout(LayoutKind.Sequential)]
	private struct RenderModelPropertiesInternal
	{
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = OVRP_1_68_0.OVRP_RENDER_MODEL_MAX_NAME_LENGTH)]
		public byte[] ModelName;
		public UInt64 ModelKey;
		public uint VendorId;
		public uint ModelVersion;
	}

	[Flags]
	public enum RenderModelFlags
	{
		SupportsGltf20Subset1 = 1,
		SupportsGltf20Subset2 = 2,
	}



	public enum InsightPassthroughColorMapType
	{
		None = 0,
		MonoToRgba = 1,
		MonoToMono = 2,
		BrightnessContrastSaturation = 4,
	}

	public enum InsightPassthroughStyleFlags
	{
		HasTextureOpacityFactor = 1 << 0,
		HasEdgeColor            = 1 << 1,
		HasTextureColorMap      = 1 << 2
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct InsightPassthroughStyle
	{
		public InsightPassthroughStyleFlags Flags;
		public float TextureOpacityFactor;
		public Colorf EdgeColor;
		public InsightPassthroughColorMapType TextureColorMapType;
		public uint TextureColorMapDataSize;
		public IntPtr TextureColorMapData;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct InsightPassthroughKeyboardHandsIntensity
	{
		public float LeftHandIntensity;
		public float RightHandIntensity;
	}

	public enum PassthroughCapabilityFlags
	{
		Passthrough = 1 << 0,
		Color       = 1 << 1,
		Depth       = 1 << 2
	}

	public enum SpaceComponentType
	{
		Locatable = 0,
		Storable = 1,
		Sharable = 2,
		Bounded2D = 3,
		Bounded3D = 4,
		SemanticLabels = 5,
		RoomLayout = 6,
		SpaceContainer = 7,
	}

	public enum SpaceStorageLocation
	{
		Invalid = 0,
		Local = 1,
		Cloud = 2,
	}

	public enum SpaceStoragePersistenceMode
	{
		Invalid = 0,
		Indefinite = 1
	}

	public enum SpaceQueryActionType
	{
		Load = 0,
	}

	public enum SpaceQueryType
	{
		Action = 0
	}

	public enum SpaceQueryFilterType
	{
		None = 0,
		Ids = 1,
		Components = 2,
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct SpatialAnchorCreateInfo
	{
		public TrackingOrigin BaseTracking;
		public Posef PoseInSpace;
		public double Time;
	}


	public const int SpaceFilterInfoIdsMaxSize = 1024;

	[StructLayout(LayoutKind.Sequential)]
	public struct SpaceFilterInfoIds
	{
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = SpaceFilterInfoIdsMaxSize)]
		public Guid[] Ids;
		public int NumIds;
	}

	public const int SpaceFilterInfoComponentsMaxSize = 16;

	[StructLayout(LayoutKind.Sequential)]
	public struct SpaceFilterInfoComponents
	{
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = SpaceFilterInfoComponentsMaxSize)]
		public SpaceComponentType[] Components;
		public int NumComponents;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct SpaceQueryInfo
	{
		public SpaceQueryType QueryType;
		public int MaxQuerySpaces;
		public double Timeout;
		public SpaceStorageLocation Location;
		public SpaceQueryActionType ActionType;
		public SpaceQueryFilterType FilterType;
		public SpaceFilterInfoIds IdInfo;
		public SpaceFilterInfoComponents ComponentsInfo;
	}

	public const int SpatialEntityMaxQueryResultsPerEvent = 128;

	[StructLayout(LayoutKind.Sequential)]
	public struct SpaceQueryResult
	{
		public UInt64 space;
		public Guid uuid;
	}




	//-----------------------------------------------------------------
	// Methods
	//-----------------------------------------------------------------
	public static bool initialized
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			return OVRP_1_1_0.ovrp_GetInitialized() == OVRPlugin.Bool.True;
#endif
		}
	}

#if !OVRPLUGIN_UNSUPPORTED_PLATFORM
	private static XrApi? _nativeXrApi = null;
#endif

	public static XrApi nativeXrApi
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return XrApi.Unknown;
#else
			if (!_nativeXrApi.HasValue)
			{
				_nativeXrApi = XrApi.Unknown;
				if (version >= OVRP_1_55_0.version)
				{
					XrApi xrApi;
					if (OVRP_1_55_0.ovrp_GetNativeXrApiType(out xrApi) == Result.Success)
					{
						_nativeXrApi = xrApi;
					}
				}
			}
			return _nativeXrApi.Value;
#endif
		}
	}

	public static bool chromatic
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_7_0.version)
				return initialized && OVRP_1_7_0.ovrp_GetAppChromaticCorrection() == OVRPlugin.Bool.True;

#if UNITY_ANDROID && !UNITY_EDITOR
			return false;
#else
			return true;
#endif
#endif
		}

		set {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			if (initialized && version >= OVRP_1_7_0.version)
				OVRP_1_7_0.ovrp_SetAppChromaticCorrection(ToBool(value));
#endif
		}
	}

	public static bool monoscopic
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			return initialized && OVRP_1_1_0.ovrp_GetAppMonoscopic() == OVRPlugin.Bool.True;
#endif
		}
		set {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			if (initialized)
			{
				OVRP_1_1_0.ovrp_SetAppMonoscopic(ToBool(value));
			}
#endif
		}
	}

	public static bool rotation
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			return initialized && OVRP_1_1_0.ovrp_GetTrackingOrientationEnabled() == Bool.True;
#endif
		}
		set {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			if (initialized)
			{
				OVRP_1_1_0.ovrp_SetTrackingOrientationEnabled(ToBool(value));
			}
#endif
		}
	}

	public static bool position
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			return initialized && OVRP_1_1_0.ovrp_GetTrackingPositionEnabled() == Bool.True;
#endif
		}
		set {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			if (initialized)
			{
				OVRP_1_1_0.ovrp_SetTrackingPositionEnabled(ToBool(value));
			}
#endif
		}
	}

	public static bool useIPDInPositionTracking
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (initialized && version >= OVRP_1_6_0.version)
				return OVRP_1_6_0.ovrp_GetTrackingIPDEnabled() == OVRPlugin.Bool.True;

			return true;
#endif
		}

		set {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			if (initialized && version >= OVRP_1_6_0.version)
				OVRP_1_6_0.ovrp_SetTrackingIPDEnabled(ToBool(value));
#endif
		}
	}

	public static bool positionSupported
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			return initialized && OVRP_1_1_0.ovrp_GetTrackingPositionSupported() == Bool.True;
#endif
		}
	}

	public static bool positionTracked
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			return initialized && OVRP_1_1_0.ovrp_GetNodePositionTracked(Node.EyeCenter) == Bool.True;
#endif
		}
	}

	public static bool powerSaving
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			return initialized && OVRP_1_1_0.ovrp_GetSystemPowerSavingMode() == Bool.True;
#endif
		}
	}

	public static bool hmdPresent
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			return initialized && OVRP_1_1_0.ovrp_GetNodePresent(Node.EyeCenter) == Bool.True;
#endif
		}
	}

	public static bool userPresent
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			return initialized && OVRP_1_1_0.ovrp_GetUserPresent() == Bool.True;
#endif
		}
	}

	[System.Obsolete("Deprecated. This function will not be supported in OpenXR", false)]
	public static bool headphonesPresent
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			return initialized && OVRP_1_3_0.ovrp_GetSystemHeadphonesPresent() == OVRPlugin.Bool.True;
#endif
		}
	}

	public static int recommendedMSAALevel
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 2;
#else
			if (initialized && version >= OVRP_1_6_0.version)
				return OVRP_1_6_0.ovrp_GetSystemRecommendedMSAALevel();
			else
				return 2;
#endif
		}
	}

	public static SystemRegion systemRegion
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return SystemRegion.Unspecified;
#else
			if (initialized && version >= OVRP_1_5_0.version)
				return OVRP_1_5_0.ovrp_GetSystemRegion();
			else
				return SystemRegion.Unspecified;
#endif
		}
	}

#if !OVRPLUGIN_UNSUPPORTED_PLATFORM
	private static GUID _nativeAudioOutGuid = new OVRPlugin.GUID();
	private static Guid _cachedAudioOutGuid;
	private static string _cachedAudioOutString;
#endif

	public static string audioOutId
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return string.Empty;
#else
			try
			{
				if (_nativeAudioOutGuid == null)
					_nativeAudioOutGuid = new OVRPlugin.GUID();

				IntPtr ptr = OVRP_1_1_0.ovrp_GetAudioOutId();
				if (ptr != IntPtr.Zero)
				{
					Marshal.PtrToStructure(ptr, _nativeAudioOutGuid);
					Guid managedGuid = new Guid(
						_nativeAudioOutGuid.a,
						_nativeAudioOutGuid.b,
						_nativeAudioOutGuid.c,
						_nativeAudioOutGuid.d0,
						_nativeAudioOutGuid.d1,
						_nativeAudioOutGuid.d2,
						_nativeAudioOutGuid.d3,
						_nativeAudioOutGuid.d4,
						_nativeAudioOutGuid.d5,
						_nativeAudioOutGuid.d6,
						_nativeAudioOutGuid.d7);

					if (managedGuid != _cachedAudioOutGuid)
					{
						_cachedAudioOutGuid = managedGuid;
						_cachedAudioOutString = _cachedAudioOutGuid.ToString();
					}

					return _cachedAudioOutString;
				}
			}
			catch { }

			return string.Empty;
#endif
		}
	}

#if !OVRPLUGIN_UNSUPPORTED_PLATFORM
	private static GUID _nativeAudioInGuid = new OVRPlugin.GUID();
	private static Guid _cachedAudioInGuid;
	private static string _cachedAudioInString;
#endif

	public static string audioInId
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return string.Empty;
#else
			try
			{
				if (_nativeAudioInGuid == null)
					_nativeAudioInGuid = new OVRPlugin.GUID();

				IntPtr ptr = OVRP_1_1_0.ovrp_GetAudioInId();
				if (ptr != IntPtr.Zero)
				{
					Marshal.PtrToStructure(ptr, _nativeAudioInGuid);
					Guid managedGuid = new Guid(
						_nativeAudioInGuid.a,
						_nativeAudioInGuid.b,
						_nativeAudioInGuid.c,
						_nativeAudioInGuid.d0,
						_nativeAudioInGuid.d1,
						_nativeAudioInGuid.d2,
						_nativeAudioInGuid.d3,
						_nativeAudioInGuid.d4,
						_nativeAudioInGuid.d5,
						_nativeAudioInGuid.d6,
						_nativeAudioInGuid.d7);

					if (managedGuid != _cachedAudioInGuid)
					{
						_cachedAudioInGuid = managedGuid;
						_cachedAudioInString = _cachedAudioInGuid.ToString();
					}

					return _cachedAudioInString;
				}
			}
			catch { }

			return string.Empty;
#endif
		}
	}

	public static bool hasVrFocus
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			return OVRP_1_1_0.ovrp_GetAppHasVrFocus() == Bool.True;
#endif
		}
	}

	public static bool hasInputFocus
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return true;
#else
			if (version >= OVRP_1_18_0.version)
			{
				Bool inputFocus = Bool.False;
				Result result = OVRP_1_18_0.ovrp_GetAppHasInputFocus(out inputFocus);
				if (Result.Success == result)
					return inputFocus == Bool.True;
				else
				{
					//Debug.LogWarning("ovrp_GetAppHasInputFocus return " + result);
					return false;
				}
			}

			return true;
#endif
		}
	}

	public static bool shouldQuit
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			return OVRP_1_1_0.ovrp_GetAppShouldQuit() == Bool.True;
#endif
		}
	}

	public static bool shouldRecenter
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			return OVRP_1_1_0.ovrp_GetAppShouldRecenter() == Bool.True;
#endif
		}
	}

	public static string productName
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return string.Empty;
#else
			return OVRP_1_1_0.ovrp_GetSystemProductName();
#endif
		}
	}

	public static string latency
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return string.Empty;
#else
			if (!initialized)
				return string.Empty;

			return OVRP_1_1_0.ovrp_GetAppLatencyTimings();
#endif
		}
	}

	public static float eyeDepth
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0.0f;
#else
			if (!initialized)
				return 0.0f;

			return OVRP_1_1_0.ovrp_GetUserEyeDepth();
#endif
		}
		set {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			OVRP_1_1_0.ovrp_SetUserEyeDepth(value);
#endif
		}
	}

	public static float eyeHeight
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0.0f;
#else
			return OVRP_1_1_0.ovrp_GetUserEyeHeight();
#endif
		}
		set {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			OVRP_1_1_0.ovrp_SetUserEyeHeight(value);
#endif
		}
	}

	[System.Obsolete("Deprecated. Please use SystemInfo.batteryLevel", false)]
	public static float batteryLevel
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0.0f;
#else
			return OVRP_1_1_0.ovrp_GetSystemBatteryLevel();
#endif
		}
	}

	[System.Obsolete("Deprecated. This function will not be supported in OpenXR", false)]
	public static float batteryTemperature
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0.0f;
#else
			return OVRP_1_1_0.ovrp_GetSystemBatteryTemperature();
#endif
		}
	}

	public static ProcessorPerformanceLevel suggestedCpuPerfLevel
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return ProcessorPerformanceLevel.SustainedHigh;
#else
			if (version >= OVRP_1_71_0.version)
			{
				ProcessorPerformanceLevel level;
				if (OVRP_1_71_0.ovrp_GetSuggestedCpuPerformanceLevel(out level) == Result.Success)
				{
					return level;
				}
			}
			return ProcessorPerformanceLevel.SustainedHigh;
#endif
		}
		set
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			if (version >= OVRP_1_71_0.version)
			{
				OVRP_1_71_0.ovrp_SetSuggestedCpuPerformanceLevel(value);
			}
#endif
		}
	}

	public static ProcessorPerformanceLevel suggestedGpuPerfLevel
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return ProcessorPerformanceLevel.SustainedHigh;
#else
			if (version >= OVRP_1_71_0.version)
			{
				ProcessorPerformanceLevel level;
				if (OVRP_1_71_0.ovrp_GetSuggestedGpuPerformanceLevel(out level) == Result.Success)
				{
					return level;
				}
			}
			return ProcessorPerformanceLevel.SustainedHigh;
#endif
		}
		set
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			if (version >= OVRP_1_71_0.version)
			{
				OVRP_1_71_0.ovrp_SetSuggestedGpuPerformanceLevel(value);
			}
#endif
		}
	}

	[System.Obsolete("Deprecated. Please use suggestedCpuPerfLevel.", false)]
	public static int cpuLevel
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0;
#else
			return OVRP_1_1_0.ovrp_GetSystemCpuLevel();
#endif
		}
		set {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			OVRP_1_1_0.ovrp_SetSystemCpuLevel(value);
#endif
		}
	}

	[System.Obsolete("Deprecated. Please use suggestedGpuPerfLevel.", false)]
	public static int gpuLevel
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0;
#else
			return OVRP_1_1_0.ovrp_GetSystemGpuLevel();
#endif
		}
		set {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			OVRP_1_1_0.ovrp_SetSystemGpuLevel(value);
#endif
		}
	}

	public static int vsyncCount
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0;
#else
			return OVRP_1_1_0.ovrp_GetSystemVSyncCount();
#endif
		}
		set {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			OVRP_1_2_0.ovrp_SetSystemVSyncCount(value);
#endif
		}
	}

	[System.Obsolete("Deprecated. This function will not be supported in OpenXR", false)]
	public static float systemVolume
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0.0f;
#else
			return OVRP_1_1_0.ovrp_GetSystemVolume();
#endif
		}
	}

	public static float ipd
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0.0f;
#else
			return OVRP_1_1_0.ovrp_GetUserIPD();
#endif
		}
		set {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			OVRP_1_1_0.ovrp_SetUserIPD(value);
#endif
		}
	}

	public static bool occlusionMesh
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			return initialized && (OVRP_1_3_0.ovrp_GetEyeOcclusionMeshEnabled() == Bool.True);
#endif
		}
		set {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			if (!initialized)
				return;

			OVRP_1_3_0.ovrp_SetEyeOcclusionMeshEnabled(ToBool(value));
#endif
		}
	}

	[System.Obsolete("Deprecated. Please use SystemInfo.batteryStatus", false)]
	public static BatteryStatus batteryStatus
	{
		get {
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return default(BatteryStatus);
#else
			return OVRP_1_1_0.ovrp_GetSystemBatteryStatus();
#endif
		}
	}

	public static Frustumf GetEyeFrustum(Eye eyeId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new Frustumf();
#else
		return OVRP_1_1_0.ovrp_GetNodeFrustum((Node)eyeId);
#endif
	}

	public static Sizei GetEyeTextureSize(Eye eyeId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new Sizei();
#else
		return OVRP_0_1_0.ovrp_GetEyeTextureSize(eyeId);
#endif
	}

	public static Posef GetTrackerPose(Tracker trackerId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return Posef.identity;
#else
		return GetNodePose((Node)((int)trackerId + (int)Node.TrackerZero), Step.Render);
#endif
	}

	public static Frustumf GetTrackerFrustum(Tracker trackerId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new Frustumf();
#else
		return OVRP_1_1_0.ovrp_GetNodeFrustum((Node)((int)trackerId + (int)Node.TrackerZero));
#endif
	}

	[System.Obsolete("Deprecated. This function will not be supported in OpenXR", false)]
	public static bool ShowUI(PlatformUI ui)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		return OVRP_1_1_0.ovrp_ShowSystemUI(ui) == Bool.True;
#endif
	}

	public static bool EnqueueSubmitLayer(bool onTop, bool headLocked, bool noDepthBufferTesting, IntPtr leftTexture, IntPtr rightTexture, int layerId, int frameIndex, Posef pose, Vector3f scale, int layerIndex = 0, OverlayShape shape = OverlayShape.Quad,
		bool overrideTextureRectMatrix = false, TextureRectMatrixf textureRectMatrix = default(TextureRectMatrixf), bool overridePerLayerColorScaleAndOffset = false, Vector4 colorScale = default(Vector4), Vector4 colorOffset = default(Vector4),
		bool expensiveSuperSample = false, bool bicubic = false, bool efficientSuperSample = false, bool efficientSharpen = false, bool expensiveSharpen = false, bool hidden = false, bool secureContent = false
	)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (!initialized)
			return false;

		if (version >= OVRP_1_6_0.version)
		{
			uint flags = (uint)OverlayFlag.None;
			if (onTop)
				flags |= (uint)OverlayFlag.OnTop;
			if (headLocked)
				flags |= (uint)OverlayFlag.HeadLocked;
			if (noDepthBufferTesting)
				flags |= (uint)OverlayFlag.NoDepth;
			if (expensiveSuperSample)
				flags |= (uint)OverlayFlag.ExpensiveSuperSample;
			if (hidden)
				flags |= (uint)OverlayFlag.Hidden;
			if (efficientSuperSample)
				flags |= (uint)OverlayFlag.EfficientSuperSample;
			if (expensiveSharpen)
				flags |= (uint)OverlayFlag.ExpensiveSharpen;
			if (efficientSharpen)
				flags |= (uint)OverlayFlag.EfficientSharpen;
			if (bicubic)
				flags |= (uint)OverlayFlag.BicubicFiltering;
			if (secureContent)
				flags |= (uint)OverlayFlag.SecureContent;

			if (shape == OverlayShape.Cylinder || shape == OverlayShape.Cubemap)
			{
#if UNITY_ANDROID
				if (version < OVRP_1_7_0.version)
					return false;
#else
				if (shape == OverlayShape.Cubemap && version < OVRP_1_10_0.version)
					return false;
				else if (shape == OverlayShape.Cylinder && version < OVRP_1_16_0.version)
					return false;
#endif
			}

			if (shape == OverlayShape.OffcenterCubemap)
			{
#if UNITY_ANDROID
				if (version < OVRP_1_11_0.version)
					return false;
#else
				return false;
#endif
			}

			if (shape == OverlayShape.Equirect)
			{
#if UNITY_ANDROID
				if (version < OVRP_1_21_0.version)
					return false;
#else
				return false;
#endif
			}

			if (shape == OverlayShape.Fisheye)
			{
#if UNITY_ANDROID
				if(version < OVRP_1_55_0.version)
					return false;
#else
				return false;
#endif
			}

			if (version >= OVRP_1_34_0.version && layerId != -1)
				return OVRP_1_34_0.ovrp_EnqueueSubmitLayer2(flags, leftTexture, rightTexture, layerId, frameIndex, ref pose, ref scale, layerIndex,
					overrideTextureRectMatrix ? Bool.True : Bool.False, ref textureRectMatrix, overridePerLayerColorScaleAndOffset ? Bool.True : Bool.False, ref colorScale, ref colorOffset) == Result.Success;
			else if (version >= OVRP_1_15_0.version && layerId != -1)
				return OVRP_1_15_0.ovrp_EnqueueSubmitLayer(flags, leftTexture, rightTexture, layerId, frameIndex, ref pose, ref scale, layerIndex) == Result.Success;

			return OVRP_1_6_0.ovrp_SetOverlayQuad3(flags, leftTexture, rightTexture, IntPtr.Zero, pose, scale, layerIndex) == Bool.True;
		}

		if (layerIndex != 0)
			return false;

		return OVRP_0_1_1.ovrp_SetOverlayQuad2(ToBool(onTop), ToBool(headLocked), leftTexture, IntPtr.Zero, pose, scale) == Bool.True;
#endif
	}

	public static LayerDesc CalculateLayerDesc(OverlayShape shape, LayerLayout layout, Sizei textureSize,
		int mipLevels, int sampleCount, EyeTextureFormat format, int layerFlags)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new LayerDesc();
#else
		if (!initialized || version < OVRP_1_15_0.version)
		{
			return new LayerDesc();
		}

		LayerDescInternal layerDescInternal = new LayerDescInternal();
		OVRP_1_15_0.ovrp_CalculateLayerDesc(shape, layout, ref textureSize, mipLevels, sampleCount,
			format, layerFlags, ref layerDescInternal);

		return layerDescInternal.ToLayerDesc();
#endif
	}

	public static bool EnqueueSetupLayer(LayerDesc desc, int compositionDepth, IntPtr layerID)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (!initialized)
			return false;

		LayerDescInternal layerDescInternal = new LayerDescInternal(desc);
		if (version >= OVRP_1_28_0.version) {
			return OVRP_1_28_0.ovrp_EnqueueSetupLayer2(ref layerDescInternal, compositionDepth, layerID) == Result.Success;
		}

		if (version >= OVRP_1_15_0.version)
		{
			if (compositionDepth != 0)
			{
				Debug.LogWarning("Use Oculus Plugin 1.28.0 or above to support non-zero compositionDepth");
			}
			return OVRP_1_15_0.ovrp_EnqueueSetupLayer(ref layerDescInternal, layerID) == Result.Success;
		}
		return false;
#endif
	}


	public static bool EnqueueDestroyLayer(IntPtr layerID)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (!initialized)
			return false;
		if (version >= OVRP_1_15_0.version)
			return OVRP_1_15_0.ovrp_EnqueueDestroyLayer(layerID) == Result.Success;

		return false;
#endif
	}

	public static IntPtr GetLayerTexture(int layerId, int stage, Eye eyeId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return IntPtr.Zero;
#else
		IntPtr textureHandle = IntPtr.Zero;
		if (!initialized)
			return textureHandle;

		if (version >= OVRP_1_15_0.version)
			OVRP_1_15_0.ovrp_GetLayerTexturePtr(layerId, stage, eyeId, ref textureHandle);

		return textureHandle;
#endif
	}

	public static int GetLayerTextureStageCount(int layerId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return 1;
#else
		if (!initialized)
			return 1;

		int stageCount = 1;

		if (version >= OVRP_1_15_0.version)
			OVRP_1_15_0.ovrp_GetLayerTextureStageCount(layerId, ref stageCount);

		return stageCount;
#endif
	}

	public static IntPtr GetLayerAndroidSurfaceObject(int layerId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return IntPtr.Zero;
#else
		IntPtr surfaceObject = IntPtr.Zero;
		if (!initialized)
			return surfaceObject;

		if (version >= OVRP_1_29_0.version)
			OVRP_1_29_0.ovrp_GetLayerAndroidSurfaceObject(layerId, ref surfaceObject);

		return surfaceObject;
#endif
	}

	public static bool UpdateNodePhysicsPoses(int frameIndex, double predictionSeconds)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_8_0.version)
			return OVRP_1_8_0.ovrp_Update2((int)Step.Physics, frameIndex, predictionSeconds) == Bool.True;

		return false;
#endif
	}

	public static Posef GetNodePose(Node nodeId, Step stepId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return Posef.identity;
#else
		if (nativeXrApi == XrApi.OpenXR && stepId == Step.Physics)
		{
			Debug.LogWarning("Step.Physics is deprecated when using OpenXR");
			stepId = Step.Render;
		}

		if (version >= OVRP_1_12_0.version)
			return OVRP_1_12_0.ovrp_GetNodePoseState(stepId, nodeId).Pose;

		if (version >= OVRP_1_8_0.version && stepId == Step.Physics)
			return OVRP_1_8_0.ovrp_GetNodePose2(0, nodeId);

		return OVRP_0_1_2.ovrp_GetNodePose(nodeId);
#endif
	}

	public static Vector3f GetNodeVelocity(Node nodeId, Step stepId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new Vector3f();
#else
		if (nativeXrApi == XrApi.OpenXR && stepId == Step.Physics)
		{
			Debug.LogWarning("Step.Physics is deprecated when using OpenXR");
			stepId = Step.Render;
		}

		if (version >= OVRP_1_12_0.version)
			return OVRP_1_12_0.ovrp_GetNodePoseState(stepId, nodeId).Velocity;

		if (version >= OVRP_1_8_0.version && stepId == Step.Physics)
			return OVRP_1_8_0.ovrp_GetNodeVelocity2(0, nodeId).Position;

		return OVRP_0_1_3.ovrp_GetNodeVelocity(nodeId).Position;
#endif
	}

	public static Vector3f GetNodeAngularVelocity(Node nodeId, Step stepId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new Vector3f();
#else
		if (nativeXrApi == XrApi.OpenXR && stepId == Step.Physics)
		{
			Debug.LogWarning("Step.Physics is deprecated when using OpenXR");
			stepId = Step.Render;
		}

		if (version >= OVRP_1_12_0.version)
			return OVRP_1_12_0.ovrp_GetNodePoseState(stepId, nodeId).AngularVelocity;

		return new Vector3f(); //TODO: Convert legacy quat to vec3?
#endif
	}

	public static Vector3f GetNodeAcceleration(Node nodeId, Step stepId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new Vector3f();
#else
		if (nativeXrApi == XrApi.OpenXR && stepId == Step.Physics)
		{
			Debug.LogWarning("Step.Physics is deprecated when using OpenXR");
			stepId = Step.Render;
		}

		if (version >= OVRP_1_12_0.version)
			return OVRP_1_12_0.ovrp_GetNodePoseState(stepId, nodeId).Acceleration;

		if (version >= OVRP_1_8_0.version && stepId == Step.Physics)
			return OVRP_1_8_0.ovrp_GetNodeAcceleration2(0, nodeId).Position;

		return OVRP_0_1_3.ovrp_GetNodeAcceleration(nodeId).Position;
#endif
	}

	public static Vector3f GetNodeAngularAcceleration(Node nodeId, Step stepId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new Vector3f();
#else
		if (nativeXrApi == XrApi.OpenXR && stepId == Step.Physics)
		{
			Debug.LogWarning("Step.Physics is deprecated when using OpenXR");
			stepId = Step.Render;
		}

		if (version >= OVRP_1_12_0.version)
			return OVRP_1_12_0.ovrp_GetNodePoseState(stepId, nodeId).AngularAcceleration;

		return new Vector3f(); //TODO: Convert legacy quat to vec3?
#endif
	}

	public static bool GetNodePresent(Node nodeId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		return OVRP_1_1_0.ovrp_GetNodePresent(nodeId) == Bool.True;
#endif
	}

	public static bool GetNodeOrientationTracked(Node nodeId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		return OVRP_1_1_0.ovrp_GetNodeOrientationTracked(nodeId) == Bool.True;
#endif
	}

	public static bool GetNodeOrientationValid(Node nodeId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_38_0.version)
		{
			Bool orientationValid = Bool.False;
			Result result = OVRP_1_38_0.ovrp_GetNodeOrientationValid(nodeId, ref orientationValid);
			return result == Result.Success && orientationValid == Bool.True;
		}
		else
		{
			return GetNodeOrientationTracked(nodeId);
		}
#endif

	}

	public static bool GetNodePositionTracked(Node nodeId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		return OVRP_1_1_0.ovrp_GetNodePositionTracked(nodeId) == Bool.True;
#endif
	}

	public static bool GetNodePositionValid(Node nodeId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_38_0.version)
		{
			Bool positionValid = Bool.False;
			Result result = OVRP_1_38_0.ovrp_GetNodePositionValid(nodeId, ref positionValid);
			return result == Result.Success && positionValid == Bool.True;
		}
		else
		{
			return GetNodePositionTracked(nodeId);
		}
#endif
	}

	public static PoseStatef GetNodePoseStateRaw(Node nodeId, Step stepId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return PoseStatef.identity;
#else
		if (nativeXrApi == XrApi.OpenXR && stepId == Step.Physics)
		{
			Debug.LogWarning("Step.Physics is deprecated when using OpenXR");
			stepId = Step.Render;
		}

		if (version >= OVRP_1_29_0.version)
		{
			PoseStatef nodePoseState;
			Result result = OVRP_1_29_0.ovrp_GetNodePoseStateRaw(stepId, -1, nodeId, out nodePoseState);
			if (result == Result.Success)
			{
				return nodePoseState;
			}
			else
			{
				return PoseStatef.identity;
			}
		}
		if (version >= OVRP_1_12_0.version)
			return OVRP_1_12_0.ovrp_GetNodePoseState(stepId, nodeId);
		else
			return PoseStatef.identity;
#endif
	}

	public static PoseStatef GetNodePoseStateAtTime(double time, Node nodeId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return PoseStatef.identity;
#else
		if (version >= OVRP_1_76_0.version)
		{
			PoseStatef nodePoseState;
			Result result = OVRP_1_76_0.ovrp_GetNodePoseStateAtTime(time, nodeId, out nodePoseState);
			if (result == Result.Success)
			{
				return nodePoseState;
			}
		}

		return PoseStatef.identity;
#endif
	}

	public static PoseStatef GetNodePoseStateImmediate(Node nodeId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return PoseStatef.identity;
#else
		if (version >= OVRP_1_69_0.version)
		{
			PoseStatef nodePoseState;
			Result result = OVRP_1_69_0.ovrp_GetNodePoseStateImmediate(nodeId, out nodePoseState);
			if (result == Result.Success)
			{
				return nodePoseState;
			}
			else
			{
				return PoseStatef.identity;
			}
		}
		else
		{
			return PoseStatef.identity;
		}
#endif
	}

	public static Posef GetCurrentTrackingTransformPose()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return Posef.identity;
#else
		if (version >= OVRP_1_30_0.version)
		{
			Posef trackingTransformPose;
			Result result = OVRP_1_30_0.ovrp_GetCurrentTrackingTransformPose(out trackingTransformPose);
			if (result == Result.Success)
			{
				return trackingTransformPose;
			}
			else
			{
				return Posef.identity;
			}
		}
		else
		{
			return Posef.identity;
		}
#endif
	}

	public static Posef GetTrackingTransformRawPose()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return Posef.identity;
#else
		if (version >= OVRP_1_30_0.version)
		{
			Posef trackingTransforRawPose;
			Result result = OVRP_1_30_0.ovrp_GetTrackingTransformRawPose(out trackingTransforRawPose);
			if (result == Result.Success)
			{
				return trackingTransforRawPose;
			}
			else
			{
				return Posef.identity;
			}
		}
		else
		{
			return Posef.identity;
		}
#endif
	}

	public static Posef GetTrackingTransformRelativePose(TrackingOrigin trackingOrigin)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return Posef.identity;
#else
		if (version >= OVRP_1_38_0.version)
		{
			Posef trackingTransformRelativePose = Posef.identity;
			Result result = OVRP_1_38_0.ovrp_GetTrackingTransformRelativePose(ref trackingTransformRelativePose, trackingOrigin);
			if (result == Result.Success)
			{
				return trackingTransformRelativePose;
			}
			else
			{
				return Posef.identity;
			}
		}
		else
		{
			return Posef.identity;
		}
#endif
	}

	public static ControllerState GetControllerState(uint controllerMask)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new ControllerState();
#else
		return OVRP_1_1_0.ovrp_GetControllerState(controllerMask);
#endif
	}

	public static ControllerState2 GetControllerState2(uint controllerMask)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new ControllerState2();
#else
		if (version >= OVRP_1_12_0.version)
		{
			return OVRP_1_12_0.ovrp_GetControllerState2(controllerMask);
		}

		return new ControllerState2(OVRP_1_1_0.ovrp_GetControllerState(controllerMask));
#endif
	}

	public static ControllerState4 GetControllerState4(uint controllerMask)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new ControllerState4();
#else
		if (version >= OVRP_1_16_0.version)
		{
			ControllerState4 controllerState = new ControllerState4();
			OVRP_1_16_0.ovrp_GetControllerState4(controllerMask, ref controllerState);
			return controllerState;
		}

		return new ControllerState4(GetControllerState2(controllerMask));
#endif
	}

	public static ControllerState5 GetControllerState5(uint controllerMask)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new ControllerState5();
#else
		if (version >= OVRP_1_78_0.version)
		{
			ControllerState5 controllerState = new ControllerState5();
			OVRP_1_78_0.ovrp_GetControllerState5(controllerMask, ref controllerState);
			return controllerState;
		}

		return new ControllerState5(GetControllerState4(controllerMask));
#endif
	}


	public static InteractionProfile GetCurrentInteractionProfile(Hand hand)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return InteractionProfile.None;
#else
		InteractionProfile interactionProfile = InteractionProfile.None;

		if (version >= OVRP_1_78_0.version)
		{
			OVRP_1_78_0.ovrp_GetCurrentInteractionProfile(hand, out interactionProfile);
		}

		return interactionProfile;
#endif
	}

	public static bool SetControllerVibration(uint controllerMask, float frequency, float amplitude)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		return OVRP_0_1_2.ovrp_SetControllerVibration(controllerMask, frequency, amplitude) == Bool.True;
#endif
	}

	public static bool SetControllerLocalizedVibration(Controller controllerMask, HapticsLocation hapticsLocationMask, float frequency, float amplitude)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_78_0.version)
		{
			Result result = OVRP_1_78_0.ovrp_SetControllerLocalizedVibration(controllerMask, hapticsLocationMask, frequency, amplitude);
			return (result == Result.Success);
		}

		return false;
#endif
	}

	public static bool SetControllerHapticsAmplitudeEnvelope(Controller controllerMask, HapticsAmplitudeEnvelopeVibration hapticsVibration)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_78_0.version)
		{
			Result result = OVRP_1_78_0.ovrp_SetControllerHapticsAmplitudeEnvelope(controllerMask, hapticsVibration);
			return (result == Result.Success);
		}

		return false;
#endif
	}

	public static bool SetControllerHapticsPcm(Controller controllerMask, HapticsPcmVibration hapticsVibration)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_78_0.version)
		{
			Result result = OVRP_1_78_0.ovrp_SetControllerHapticsPcm(controllerMask, hapticsVibration);
			return (result == Result.Success);
		}

		return false;
#endif
	}

	public static bool GetControllerSampleRateHz(Controller controllerMask, out float sampleRateHz)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		sampleRateHz = 0.0f;
		return false;
#else
		if (version >= OVRP_1_78_0.version)
		{
			Result result = OVRP_1_78_0.ovrp_GetControllerSampleRateHz(controllerMask, out sampleRateHz);
			return (result == Result.Success);
		}

		sampleRateHz = 0.0f;
		return false;
#endif
	}

	public static HapticsDesc GetControllerHapticsDesc(uint controllerMask)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new HapticsDesc();
#else
		if (version >= OVRP_1_6_0.version)
		{
			return OVRP_1_6_0.ovrp_GetControllerHapticsDesc(controllerMask);
		}
		else
		{
			return new HapticsDesc();
		}
#endif
	}

	public static HapticsState GetControllerHapticsState(uint controllerMask)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new HapticsState();
#else
		if (version >= OVRP_1_6_0.version)
		{
			return OVRP_1_6_0.ovrp_GetControllerHapticsState(controllerMask);
		}
		else
		{
			return new HapticsState();
		}
#endif
	}

	public static bool SetControllerHaptics(uint controllerMask, HapticsBuffer hapticsBuffer)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_6_0.version)
		{
			return OVRP_1_6_0.ovrp_SetControllerHaptics(controllerMask, hapticsBuffer) == Bool.True;
		}
		else
		{
			return false;
		}
#endif
	}

	public static float GetEyeRecommendedResolutionScale()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return 1.0f;
#else
		if (version >= OVRP_1_6_0.version)
		{
			return OVRP_1_6_0.ovrp_GetEyeRecommendedResolutionScale();
		}
		else
		{
			return 1.0f;
		}
#endif
	}

	public static float GetAppCpuStartToGpuEndTime()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return 0.0f;
#else
		if (version >= OVRP_1_6_0.version)
		{
			return OVRP_1_6_0.ovrp_GetAppCpuStartToGpuEndTime();
		}
		else
		{
			return 0.0f;
		}
#endif
	}

	public static bool GetBoundaryConfigured()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_8_0.version)
		{
			return OVRP_1_8_0.ovrp_GetBoundaryConfigured() == OVRPlugin.Bool.True;
		}
		else
		{
			return false;
		}
#endif
	}

	[System.Obsolete("Deprecated. This function will not be supported in OpenXR", false)]
	public static BoundaryTestResult TestBoundaryNode(Node nodeId, BoundaryType boundaryType)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new BoundaryTestResult();
#else
		if (version >= OVRP_1_8_0.version)
		{
			return OVRP_1_8_0.ovrp_TestBoundaryNode(nodeId, boundaryType);
		}
		else
		{
			return new BoundaryTestResult();
		}
#endif
	}

	[System.Obsolete("Deprecated. This function will not be supported in OpenXR", false)]
	public static BoundaryTestResult TestBoundaryPoint(Vector3f point, BoundaryType boundaryType)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new BoundaryTestResult();
#else
		if (version >= OVRP_1_8_0.version)
		{
			return OVRP_1_8_0.ovrp_TestBoundaryPoint(point, boundaryType);
		}
		else
		{
			return new BoundaryTestResult();
		}
#endif
	}

	public static BoundaryGeometry GetBoundaryGeometry(BoundaryType boundaryType)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new BoundaryGeometry();
#else
		if (version >= OVRP_1_8_0.version)
		{
			return OVRP_1_8_0.ovrp_GetBoundaryGeometry(boundaryType);
		}
		else
		{
			return new BoundaryGeometry();
		}
#endif
	}

	public static bool GetBoundaryGeometry2(BoundaryType boundaryType, IntPtr points, ref int pointsCount)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		pointsCount = 0;
		return false;
#else
		if (version >= OVRP_1_9_0.version)
		{
			return OVRP_1_9_0.ovrp_GetBoundaryGeometry2(boundaryType, points, ref pointsCount) == OVRPlugin.Bool.True;
		}
		else
		{
			pointsCount = 0;

			return false;
		}
#endif
	}

	private static bool perfStatWarningPrinted = false;
	public static AppPerfStats GetAppPerfStats()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new AppPerfStats();
#else
		if (nativeXrApi == XrApi.OpenXR)
		{
			if (!perfStatWarningPrinted)
			{
				Debug.LogWarning("GetAppPerfStats is currently unsupported on OpenXR.");
				perfStatWarningPrinted = true;
			}
			return new AppPerfStats();
		}

		if (version >= OVRP_1_9_0.version)
		{
			return OVRP_1_9_0.ovrp_GetAppPerfStats();
		}
		else
		{
			return new AppPerfStats();
		}
#endif
	}

	private static bool resetPerfStatWarningPrinted = false;
	public static bool ResetAppPerfStats()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (nativeXrApi == XrApi.OpenXR)
		{
			if (!resetPerfStatWarningPrinted)
			{
				Debug.LogWarning("ResetAppPerfStats is currently unsupported on OpenXR.");
				resetPerfStatWarningPrinted = true;
			}
			return false;
		}

		if (version >= OVRP_1_9_0.version)
		{
			return OVRP_1_9_0.ovrp_ResetAppPerfStats() == OVRPlugin.Bool.True;
		}
		else
		{
			return false;
		}
#endif
	}

	public static float GetAppFramerate()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return 0.0f;
#else
		if (version >= OVRP_1_12_0.version)
		{
			return OVRP_1_12_0.ovrp_GetAppFramerate();
		}
		else
		{
			return 0.0f;
		}
#endif
	}

	public static bool SetHandNodePoseStateLatency(double latencyInSeconds)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_18_0.version)
		{
			Result result = OVRP_1_18_0.ovrp_SetHandNodePoseStateLatency(latencyInSeconds);
			if (result == Result.Success)
			{
				return true;
			}
			else
			{
				//Debug.LogWarning("ovrp_SetHandNodePoseStateLatency return " + result);
				return false;
			}
		}
		else
		{
			return false;
		}
#endif
	}

	public static double GetHandNodePoseStateLatency()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return 0.0;
#else
		if (version >= OVRP_1_18_0.version)
		{
			double value = 0.0;
			if (OVRP_1_18_0.ovrp_GetHandNodePoseStateLatency(out value) == OVRPlugin.Result.Success)
			{
				return value;
			}
			else
			{
				return 0.0;
			}
		}
		else
		{
			return 0.0;
		}
#endif
	}

	public static EyeTextureFormat GetDesiredEyeTextureFormat()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return EyeTextureFormat.Default;
#else
		if (version >= OVRP_1_11_0.version)
		{
			uint eyeTextureFormatValue = (uint)OVRP_1_11_0.ovrp_GetDesiredEyeTextureFormat();

			// convert both R8G8B8A8 and R8G8B8A8_SRGB to R8G8B8A8 here for avoid confusing developers
			if (eyeTextureFormatValue == 1)
				eyeTextureFormatValue = 0;

			return (EyeTextureFormat)eyeTextureFormatValue;
		}
		else
		{
			return EyeTextureFormat.Default;
		}
#endif
	}

	public static bool SetDesiredEyeTextureFormat(EyeTextureFormat value)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_11_0.version)
		{
			return OVRP_1_11_0.ovrp_SetDesiredEyeTextureFormat(value) == OVRPlugin.Bool.True;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool InitializeMixedReality()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else

#if OVRPLUGIN_INCLUDE_MRC_ANDROID
		if (version >= OVRP_1_38_0.version)		// MRC functions are invalid before 1.38.0
#else
		if (version >= OVRP_1_15_0.version)
#endif
		{
			Result result = OVRP_1_15_0.ovrp_InitializeMixedReality();
			if (result != Result.Success)
			{
				//Debug.LogWarning("ovrp_InitializeMixedReality return " + result);
			}
			return result == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool ShutdownMixedReality()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else

#if OVRPLUGIN_INCLUDE_MRC_ANDROID
		if (version >= OVRP_1_38_0.version)		// MRC functions are invalid before 1.38.0
#else
		if (version >= OVRP_1_15_0.version)
#endif
		{
			Result result = OVRP_1_15_0.ovrp_ShutdownMixedReality();
			if (result != Result.Success)
			{
				//Debug.LogWarning("ovrp_ShutdownMixedReality return " + result);
			}
			return result == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool IsMixedRealityInitialized()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else

#if OVRPLUGIN_INCLUDE_MRC_ANDROID
		if (version >= OVRP_1_38_0.version)		// MRC functions are invalid before 1.38.0
#else
		if (version >= OVRP_1_15_0.version)
#endif
		{
			return OVRP_1_15_0.ovrp_GetMixedRealityInitialized() == Bool.True;
		}
		else
		{
			return false;
		}
#endif
	}

	public static int GetExternalCameraCount()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return 0;
#else

#if OVRPLUGIN_INCLUDE_MRC_ANDROID
		if (version >= OVRP_1_38_0.version)		// MRC functions are invalid before 1.38.0
#else
		if (version >= OVRP_1_15_0.version)
#endif
		{
			int cameraCount = 0;
			Result result = OVRP_1_15_0.ovrp_GetExternalCameraCount(out cameraCount);
			if (result != OVRPlugin.Result.Success)
			{
				//Debug.LogWarning("ovrp_GetExternalCameraCount return " + result);
				return 0;
			}

			return cameraCount;
		}
		else
		{
			return 0;
		}
#endif
	}

	public static bool UpdateExternalCamera()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else

#if OVRPLUGIN_INCLUDE_MRC_ANDROID
		if (version >= OVRP_1_38_0.version)		// MRC functions are invalid before 1.38.0
#else
		if (version >= OVRP_1_15_0.version)
#endif
		{
			Result result = OVRP_1_15_0.ovrp_UpdateExternalCamera();
			if (result != Result.Success)
			{
				//Debug.LogWarning("ovrp_UpdateExternalCamera return " + result);
			}
			return result == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool GetMixedRealityCameraInfo(int cameraId, out CameraExtrinsics cameraExtrinsics, out CameraIntrinsics cameraIntrinsics)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		cameraExtrinsics = default(CameraExtrinsics);
		cameraIntrinsics = default(CameraIntrinsics);
		return false;
#else

		cameraExtrinsics = default(CameraExtrinsics);
		cameraIntrinsics = default(CameraIntrinsics);

#if OVRPLUGIN_INCLUDE_MRC_ANDROID
		if (version >= OVRP_1_38_0.version)		// MRC functions are invalid before 1.38.0
#else
		if (version >= OVRP_1_15_0.version)
#endif
		{
			bool retValue = true;

			Result result = OVRP_1_15_0.ovrp_GetExternalCameraExtrinsics(cameraId, out cameraExtrinsics);
			if (result != Result.Success)
			{
				retValue = false;
				//Debug.LogWarning("ovrp_GetExternalCameraExtrinsics return " + result);
			}

			result = OVRP_1_15_0.ovrp_GetExternalCameraIntrinsics(cameraId, out cameraIntrinsics);
			if (result != Result.Success)
			{
				retValue = false;
				//Debug.LogWarning("ovrp_GetExternalCameraIntrinsics return " + result);
			}

			return retValue;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool OverrideExternalCameraFov(int cameraId, bool useOverriddenFov, Fovf fov)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_44_0.version)
		{
			bool retValue = true;
			Result result = OVRP_1_44_0.ovrp_OverrideExternalCameraFov(cameraId, useOverriddenFov ? Bool.True : Bool.False, ref fov);
			if (result != Result.Success)
			{
				retValue = false;
			}
			return retValue;
		}
		else
		{
			return false;
		}
#endif
	}


	public static bool GetUseOverriddenExternalCameraFov(int cameraId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_44_0.version)
		{
			bool retValue = true;
			Bool useOverriddenFov = Bool.False;
			Result result = OVRP_1_44_0.ovrp_GetUseOverriddenExternalCameraFov(cameraId, out useOverriddenFov);
			if (result != Result.Success)
			{
				retValue = false;
			}
			if (useOverriddenFov == Bool.False)
			{
				retValue = false;
			}
			return retValue;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool OverrideExternalCameraStaticPose(int cameraId, bool useOverriddenPose, Posef poseInStageOrigin)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_44_0.version)
		{
			bool retValue = true;
			Result result = OVRP_1_44_0.ovrp_OverrideExternalCameraStaticPose(cameraId, useOverriddenPose ? Bool.True : Bool.False, ref poseInStageOrigin);
			if (result != Result.Success)
			{
				retValue = false;
			}
			return retValue;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool GetUseOverriddenExternalCameraStaticPose(int cameraId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_44_0.version)
		{
			bool retValue = true;
			Bool useOverriddenStaticPose = Bool.False;
			Result result = OVRP_1_44_0.ovrp_GetUseOverriddenExternalCameraStaticPose(cameraId, out useOverriddenStaticPose);
			if (result != Result.Success)
			{
				retValue = false;
			}
			if (useOverriddenStaticPose == Bool.False)
			{
				retValue = false;
			}
			return retValue;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool ResetDefaultExternalCamera()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_44_0.version)
		{
			Result result = OVRP_1_44_0.ovrp_ResetDefaultExternalCamera();
			if (result != Result.Success)
			{
				return false;
			}
			return true;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool SetDefaultExternalCamera(string cameraName, ref CameraIntrinsics cameraIntrinsics, ref CameraExtrinsics cameraExtrinsics)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_44_0.version)
		{
			Result result = OVRP_1_44_0.ovrp_SetDefaultExternalCamera(cameraName, ref cameraIntrinsics, ref cameraExtrinsics);
			if (result != Result.Success)
			{
				return false;
			}
			return true;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool SetExternalCameraProperties(string cameraName, ref CameraIntrinsics cameraIntrinsics, ref CameraExtrinsics cameraExtrinsics)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_48_0.version)
		{
			Result result = OVRP_1_48_0.ovrp_SetExternalCameraProperties(cameraName, ref cameraIntrinsics, ref cameraExtrinsics);
			if (result != Result.Success)
			{
				return false;
			}
			return true;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool IsInsightPassthroughSupported()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_71_0.version)
		{
			Bool supported = OVRPlugin.Bool.False;
			Result result = OVRP_1_71_0.ovrp_IsInsightPassthroughSupported(ref supported);
			if (result == Result.Success)
			{
				return supported == OVRPlugin.Bool.True;
			}

			Debug.LogError("Unable to determine whether passthrough is supported. Try calling IsInsightPassthroughSupported() while the XR plug-in is initialized. Failed with reason: " + result);
			return false;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool InitializeInsightPassthrough()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_63_0.version)
		{
			Result result = OVRP_1_63_0.ovrp_InitializeInsightPassthrough();
			if (result != Result.Success)
			{
				return false;
			}
			return true;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool ShutdownInsightPassthrough()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_63_0.version)
		{
			Result result = OVRP_1_63_0.ovrp_ShutdownInsightPassthrough();
			if (result != Result.Success)
			{
				return false;
			}
			return true;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool IsInsightPassthroughInitialized()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_63_0.version)
		{
			Bool result = OVRP_1_63_0.ovrp_GetInsightPassthroughInitialized();
			return result == Bool.True;
		}
		else
		{
			return false;
		}
#endif
	}

	public static Result GetInsightPassthroughInitializationState()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
				return Result.Failure_Unsupported;
#else
		if (version >= OVRP_1_66_0.version)
		{
			return OVRP_1_66_0.ovrp_GetInsightPassthroughInitializationState();
		}
		else
		{
			return Result.Failure_Unsupported;
		}
#endif
	}

	public static bool CreateInsightTriangleMesh(int layerId, Vector3[] vertices, int[] triangles, out ulong meshHandle)
	{
		meshHandle = 0;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_63_0.version)
		{
			if (vertices == null || triangles == null || vertices.Length == 0 || triangles.Length == 0)
			{
				return false;
			}
			int vertexCount = vertices.Length;
			int triangleCount = triangles.Length / 3;
			GCHandle pinnedVertexData = GCHandle.Alloc(vertices, GCHandleType.Pinned);
			IntPtr vertexDataPtr = pinnedVertexData.AddrOfPinnedObject();
			GCHandle pinnedTriangleData = GCHandle.Alloc(triangles, GCHandleType.Pinned);
			IntPtr triangleDataPtr = pinnedTriangleData.AddrOfPinnedObject();
			Result result = OVRP_1_63_0.ovrp_CreateInsightTriangleMesh(
				layerId, vertexDataPtr, vertexCount, triangleDataPtr, triangleCount, out meshHandle);
			pinnedTriangleData.Free();
			pinnedVertexData.Free();
			if (result != Result.Success)
			{
				return false;
			}
			return true;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool DestroyInsightTriangleMesh(ulong meshHandle)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_63_0.version)
		{
			Result result = OVRP_1_63_0.ovrp_DestroyInsightTriangleMesh(meshHandle);
			if (result != Result.Success)
			{
				return false;
			}
			return true;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool AddInsightPassthroughSurfaceGeometry(int layerId, ulong meshHandle, Matrix4x4 T_world_model, out ulong geometryInstanceHandle)
	{
		geometryInstanceHandle = 0;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_63_0.version)
		{
			Result result = OVRP_1_63_0.ovrp_AddInsightPassthroughSurfaceGeometry(layerId, meshHandle, T_world_model, out geometryInstanceHandle);
			if (result != Result.Success)
			{
				return false;
			}
			return true;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool DestroyInsightPassthroughGeometryInstance(ulong geometryInstanceHandle)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_63_0.version)
		{
			Result result = OVRP_1_63_0.ovrp_DestroyInsightPassthroughGeometryInstance(geometryInstanceHandle);
			if (result != Result.Success)
			{
				return false;
			}
			return true;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool UpdateInsightPassthroughGeometryTransform(ulong geometryInstanceHandle, Matrix4x4 transform)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_63_0.version)
		{
			Result result = OVRP_1_63_0.ovrp_UpdateInsightPassthroughGeometryTransform(geometryInstanceHandle, transform);
			if (result != Result.Success)
			{
				return false;
			}
			return true;
		}
		else
		{
			return false;
		}
#endif
	}
	public static bool SetInsightPassthroughStyle(int layerId, InsightPassthroughStyle style)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_63_0.version)
		{
			Result result = OVRP_1_63_0.ovrp_SetInsightPassthroughStyle(layerId, style);
			if (result != Result.Success)
			{
				return false;
			}
			return true;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool SetInsightPassthroughKeyboardHandsIntensity(int layerId, InsightPassthroughKeyboardHandsIntensity intensity)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_68_0.version)
		{
			Result result = OVRP_1_68_0.ovrp_SetInsightPassthroughKeyboardHandsIntensity(layerId, intensity);
			if (result != Result.Success)
			{
				return false;
			}
			return true;
		}
		else
		{
			return false;
		}
#endif
	}

	public static PassthroughCapabilityFlags GetPassthroughCapabilityFlags()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return default;
#else
		if (version >= OVRP_1_78_0.version)
		{
			PassthroughCapabilityFlags returnValue = 0;
			Result result = OVRP_1_78_0.ovrp_GetPassthroughCapabilityFlags(ref returnValue);
			if (result == Result.Success)
			{
				return returnValue;
			}
			Debug.LogError("Unable to retrieve passthrough capability flags. Try calling GetInsightPassthroughCapabilityFlags() while the XR plug-in is initialized. Failed with reason: " + result);
		} else {
			Debug.LogWarning("ovrp_GetPassthroughCapabilityFlags() not yet supported by OVRPlugin. Result of GetInsightPassthroughCapabilityFlags() is not accurate.");
		}
		// Fallback to returning result of IsInsightPassthroughSupported().
		return IsInsightPassthroughSupported() ? PassthroughCapabilityFlags.Passthrough : 0;
#endif
	}

	public static Vector3f GetBoundaryDimensions(BoundaryType boundaryType)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return new Vector3f();
#else
		if (version >= OVRP_1_8_0.version)
		{
			return OVRP_1_8_0.ovrp_GetBoundaryDimensions(boundaryType);
		}
		else
		{
			return new Vector3f();
		}
#endif
	}

	[System.Obsolete("Deprecated. This function will not be supported in OpenXR", false)]
	public static bool GetBoundaryVisible()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_8_0.version)
		{
			return OVRP_1_8_0.ovrp_GetBoundaryVisible() == OVRPlugin.Bool.True;
		}
		else
		{
			return false;
		}
#endif
	}

	[System.Obsolete("Deprecated. This function will not be supported in OpenXR", false)]
	public static bool SetBoundaryVisible(bool value)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_8_0.version)
		{
			return OVRP_1_8_0.ovrp_SetBoundaryVisible(ToBool(value)) == OVRPlugin.Bool.True;
		}
		else
		{
			return false;
		}
#endif
	}

	public static SystemHeadset GetSystemHeadsetType()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return SystemHeadset.None;
#else
		if (version >= OVRP_1_9_0.version)
			return OVRP_1_9_0.ovrp_GetSystemHeadsetType();

		return SystemHeadset.None;
#endif
	}

	public static Controller GetActiveController()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return Controller.None;
#else
		if (version >= OVRP_1_9_0.version)
			return OVRP_1_9_0.ovrp_GetActiveController();

		return Controller.None;
#endif
	}

	public static Controller GetConnectedControllers()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return Controller.None;
#else
		if (version >= OVRP_1_9_0.version)
			return OVRP_1_9_0.ovrp_GetConnectedControllers();

		return Controller.None;
#endif
	}

	private static Bool ToBool(bool b)
	{
		return (b) ? OVRPlugin.Bool.True : OVRPlugin.Bool.False;
	}

	public static TrackingOrigin GetTrackingOriginType()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return default(TrackingOrigin);
#else
		return OVRP_1_0_0.ovrp_GetTrackingOriginType();
#endif
	}

	public static bool SetTrackingOriginType(TrackingOrigin originType)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		return OVRP_1_0_0.ovrp_SetTrackingOriginType(originType) == Bool.True;
#endif
	}

	public static Posef GetTrackingCalibratedOrigin()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return Posef.identity;
#else
		return OVRP_1_0_0.ovrp_GetTrackingCalibratedOrigin();
#endif
	}

	public static bool SetTrackingCalibratedOrigin()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		return OVRP_1_2_0.ovrpi_SetTrackingCalibratedOrigin() == Bool.True;
#endif
	}

	public static bool RecenterTrackingOrigin(RecenterFlags flags)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		return OVRP_1_0_0.ovrp_RecenterTrackingOrigin((uint)flags) == Bool.True;
#endif
	}

#if UNITY_EDITOR || UNITY_STANDALONE_WIN
	public static bool UpdateCameraDevices()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_16_0.version)
		{
			Result result = OVRP_1_16_0.ovrp_UpdateCameraDevices();
			if (result != Result.Success)
			{
				//Debug.LogWarning("ovrp_UpdateCameraDevices return " + result);
			}
			return result == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool IsCameraDeviceAvailable(CameraDevice cameraDevice)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_16_0.version)
		{
			Bool result = OVRP_1_16_0.ovrp_IsCameraDeviceAvailable(cameraDevice);
			return result == Bool.True;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool SetCameraDevicePreferredColorFrameSize(CameraDevice cameraDevice, int width, int height)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_16_0.version)
		{
			Sizei size = new Sizei();
			size.w = width;
			size.h = height;
			Result result = OVRP_1_16_0.ovrp_SetCameraDevicePreferredColorFrameSize(cameraDevice, size);
			if (result != Result.Success)
			{
				//Debug.LogWarning("ovrp_SetCameraDevicePreferredColorFrameSize return " + result);
			}
			return result == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool OpenCameraDevice(CameraDevice cameraDevice)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_16_0.version)
		{
			Result result = OVRP_1_16_0.ovrp_OpenCameraDevice(cameraDevice);
			if (result != Result.Success)
			{
				//Debug.LogWarning("ovrp_OpenCameraDevice return " + result);
			}
			return result == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool CloseCameraDevice(CameraDevice cameraDevice)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_16_0.version)
		{
			Result result = OVRP_1_16_0.ovrp_CloseCameraDevice(cameraDevice);
			if (result != Result.Success)
			{
				//Debug.LogWarning("ovrp_OpenCameraDevice return " + result);
			}
			return result == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool HasCameraDeviceOpened(CameraDevice cameraDevice)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_16_0.version)
		{
			Bool result = OVRP_1_16_0.ovrp_HasCameraDeviceOpened(cameraDevice);
			return result == Bool.True;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool IsCameraDeviceColorFrameAvailable(CameraDevice cameraDevice)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_16_0.version)
		{
			Bool result = OVRP_1_16_0.ovrp_IsCameraDeviceColorFrameAvailable(cameraDevice);
			return result == Bool.True;
		}
		else
		{
			return false;
		}
#endif
	}

#if !OVRPLUGIN_UNSUPPORTED_PLATFORM
	private static Texture2D cachedCameraFrameTexture = null;
#endif
	public static Texture2D GetCameraDeviceColorFrameTexture(CameraDevice cameraDevice)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return null;
#else
		if (version >= OVRP_1_16_0.version)
		{
			Sizei size = new Sizei();
			Result result = OVRP_1_16_0.ovrp_GetCameraDeviceColorFrameSize(cameraDevice, out size);
			if (result != Result.Success)
			{
				//Debug.LogWarning("ovrp_GetCameraDeviceColorFrameSize return " + result);
				return null;
			}
			IntPtr pixels;
			int rowPitch;
			result = OVRP_1_16_0.ovrp_GetCameraDeviceColorFrameBgraPixels(cameraDevice, out pixels, out rowPitch);
			if (result != Result.Success)
			{
				//Debug.LogWarning("ovrp_GetCameraDeviceColorFrameBgraPixels return " + result);
				return null;
			}
			if (rowPitch != size.w * 4)
			{
				//Debug.LogWarning(string.Format("RowPitch mismatch, expected {0}, get {1}", size.w * 4, rowPitch));
				return null;
			}
			if (!cachedCameraFrameTexture || cachedCameraFrameTexture.width != size.w || cachedCameraFrameTexture.height != size.h)
			{
				cachedCameraFrameTexture = new Texture2D(size.w, size.h, TextureFormat.BGRA32, false);
			}
			cachedCameraFrameTexture.LoadRawTextureData(pixels, rowPitch * size.h);
			cachedCameraFrameTexture.Apply();
			return cachedCameraFrameTexture;
		}
		else
		{
			return null;
		}
#endif
	}

	public static bool DoesCameraDeviceSupportDepth(CameraDevice cameraDevice)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_17_0.version)
		{
			Bool supportDepth;
			Result result = OVRP_1_17_0.ovrp_DoesCameraDeviceSupportDepth(cameraDevice, out supportDepth);
			return result == Result.Success && supportDepth == Bool.True;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool SetCameraDeviceDepthSensingMode(CameraDevice camera, CameraDeviceDepthSensingMode depthSensoringMode)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_17_0.version)
		{
			Result result = OVRP_1_17_0.ovrp_SetCameraDeviceDepthSensingMode(camera, depthSensoringMode);
			return result == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool SetCameraDevicePreferredDepthQuality(CameraDevice camera, CameraDeviceDepthQuality depthQuality)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_17_0.version)
		{
			Result result = OVRP_1_17_0.ovrp_SetCameraDevicePreferredDepthQuality(camera, depthQuality);
			return result == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool IsCameraDeviceDepthFrameAvailable(CameraDevice cameraDevice)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_17_0.version)
		{
			Bool available;
			Result result = OVRP_1_17_0.ovrp_IsCameraDeviceDepthFrameAvailable(cameraDevice, out available);
			return result == Result.Success && available == Bool.True;
		}
		else
		{
			return false;
		}
#endif
	}

#if !OVRPLUGIN_UNSUPPORTED_PLATFORM
	private static Texture2D cachedCameraDepthTexture = null;
#endif
	public static Texture2D GetCameraDeviceDepthFrameTexture(CameraDevice cameraDevice)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return null;
#else
		if (version >= OVRP_1_17_0.version)
		{
			Sizei size = new Sizei();
			Result result = OVRP_1_17_0.ovrp_GetCameraDeviceDepthFrameSize(cameraDevice, out size);
			if (result != Result.Success)
			{
				//Debug.LogWarning("ovrp_GetCameraDeviceDepthFrameSize return " + result);
				return null;
			}
			IntPtr depthData;
			int rowPitch;
			result = OVRP_1_17_0.ovrp_GetCameraDeviceDepthFramePixels(cameraDevice, out depthData, out rowPitch);
			if (result != Result.Success)
			{
				//Debug.LogWarning("ovrp_GetCameraDeviceDepthFramePixels return " + result);
				return null;
			}
			if (rowPitch != size.w * 4)
			{
				//Debug.LogWarning(string.Format("RowPitch mismatch, expected {0}, get {1}", size.w * 4, rowPitch));
				return null;
			}
			if (!cachedCameraDepthTexture || cachedCameraDepthTexture.width != size.w || cachedCameraDepthTexture.height != size.h)
			{
				cachedCameraDepthTexture = new Texture2D(size.w, size.h, TextureFormat.RFloat, false);
				cachedCameraDepthTexture.filterMode = FilterMode.Point;
			}
			cachedCameraDepthTexture.LoadRawTextureData(depthData, rowPitch * size.h);
			cachedCameraDepthTexture.Apply();
			return cachedCameraDepthTexture;
		}
		else
		{
			return null;
		}
#endif
	}

#if !OVRPLUGIN_UNSUPPORTED_PLATFORM
	private static Texture2D cachedCameraDepthConfidenceTexture = null;
#endif
	public static Texture2D GetCameraDeviceDepthConfidenceTexture(CameraDevice cameraDevice)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return null;
#else
		if (version >= OVRP_1_17_0.version)
		{
			Sizei size = new Sizei();
			Result result = OVRP_1_17_0.ovrp_GetCameraDeviceDepthFrameSize(cameraDevice, out size);
			if (result != Result.Success)
			{
				//Debug.LogWarning("ovrp_GetCameraDeviceDepthFrameSize return " + result);
				return null;
			}
			IntPtr confidenceData;
			int rowPitch;
			result = OVRP_1_17_0.ovrp_GetCameraDeviceDepthConfidencePixels(cameraDevice, out confidenceData, out rowPitch);
			if (result != Result.Success)
			{
				//Debug.LogWarning("ovrp_GetCameraDeviceDepthConfidencePixels return " + result);
				return null;
			}
			if (rowPitch != size.w * 4)
			{
				//Debug.LogWarning(string.Format("RowPitch mismatch, expected {0}, get {1}", size.w * 4, rowPitch));
				return null;
			}
			if (!cachedCameraDepthConfidenceTexture || cachedCameraDepthConfidenceTexture.width != size.w || cachedCameraDepthConfidenceTexture.height != size.h)
			{
				cachedCameraDepthConfidenceTexture = new Texture2D(size.w, size.h, TextureFormat.RFloat, false);
			}
			cachedCameraDepthConfidenceTexture.LoadRawTextureData(confidenceData, rowPitch * size.h);
			cachedCameraDepthConfidenceTexture.Apply();
			return cachedCameraDepthConfidenceTexture;
		}
		else
		{
			return null;
		}
#endif
	}
#endif

	private static bool foveatedRenderingSupported
	{
		get
		{
			return fixedFoveatedRenderingSupported || eyeTrackedFoveatedRenderingSupported;
		}
	}

	public static bool eyeTrackedFoveatedRenderingSupported
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_78_0.version)
			{
				Bool supported;
				Result result = OVRP_1_78_0.ovrp_GetFoveationEyeTrackedSupported(out supported);
				if (result != Result.Success)
				{
					//Debug.LogWarning("ovrp_GetFoveationEyeTrackedSupported return " + result);
				}
				return supported == Bool.True;
			}
			else
			{
				return false;
			}
#endif
		}
	}

	public static bool eyeTrackedFoveatedRenderingEnabled
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_78_0.version && eyeTrackedFoveatedRenderingSupported)
			{
				Bool enabled;
				Result result = OVRP_1_78_0.ovrp_GetFoveationEyeTracked(out enabled);
				if (result != Result.Success)
				{
					//Debug.LogWarning("ovrp_GetFoveationEyeTracked return " + result);
				}
				return enabled == Bool.True;
			}
			else
			{
				return false;
			}
#endif
		}
		set
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			if (version >= OVRP_1_78_0.version && eyeTrackedFoveatedRenderingSupported)
			{
				Result result = OVRP_1_78_0.ovrp_SetFoveationEyeTracked(value ? Bool.True : Bool.False);
				if (result != Result.Success)
				{
					//Debug.LogWarning("ovrp_SetFoveationEyeTracked return " + result);
				}
			}
#endif
		}
	}


	public static bool fixedFoveatedRenderingSupported
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_21_0.version)
			{
				Bool supported;
				Result result = OVRP_1_21_0.ovrp_GetTiledMultiResSupported(out supported);
				if (result == Result.Success)
				{
					return supported == Bool.True;
				}
				else
				{
					//Debug.LogWarning("ovrp_GetTiledMultiResSupported return " + result);
					return false;
				}
			}
			else
			{
				return false;
			}
#endif
		}
	}

	public static FoveatedRenderingLevel foveatedRenderingLevel
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return FoveatedRenderingLevel.Off;
#else
			if (version >= OVRP_1_21_0.version && foveatedRenderingSupported)
			{
				FoveatedRenderingLevel level;
				Result result = OVRP_1_21_0.ovrp_GetTiledMultiResLevel(out level);
				if (result != Result.Success)
				{
					//Debug.LogWarning("ovrp_GetTiledMultiResLevel return " + result);
				}
				return level;
			}
			else
			{
				return FoveatedRenderingLevel.Off;
			}
#endif
		}
		set
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			if (version >= OVRP_1_21_0.version && foveatedRenderingSupported)
			{
				Result result = OVRP_1_21_0.ovrp_SetTiledMultiResLevel(value);
				if (result != Result.Success)
				{
					//Debug.LogWarning("ovrp_SetTiledMultiResLevel return " + result);
				}
			}
#endif
		}
	}

	[Obsolete("Please use foveatedRenderingLevel instead", false)]
	public static FixedFoveatedRenderingLevel fixedFoveatedRenderingLevel
	{
		get
		{
			return (FixedFoveatedRenderingLevel)foveatedRenderingLevel;
		}
		set
		{
			foveatedRenderingLevel = (FoveatedRenderingLevel)value;
		}
	}

	public static bool useDynamicFoveatedRendering
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_46_0.version && foveatedRenderingSupported)
			{
				Bool isDynamic = Bool.False;
				Result result = OVRP_1_46_0.ovrp_GetTiledMultiResDynamic(out isDynamic);
				if (result != Result.Success)
				{
					//Debug.LogWarning("ovrp_GetTiledMultiResDynamic return " + result);
				}
				return isDynamic != Bool.False;
			}
			else
			{
				return false;
			}
#endif
		}
		set
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			if (version >= OVRP_1_46_0.version && foveatedRenderingSupported)
			{
				Result result = OVRP_1_46_0.ovrp_SetTiledMultiResDynamic(value ? Bool.True : Bool.False);
				if (result != Result.Success)
				{
					//Debug.LogWarning("ovrp_SetTiledMultiResDynamic return " + result);
				}
			}
#endif
		}
	}


	[Obsolete("Please use useDynamicFoveatedRendering instead", false)]
	public static bool useDynamicFixedFoveatedRendering
	{
		get
		{
			return useDynamicFoveatedRendering;
		}
		set
		{
			useDynamicFoveatedRendering = value;
		}
	}

	[Obsolete("Please use fixedFoveatedRenderingSupported instead", false)]
	public static bool tiledMultiResSupported
	{
		get
		{
			return fixedFoveatedRenderingSupported;
		}
	}

	[Obsolete("Please use foveatedRenderingLevel instead", false)]
	public static TiledMultiResLevel tiledMultiResLevel
	{
		get
		{
			return (TiledMultiResLevel)foveatedRenderingLevel;
		}
		set
		{
			foveatedRenderingLevel = (FoveatedRenderingLevel)value;
		}
	}

	public static bool gpuUtilSupported
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_21_0.version)
			{
				Bool supported;
				Result result = OVRP_1_21_0.ovrp_GetGPUUtilSupported(out supported);
				if (result == Result.Success)
				{
					return supported == Bool.True;
				}
				else
				{
					//Debug.LogWarning("ovrp_GetGPUUtilSupported return " + result);
					return false;
				}
			}
			else
			{
				return false;
			}
#endif
		}
	}

	public static float gpuUtilLevel
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0.0f;
#else
			if (version >= OVRP_1_21_0.version && gpuUtilSupported)
			{
				float level;
				Result result = OVRP_1_21_0.ovrp_GetGPUUtilLevel(out level);
				if (result == Result.Success)
				{
					return level;
				}
				else
				{
					//Debug.LogWarning("ovrp_GetGPUUtilLevel return " + result);
					return 0.0f;
				}
			}
			else
			{
				return 0.0f;
			}
#endif
		}
	}

#if !OVRPLUGIN_UNSUPPORTED_PLATFORM
	private static OVRNativeBuffer _nativeSystemDisplayFrequenciesAvailable = null;
	private static float[] _cachedSystemDisplayFrequenciesAvailable = null;
#endif

	public static float[] systemDisplayFrequenciesAvailable
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return new float[0];
#else
			if (_cachedSystemDisplayFrequenciesAvailable == null)
			{
				_cachedSystemDisplayFrequenciesAvailable = new float[0];

				if (version >= OVRP_1_21_0.version)
				{
					int numFrequencies = 0;
					Result result = OVRP_1_21_0.ovrp_GetSystemDisplayAvailableFrequencies(IntPtr.Zero, ref numFrequencies);
					if (result == Result.Success)
					{
						if (numFrequencies > 0)
						{
							int maxNumElements = numFrequencies;
							_nativeSystemDisplayFrequenciesAvailable = new OVRNativeBuffer(sizeof(float) * maxNumElements);
							result = OVRP_1_21_0.ovrp_GetSystemDisplayAvailableFrequencies(_nativeSystemDisplayFrequenciesAvailable.GetPointer(), ref numFrequencies);
							if (result == Result.Success)
							{
								int numElementsToCopy = (numFrequencies <= maxNumElements) ? numFrequencies : maxNumElements;
								if (numElementsToCopy > 0)
								{
									_cachedSystemDisplayFrequenciesAvailable = new float[numElementsToCopy];
									Marshal.Copy(_nativeSystemDisplayFrequenciesAvailable.GetPointer(), _cachedSystemDisplayFrequenciesAvailable, 0, numElementsToCopy);
								}
							}
						}
					}
				}
			}

			return _cachedSystemDisplayFrequenciesAvailable;
#endif
		}
	}

	public static float systemDisplayFrequency
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0.0f;
#else
			if (version >= OVRP_1_21_0.version)
			{
				float displayFrequency;
				Result result = OVRP_1_21_0.ovrp_GetSystemDisplayFrequency2(out displayFrequency);
				if (result == Result.Success)
				{
					return displayFrequency;
				}

				return 0.0f;
			}
			else if (version >= OVRP_1_1_0.version)
			{
				return OVRP_1_1_0.ovrp_GetSystemDisplayFrequency();
			}
			else
			{
				return 0.0f;
			}
#endif
		}
		set
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			if (version >= OVRP_1_21_0.version)
			{
				OVRP_1_21_0.ovrp_SetSystemDisplayFrequency(value);
			}
#endif
		}
	}

	public static bool eyeFovPremultipliedAlphaModeEnabled
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return true;
#else
			Bool isEnabled = Bool.True;
			if (version >= OVRP_1_57_0.version)
			{
				OVRP_1_57_0.ovrp_GetEyeFovPremultipliedAlphaMode(ref isEnabled);
			}
			return isEnabled == Bool.True ? true : false;
#endif
		}
		set
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return;
#else
			if (version >= OVRP_1_57_0.version)
			{
				OVRP_1_57_0.ovrp_SetEyeFovPremultipliedAlphaMode(ToBool(value));
			}
#endif
		}
	}

	public static bool GetNodeFrustum2(Node nodeId, out Frustumf2 frustum)
	{
		frustum = default(Frustumf2);

#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_15_0.version)
		{
			Result result = OVRP_1_15_0.ovrp_GetNodeFrustum2(nodeId, out frustum);
			if (result != Result.Success)
			{
				return false;
			}
			else
			{
				return true;
			}
		}
		else
		{
			return false;
		}
#endif
	}


	public static bool AsymmetricFovEnabled
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_21_0.version)
			{
				Bool asymmetricFovEnabled = Bool.False;
				Result result = OVRP_1_21_0.ovrp_GetAppAsymmetricFov(out asymmetricFovEnabled);

				if (result != Result.Success)
				{
					return false;
				}
				else
				{
					return asymmetricFovEnabled == Bool.True;
				}
			}
			else
			{
				return false;
			}
#endif
		}
	}

	public static bool EyeTextureArrayEnabled
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_15_0.version)
			{
				Bool enabled = Bool.False;
				enabled = OVRP_1_15_0.ovrp_GetEyeTextureArrayEnabled();
				return enabled == Bool.True;
			}
			else
			{
				return false;
			}
#endif
		}
	}

	public static bool localDimmingSupported
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
            return false;
#else
			if (version >= OVRP_1_78_0.version)
			{
				Bool supported = Bool.False;
				Result result = OVRP_1_78_0.ovrp_GetLocalDimmingSupported(out supported);
				if (result == Result.Success)
				{
					return supported == Bool.True;
				}
				return false;
			}
			return false;
#endif
		}
	}

	public static bool localDimming
	{
		get
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
            return false;
#else
			if (version >= OVRP_1_78_0.version && localDimmingSupported)
			{
				Bool LocalDimmingEnabled = Bool.False;
				Result result = OVRP_1_78_0.ovrp_GetLocalDimming(out LocalDimmingEnabled);
				if (result == Result.Success)
				{
					return LocalDimmingEnabled == Bool.True ? true : false;
				}
			}
			return false;
#endif
		}
		set
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
            return;
#else
			if (version >= OVRP_1_78_0.version && localDimmingSupported)
			{
				Result result = OVRP_1_78_0.ovrp_SetLocalDimming(value == true ? Bool.True : Bool.False);
			}
#endif
		}
	}

	public static Handedness GetDominantHand()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return Handedness.Unsupported;
#else
		Handedness dominantHand;

		if (version >= OVRP_1_28_0.version && OVRP_1_28_0.ovrp_GetDominantHand(out dominantHand) == Result.Success)
		{
			return dominantHand;
		}

		return Handedness.Unsupported;
#endif
	}

	public static bool SendEvent(string name, string param = "", string source = "")
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_30_0.version)
		{
			return OVRP_1_30_0.ovrp_SendEvent2(name, param, source.Length == 0 ? "integration": source) == Result.Success;
		}
		else if (version >= OVRP_1_28_0.version)
		{
			return OVRP_1_28_0.ovrp_SendEvent(name, param) == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool SetHeadPoseModifier(ref Quatf relativeRotation, ref Vector3f relativeTranslation)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_29_0.version)
		{
			return OVRP_1_29_0.ovrp_SetHeadPoseModifier(ref relativeRotation, ref relativeTranslation) == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool GetHeadPoseModifier(out Quatf relativeRotation, out Vector3f relativeTranslation)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		relativeRotation = Quatf.identity;
		relativeTranslation = Vector3f.zero;
		return false;
#else
		if (version >= OVRP_1_29_0.version)
		{
			return OVRP_1_29_0.ovrp_GetHeadPoseModifier(out relativeRotation, out relativeTranslation) == Result.Success;
		}
		else
		{
			relativeRotation = Quatf.identity;
			relativeTranslation = Vector3f.zero;
			return false;
		}
#endif
	}

	public static bool IsPerfMetricsSupported(PerfMetrics perfMetrics)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_30_0.version)
		{
			Bool isSupported;
			Result result = OVRP_1_30_0.ovrp_IsPerfMetricsSupported(perfMetrics, out isSupported);
			if (result == Result.Success)
			{
				return isSupported == Bool.True;
			}
			else
			{
				return false;
			}
		}
		else
		{
			return false;
		}
#endif
	}

	public static float? GetPerfMetricsFloat(PerfMetrics perfMetrics)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return null;
#else
		if (version >= OVRP_1_30_0.version)
		{
			float value;
			Result result = OVRP_1_30_0.ovrp_GetPerfMetricsFloat(perfMetrics, out value);
			if (result == Result.Success)
			{
				return value;
			}
			else
			{
				return null;
			}
		}
		else
		{
			return null;
		}
#endif
	}

	public static int? GetPerfMetricsInt(PerfMetrics perfMetrics)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return null;
#else
		if (version >= OVRP_1_30_0.version)
		{
			int value;
			Result result = OVRP_1_30_0.ovrp_GetPerfMetricsInt(perfMetrics, out value);
			if (result == Result.Success)
			{
				return value;
			}
			else
			{
				return null;
			}
		}
		else
		{
			return null;
		}
#endif
	}

	public static double GetTimeInSeconds()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return 0.0;
#else
		if (version >= OVRP_1_31_0.version)
		{
			double value;
			Result result = OVRP_1_31_0.ovrp_GetTimeInSeconds(out value);
			if (result == Result.Success)
			{
				return value;
			}
			else
			{
				return 0.0;
			}
		}
		else
		{
			return 0.0;
		}
#endif
	}

	public static bool SetColorScaleAndOffset(Vector4 colorScale, Vector4 colorOffset, bool applyToAllLayers)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
#if USING_XR_SDK_OCULUS
		OculusXRPlugin.SetColorScale(colorScale.x, colorScale.y, colorScale.z, colorScale.w);
		OculusXRPlugin.SetColorOffset(colorOffset.x, colorOffset.y, colorOffset.z, colorOffset.w);
		return true;
#elif REQUIRES_XR_SDK
		return false;
#else
		if (version >= OVRP_1_31_0.version)
		{
			Bool ovrpApplyToAllLayers = applyToAllLayers ? Bool.True : Bool.False;
			return OVRP_1_31_0.ovrp_SetColorScaleAndOffset(colorScale, colorOffset, ovrpApplyToAllLayers) == Result.Success;
		}
		else
		{
			return false;
		}
#endif
#endif
	}

	public static bool AddCustomMetadata(string name, string param = "")
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_32_0.version)
		{
			return OVRP_1_32_0.ovrp_AddCustomMetadata(name, param) == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public class Media
	{
		public enum MrcActivationMode
		{
			Automatic = 0,
			Disabled = 1,
			EnumSize = 0x7fffffff
		}

		public enum PlatformCameraMode
		{
			Disabled = -1,
			Initialized = 0,
			UserControlled = 1,
			SmartNavigated = 2,
			StabilizedPoV = 3,
			RemoteDroneControlled = 4,
			RemoteSpatialMapped = 5,
			SpectatorMode = 6,
			MobileMRC = 7,
			EnumSize = 0x7fffffff
		}

		public enum InputVideoBufferType
		{
			Memory = 0,
			TextureHandle = 1,
			EnumSize = 0x7fffffff
		}

		public static bool Initialize()
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_38_0.version)
			{
				return OVRP_1_38_0.ovrp_Media_Initialize() == Result.Success;
			}
			else
			{
				return false;
			}
#endif
		}

		public static bool Shutdown()
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_38_0.version)
			{
				return OVRP_1_38_0.ovrp_Media_Shutdown() == Result.Success;
			}
			else
			{
				return false;
			}
#endif
		}

		public static bool GetInitialized()
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_38_0.version)
			{
				Bool initialized = Bool.False;
				Result result = OVRP_1_38_0.ovrp_Media_GetInitialized(out initialized);
				if (result == Result.Success)
				{
					return initialized == Bool.True ? true : false;
				}
				else
				{
					return false;
				}
			}
			else
			{
				return false;
			}
#endif
		}

		public static bool Update()
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_38_0.version)
			{
				return OVRP_1_38_0.ovrp_Media_Update() == Result.Success;
			}
			else
			{
				return false;
			}
#endif
		}

		public static MrcActivationMode GetMrcActivationMode()
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return MrcActivationMode.Disabled;
#else
			if (version >= OVRP_1_38_0.version)
			{
				MrcActivationMode mode;
				if (OVRP_1_38_0.ovrp_Media_GetMrcActivationMode(out mode) == Result.Success)
				{
					return mode;
				}
				else
				{
					return default(MrcActivationMode);
				}
			}
			else
			{
				return default(MrcActivationMode);
			}
#endif
		}

		public static bool SetMrcActivationMode(MrcActivationMode mode)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_38_0.version)
			{
				return OVRP_1_38_0.ovrp_Media_SetMrcActivationMode(mode) == Result.Success;
			}
			else
			{
				return false;
			}
#endif
		}

		public static bool SetPlatformInitialized()
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_54_0.version)
			{
				return OVRP_1_54_0.ovrp_Media_SetPlatformInitialized() == Result.Success;
			}
			else
			{
				return false;
			}
#endif
		}

		public static PlatformCameraMode GetPlatformCameraMode()
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return PlatformCameraMode.Disabled;
#else
			if (version >= OVRP_1_57_0.version)
			{
				PlatformCameraMode mode;
				if (OVRP_1_57_0.ovrp_Media_GetPlatformCameraMode(out mode) == Result.Success)
				{
					return mode;
				}
				else
				{
					return default(PlatformCameraMode);
				}
			}
			else
			{
				return default(PlatformCameraMode);
			}
#endif
		}

		public static bool SetPlatformCameraMode(PlatformCameraMode mode)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_57_0.version)
			{
				return OVRP_1_57_0.ovrp_Media_SetPlatformCameraMode(mode) == Result.Success;
			}
			else
			{
				return false;
			}
#endif
		}

		public static bool IsMrcEnabled()
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_38_0.version)
			{
				Bool b;
				if (OVRP_1_38_0.ovrp_Media_IsMrcEnabled(out b) == Result.Success)
				{
					return b == Bool.True ? true : false;
				}
				else
				{
					return false;
				}
			}
			else
			{
				return false;
			}
#endif
		}

		public static bool IsMrcActivated()
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_38_0.version)
			{
				Bool b;
				if (OVRP_1_38_0.ovrp_Media_IsMrcActivated(out b) == Result.Success)
				{
					return b == Bool.True ? true : false;
				}
				else
				{
					return false;
				}
			}
			else
			{
				return false;
			}
#endif
		}

		public static bool UseMrcDebugCamera()
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_38_0.version)
			{
				Bool b;
				if (OVRP_1_38_0.ovrp_Media_UseMrcDebugCamera(out b) == Result.Success)
				{
					return b == Bool.True ? true : false;
				}
				else
				{
					return false;
				}
			}
			else
			{
				return false;
			}
#endif
		}

		public static bool SetMrcInputVideoBufferType(InputVideoBufferType videoBufferType)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_38_0.version)
			{
				if (OVRP_1_38_0.ovrp_Media_SetMrcInputVideoBufferType(videoBufferType) == Result.Success)
				{
					return true;
				}
				else
				{
					return false;
				}
			}
			else
			{
				return false;
			}
#endif
		}

		public static InputVideoBufferType GetMrcInputVideoBufferType()
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return InputVideoBufferType.Memory;
#else
			if (version >= OVRP_1_38_0.version)
			{
				InputVideoBufferType videoBufferType = InputVideoBufferType.Memory;
				OVRP_1_38_0.ovrp_Media_GetMrcInputVideoBufferType(ref videoBufferType);
				return videoBufferType;
			}
			else
			{
				return InputVideoBufferType.Memory;
			}
#endif
		}

		public static bool SetMrcFrameSize(int frameWidth, int frameHeight)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_38_0.version)
			{
				if (OVRP_1_38_0.ovrp_Media_SetMrcFrameSize(frameWidth, frameHeight) == Result.Success)
				{
					return true;
				}
				else
				{
					return false;
				}
			}
			else
			{
				return false;
			}
#endif
		}

		public static void GetMrcFrameSize(out int frameWidth, out int frameHeight)
		{

			frameWidth = -1;
			frameHeight = -1;
#if !OVRPLUGIN_UNSUPPORTED_PLATFORM
			if (version >= OVRP_1_38_0.version)
			{
				OVRP_1_38_0.ovrp_Media_GetMrcFrameSize(ref frameWidth, ref frameHeight);
			}
#endif
		}


		public static bool SetMrcAudioSampleRate(int sampleRate)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_38_0.version)
			{
				if (OVRP_1_38_0.ovrp_Media_SetMrcAudioSampleRate(sampleRate) == Result.Success)
				{
					return true;
				}
				else
				{
					return false;
				}
			}
			else
			{
				return false;
			}
#endif
		}

		public static int GetMrcAudioSampleRate()
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0;
#else
			int sampleRate = 0;
			if (version >= OVRP_1_38_0.version)
			{
				OVRP_1_38_0.ovrp_Media_GetMrcAudioSampleRate(ref sampleRate);
			}
			return sampleRate;
#endif
		}

		public static bool SetMrcFrameImageFlipped(bool imageFlipped)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_38_0.version)
			{
				Bool flipped = imageFlipped ? Bool.True : Bool.False;
				if (OVRP_1_38_0.ovrp_Media_SetMrcFrameImageFlipped(flipped) == Result.Success)
				{
					return true;
				}
				else
				{
					return false;
				}
			}
			else
			{
				return false;
			}
#endif
		}

		public static bool GetMrcFrameImageFlipped()
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			Bool flipped = 0;
			if (version >= OVRP_1_38_0.version)
			{
				OVRP_1_38_0.ovrp_Media_GetMrcFrameImageFlipped(ref flipped);
			}
			return flipped == Bool.True ? true : false;
#endif
		}

		public static bool EncodeMrcFrame(System.IntPtr textureHandle, System.IntPtr fgTextureHandle, float[] audioData, int audioFrames, int audioChannels, double timestamp, double poseTime, ref int outSyncId)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_38_0.version)
			{
				if (textureHandle == System.IntPtr.Zero)
				{
					Debug.LogError("EncodeMrcFrame: textureHandle is null");
					return false;
				}
				InputVideoBufferType videoBufferType = GetMrcInputVideoBufferType();
				if (videoBufferType != InputVideoBufferType.TextureHandle)
				{
					Debug.LogError("EncodeMrcFrame: videoBufferType mismatch");
					return false;
				}
				GCHandle pinnedAudioData = new GCHandle();
				IntPtr audioDataPtr = IntPtr.Zero;
				int audioDataLen = 0;
				if (audioData != null)
				{
					pinnedAudioData = GCHandle.Alloc(audioData, GCHandleType.Pinned);
					audioDataPtr = pinnedAudioData.AddrOfPinnedObject();
					audioDataLen = audioFrames * 4;
				}

				Result result;
				if (fgTextureHandle == System.IntPtr.Zero)
				{
					if (version >= OVRP_1_49_0.version)
					{
						result = OVRP_1_49_0.ovrp_Media_EncodeMrcFrameWithPoseTime(textureHandle, audioDataPtr, audioDataLen, audioChannels, timestamp, poseTime, ref outSyncId);
					}
					else
					{
						result = OVRP_1_38_0.ovrp_Media_EncodeMrcFrame(textureHandle, audioDataPtr, audioDataLen, audioChannels, timestamp, ref outSyncId);
					}
				}
				else
				{
					if (version >= OVRP_1_49_0.version)
					{
						result = OVRP_1_49_0.ovrp_Media_EncodeMrcFrameDualTexturesWithPoseTime(textureHandle, fgTextureHandle, audioDataPtr, audioDataLen, audioChannels, timestamp, poseTime, ref outSyncId);
					}
					else
					{
						result = OVRP_1_38_0.ovrp_Media_EncodeMrcFrameWithDualTextures(textureHandle, fgTextureHandle, audioDataPtr, audioDataLen, audioChannels, timestamp, ref outSyncId);
					}
				}

				if (audioData != null)
				{
					pinnedAudioData.Free();
				}
				return result == Result.Success;
			}
			else
			{
				return false;
			}
#endif
		}

#if !OVRPLUGIN_UNSUPPORTED_PLATFORM
		static Texture2D cachedTexture = null;
#endif
		public static bool EncodeMrcFrame(RenderTexture frame, float[] audioData, int audioFrames, int audioChannels, double timestamp, double poseTime, ref int outSyncId)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_38_0.version)
			{
				if (frame == null)
				{
					Debug.LogError("EncodeMrcFrame: frame is null");
					return false;
				}
				InputVideoBufferType videoBufferType = GetMrcInputVideoBufferType();
				if (videoBufferType != InputVideoBufferType.Memory)
				{
					Debug.LogError("EncodeMrcFrame: videoBufferType mismatch");
					return false;
				}

				GCHandle pinnedArray = new GCHandle();
				IntPtr pointer = IntPtr.Zero;
				if (cachedTexture == null || cachedTexture.width != frame.width || cachedTexture.height != frame.height)
				{
					cachedTexture = new Texture2D(frame.width, frame.height, TextureFormat.ARGB32, false);
				}
				RenderTexture lastActive = RenderTexture.active;
				RenderTexture.active = frame;
				cachedTexture.ReadPixels(new Rect(0, 0, frame.width, frame.height), 0, 0);
				RenderTexture.active = lastActive;
				Color32[] bytes = cachedTexture.GetPixels32(0);
				pinnedArray = GCHandle.Alloc(bytes, GCHandleType.Pinned);
				pointer = pinnedArray.AddrOfPinnedObject();

				GCHandle pinnedAudioData = new GCHandle();
				IntPtr audioDataPtr = IntPtr.Zero;
				int audioDataLen = 0;
				if (audioData != null)
				{
					pinnedAudioData = GCHandle.Alloc(audioData, GCHandleType.Pinned);
					audioDataPtr = pinnedAudioData.AddrOfPinnedObject();
					audioDataLen = audioFrames * 4;
				}
				Result result;
				if (version >= OVRP_1_49_0.version)
				{
					result = OVRP_1_49_0.ovrp_Media_EncodeMrcFrameWithPoseTime(pointer, audioDataPtr, audioDataLen, audioChannels, timestamp, poseTime, ref outSyncId);
				}
				else
				{
					result = OVRP_1_38_0.ovrp_Media_EncodeMrcFrame(pointer, audioDataPtr, audioDataLen, audioChannels, timestamp, ref outSyncId);
				}
				pinnedArray.Free();
				if (audioData != null)
				{
					pinnedAudioData.Free();
				}
				return result == Result.Success;
			}
			else
			{
				return false;
			}
#endif
		}

		public static bool SyncMrcFrame(int syncId)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_38_0.version)
			{
				return OVRP_1_38_0.ovrp_Media_SyncMrcFrame(syncId) == Result.Success;
			}
			else
			{
				return false;
			}
#endif
		}

		public static bool SetAvailableQueueIndexVulkan(uint queueIndexVk)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_45_0.version)
			{
				return OVRP_1_45_0.ovrp_Media_SetAvailableQueueIndexVulkan(queueIndexVk) == Result.Success;
			}
			else
			{
				return false;
			}
#endif
		}

		public static bool SetMrcHeadsetControllerPose(Posef headsetPose, Posef leftControllerPose, Posef rightControllerPose)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
			if (version >= OVRP_1_49_0.version)
			{
				Result res = OVRP_1_49_0.ovrp_Media_SetHeadsetControllerPose(headsetPose, leftControllerPose, rightControllerPose);
				return res == Result.Success;
			}
			else
			{
				return false;
			}
#endif
		}

		public static bool IsCastingToRemoteClient()
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_66_0.version)
			{
				Bool isCasting = Bool.False;
				Result result = OVRP_1_66_0.ovrp_Media_IsCastingToRemoteClient(out isCasting);
				if (result == Result.Success)
				{
					return isCasting == Bool.True;
				}
				else
				{
					return false;
				}
			}
			else
			{
				return false;
			}
#endif
		}
	}

	public static bool SetDeveloperMode(Bool active)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if(version >= OVRP_1_38_0.version)
		{
			return OVRP_1_38_0.ovrp_SetDeveloperMode(active) == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	[System.Obsolete("Deprecated. This function will not be supported in OpenXR", false)]
	public static float GetAdaptiveGPUPerformanceScale()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return 1.0f;
#else
		if (version >= OVRP_1_42_0.version)
		{
			float adaptiveScale = 1.0f;
			if (OVRP_1_42_0.ovrp_GetAdaptiveGpuPerformanceScale2(ref adaptiveScale) == Result.Success)
			{
				return adaptiveScale;
			}
			return 1.0f;
		}
		else
		{
			return 1.0f;
		}
#endif
	}

	public static bool GetHandTrackingEnabled()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_44_0.version)
		{
			Bool val = OVRPlugin.Bool.False;
			Result res = OVRP_1_44_0.ovrp_GetHandTrackingEnabled(ref val);
			if (res == Result.Success)
			{
				return val == OVRPlugin.Bool.True;
			}

			return false;
		}
		else
		{
			return false;
		}
#endif
	}

	private static HandStateInternal cachedHandState = new HandStateInternal();
	public static bool GetHandState(Step stepId, Hand hand, ref HandState handState)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (nativeXrApi == XrApi.OpenXR && stepId == Step.Physics)
		{
			Debug.LogWarning("Step.Physics is deprecated when using OpenXR");
			stepId = Step.Render;
		}

		if (version >= OVRP_1_44_0.version)
		{
			Result res = OVRP_1_44_0.ovrp_GetHandState(stepId, hand, out cachedHandState);
			if (res == Result.Success)
			{
				// attempt to avoid allocations if client provides appropriately pre-initialized HandState
				if (handState.BoneRotations == null || handState.BoneRotations.Length != (int)SkeletonConstants.MaxHandBones)
				{
					handState.BoneRotations = new Quatf[(int)SkeletonConstants.MaxHandBones];
				}
				if (handState.PinchStrength == null || handState.PinchStrength.Length != (int)HandFinger.Max)
				{
					handState.PinchStrength = new float[(int)HandFinger.Max];
				}
				if (handState.FingerConfidences == null || handState.FingerConfidences.Length != (int)HandFinger.Max)
				{
					handState.FingerConfidences = new TrackingConfidence[(int)HandFinger.Max];
				}

				// unrolling the arrays is necessary to avoid per-frame allocations during marshaling
				handState.Status = cachedHandState.Status;
				handState.RootPose = cachedHandState.RootPose;
				handState.BoneRotations[0] = cachedHandState.BoneRotations_0;
				handState.BoneRotations[1] = cachedHandState.BoneRotations_1;
				handState.BoneRotations[2] = cachedHandState.BoneRotations_2;
				handState.BoneRotations[3] = cachedHandState.BoneRotations_3;
				handState.BoneRotations[4] = cachedHandState.BoneRotations_4;
				handState.BoneRotations[5] = cachedHandState.BoneRotations_5;
				handState.BoneRotations[6] = cachedHandState.BoneRotations_6;
				handState.BoneRotations[7] = cachedHandState.BoneRotations_7;
				handState.BoneRotations[8] = cachedHandState.BoneRotations_8;
				handState.BoneRotations[9] = cachedHandState.BoneRotations_9;
				handState.BoneRotations[10] = cachedHandState.BoneRotations_10;
				handState.BoneRotations[11] = cachedHandState.BoneRotations_11;
				handState.BoneRotations[12] = cachedHandState.BoneRotations_12;
				handState.BoneRotations[13] = cachedHandState.BoneRotations_13;
				handState.BoneRotations[14] = cachedHandState.BoneRotations_14;
				handState.BoneRotations[15] = cachedHandState.BoneRotations_15;
				handState.BoneRotations[16] = cachedHandState.BoneRotations_16;
				handState.BoneRotations[17] = cachedHandState.BoneRotations_17;
				handState.BoneRotations[18] = cachedHandState.BoneRotations_18;
				handState.BoneRotations[19] = cachedHandState.BoneRotations_19;
				handState.BoneRotations[20] = cachedHandState.BoneRotations_20;
				handState.BoneRotations[21] = cachedHandState.BoneRotations_21;
				handState.BoneRotations[22] = cachedHandState.BoneRotations_22;
				handState.BoneRotations[23] = cachedHandState.BoneRotations_23;
				handState.Pinches = cachedHandState.Pinches;
				handState.PinchStrength[0] = cachedHandState.PinchStrength_0;
				handState.PinchStrength[1] = cachedHandState.PinchStrength_1;
				handState.PinchStrength[2] = cachedHandState.PinchStrength_2;
				handState.PinchStrength[3] = cachedHandState.PinchStrength_3;
				handState.PinchStrength[4] = cachedHandState.PinchStrength_4;
				handState.PointerPose = cachedHandState.PointerPose;
				handState.HandScale = cachedHandState.HandScale;
				handState.HandConfidence = cachedHandState.HandConfidence;
				handState.FingerConfidences[0] = cachedHandState.FingerConfidences_0;
				handState.FingerConfidences[1] = cachedHandState.FingerConfidences_1;
				handState.FingerConfidences[2] = cachedHandState.FingerConfidences_2;
				handState.FingerConfidences[3] = cachedHandState.FingerConfidences_3;
				handState.FingerConfidences[4] = cachedHandState.FingerConfidences_4;
				handState.RequestedTimeStamp = cachedHandState.RequestedTimeStamp;
				handState.SampleTimeStamp = cachedHandState.SampleTimeStamp;

				return true;
			}
			else
			{
				return false;
			}
		}
		else
		{
			return false;
		}
#endif
	}


	public static bool GetSkeleton(SkeletonType skeletonType, out Skeleton skeleton)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		skeleton = default(Skeleton);
		return false;
#else
		if (version >= OVRP_1_44_0.version)
		{
			return OVRP_1_44_0.ovrp_GetSkeleton(skeletonType, out skeleton) == Result.Success;
		}
		else
		{
			skeleton = default(Skeleton);
			return false;
		}
#endif
	}

	private static Skeleton cachedSkeleton = new Skeleton();
	private static Skeleton2Internal cachedSkeleton2 = new Skeleton2Internal();
	public static bool GetSkeleton2(SkeletonType skeletonType, ref Skeleton2 skeleton)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_55_0.version)
		{
			Result res = OVRP_1_55_0.ovrp_GetSkeleton2(skeletonType, out cachedSkeleton2);
			if (res == Result.Success)
			{
				if (skeleton.Bones == null || skeleton.Bones.Length != (int)SkeletonConstants.MaxBones)
				{
					skeleton.Bones = new Bone[(int)SkeletonConstants.MaxBones];
				}
				if (skeleton.BoneCapsules == null || skeleton.BoneCapsules.Length != (int)SkeletonConstants.MaxBoneCapsules)
				{
					skeleton.BoneCapsules = new BoneCapsule[(int)SkeletonConstants.MaxBoneCapsules];
				}

				skeleton.Type = cachedSkeleton2.Type;
				skeleton.NumBones = cachedSkeleton2.NumBones;
				skeleton.NumBoneCapsules = cachedSkeleton2.NumBoneCapsules;
				skeleton.Bones[0] = cachedSkeleton2.Bones_0;
				skeleton.Bones[1] = cachedSkeleton2.Bones_1;
				skeleton.Bones[2] = cachedSkeleton2.Bones_2;
				skeleton.Bones[3] = cachedSkeleton2.Bones_3;
				skeleton.Bones[4] = cachedSkeleton2.Bones_4;
				skeleton.Bones[5] = cachedSkeleton2.Bones_5;
				skeleton.Bones[6] = cachedSkeleton2.Bones_6;
				skeleton.Bones[7] = cachedSkeleton2.Bones_7;
				skeleton.Bones[8] = cachedSkeleton2.Bones_8;
				skeleton.Bones[9] = cachedSkeleton2.Bones_9;
				skeleton.Bones[10] = cachedSkeleton2.Bones_10;
				skeleton.Bones[11] = cachedSkeleton2.Bones_11;
				skeleton.Bones[12] = cachedSkeleton2.Bones_12;
				skeleton.Bones[13] = cachedSkeleton2.Bones_13;
				skeleton.Bones[14] = cachedSkeleton2.Bones_14;
				skeleton.Bones[15] = cachedSkeleton2.Bones_15;
				skeleton.Bones[16] = cachedSkeleton2.Bones_16;
				skeleton.Bones[17] = cachedSkeleton2.Bones_17;
				skeleton.Bones[18] = cachedSkeleton2.Bones_18;
				skeleton.Bones[19] = cachedSkeleton2.Bones_19;
				skeleton.Bones[20] = cachedSkeleton2.Bones_20;
				skeleton.Bones[21] = cachedSkeleton2.Bones_21;
				skeleton.Bones[22] = cachedSkeleton2.Bones_22;
				skeleton.Bones[23] = cachedSkeleton2.Bones_23;
				skeleton.Bones[24] = cachedSkeleton2.Bones_24;
				skeleton.Bones[25] = cachedSkeleton2.Bones_25;
				skeleton.Bones[26] = cachedSkeleton2.Bones_26;
				skeleton.Bones[27] = cachedSkeleton2.Bones_27;
				skeleton.Bones[28] = cachedSkeleton2.Bones_28;
				skeleton.Bones[29] = cachedSkeleton2.Bones_29;
				skeleton.Bones[30] = cachedSkeleton2.Bones_30;
				skeleton.Bones[31] = cachedSkeleton2.Bones_31;
				skeleton.Bones[32] = cachedSkeleton2.Bones_32;
				skeleton.Bones[33] = cachedSkeleton2.Bones_33;
				skeleton.Bones[34] = cachedSkeleton2.Bones_34;
				skeleton.Bones[35] = cachedSkeleton2.Bones_35;
				skeleton.Bones[36] = cachedSkeleton2.Bones_36;
				skeleton.Bones[37] = cachedSkeleton2.Bones_37;
				skeleton.Bones[38] = cachedSkeleton2.Bones_38;
				skeleton.Bones[39] = cachedSkeleton2.Bones_39;
				skeleton.Bones[40] = cachedSkeleton2.Bones_40;
				skeleton.Bones[41] = cachedSkeleton2.Bones_41;
				skeleton.Bones[42] = cachedSkeleton2.Bones_42;
				skeleton.Bones[43] = cachedSkeleton2.Bones_43;
				skeleton.Bones[44] = cachedSkeleton2.Bones_44;
				skeleton.Bones[45] = cachedSkeleton2.Bones_45;
				skeleton.Bones[46] = cachedSkeleton2.Bones_46;
				skeleton.Bones[47] = cachedSkeleton2.Bones_47;
				skeleton.Bones[48] = cachedSkeleton2.Bones_48;
				skeleton.Bones[49] = cachedSkeleton2.Bones_49;
				skeleton.Bones[50] = cachedSkeleton2.Bones_50;
				skeleton.Bones[51] = cachedSkeleton2.Bones_51;
				skeleton.Bones[52] = cachedSkeleton2.Bones_52;
				skeleton.Bones[53] = cachedSkeleton2.Bones_53;
				skeleton.Bones[54] = cachedSkeleton2.Bones_54;
				skeleton.Bones[55] = cachedSkeleton2.Bones_55;
				skeleton.Bones[56] = cachedSkeleton2.Bones_56;
				skeleton.Bones[57] = cachedSkeleton2.Bones_57;
				skeleton.Bones[58] = cachedSkeleton2.Bones_58;
				skeleton.Bones[59] = cachedSkeleton2.Bones_59;
				skeleton.Bones[60] = cachedSkeleton2.Bones_60;
				skeleton.Bones[61] = cachedSkeleton2.Bones_61;
				skeleton.Bones[62] = cachedSkeleton2.Bones_62;
				skeleton.Bones[63] = cachedSkeleton2.Bones_63;
				skeleton.Bones[64] = cachedSkeleton2.Bones_64;
				skeleton.Bones[65] = cachedSkeleton2.Bones_65;
				skeleton.Bones[66] = cachedSkeleton2.Bones_66;
				skeleton.Bones[67] = cachedSkeleton2.Bones_67;
				skeleton.Bones[68] = cachedSkeleton2.Bones_68;
				skeleton.Bones[69] = cachedSkeleton2.Bones_69;
				skeleton.BoneCapsules[0] = cachedSkeleton2.BoneCapsules_0;
				skeleton.BoneCapsules[1] = cachedSkeleton2.BoneCapsules_1;
				skeleton.BoneCapsules[2] = cachedSkeleton2.BoneCapsules_2;
				skeleton.BoneCapsules[3] = cachedSkeleton2.BoneCapsules_3;
				skeleton.BoneCapsules[4] = cachedSkeleton2.BoneCapsules_4;
				skeleton.BoneCapsules[5] = cachedSkeleton2.BoneCapsules_5;
				skeleton.BoneCapsules[6] = cachedSkeleton2.BoneCapsules_6;
				skeleton.BoneCapsules[7] = cachedSkeleton2.BoneCapsules_7;
				skeleton.BoneCapsules[8] = cachedSkeleton2.BoneCapsules_8;
				skeleton.BoneCapsules[9] = cachedSkeleton2.BoneCapsules_9;
				skeleton.BoneCapsules[10] = cachedSkeleton2.BoneCapsules_10;
				skeleton.BoneCapsules[11] = cachedSkeleton2.BoneCapsules_11;
				skeleton.BoneCapsules[12] = cachedSkeleton2.BoneCapsules_12;
				skeleton.BoneCapsules[13] = cachedSkeleton2.BoneCapsules_13;
				skeleton.BoneCapsules[14] = cachedSkeleton2.BoneCapsules_14;
				skeleton.BoneCapsules[15] = cachedSkeleton2.BoneCapsules_15;
				skeleton.BoneCapsules[16] = cachedSkeleton2.BoneCapsules_16;
				skeleton.BoneCapsules[17] = cachedSkeleton2.BoneCapsules_17;
				skeleton.BoneCapsules[18] = cachedSkeleton2.BoneCapsules_18;

				return true;
			}
			else
			{
				return false;
			}
		}
		else
		{
			if (GetSkeleton(skeletonType, out cachedSkeleton))
			{
				if (skeleton.Bones == null || skeleton.Bones.Length != (int)SkeletonConstants.MaxBones)
				{
					skeleton.Bones = new Bone[(int)SkeletonConstants.MaxBones];
				}
				if (skeleton.BoneCapsules == null || skeleton.BoneCapsules.Length != (int)SkeletonConstants.MaxBoneCapsules)
				{
					skeleton.BoneCapsules = new BoneCapsule[(int)SkeletonConstants.MaxBoneCapsules];
				}

				skeleton.Type = cachedSkeleton.Type;
				skeleton.NumBones = cachedSkeleton.NumBones;
				skeleton.NumBoneCapsules = cachedSkeleton.NumBoneCapsules;

				for (int i = 0; i < skeleton.NumBones; i++)
				{
					skeleton.Bones[i] = cachedSkeleton.Bones[i];
				}

				for (int i = 0; i < skeleton.NumBoneCapsules; i++)
				{
					skeleton.BoneCapsules[i] = cachedSkeleton.BoneCapsules[i];
				}

				return true;
			}

			return false;
		}
#endif
	}

	/// <summary>
	/// True if body tracking is supported, otherwise false.
	/// </summary>
	public static bool bodyTrackingSupported =>
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		false;
#else
		version >= OVRP_1_78_0.version &&
		OVRP_1_78_0.ovrp_GetBodyTrackingSupported(out var value) == Result.Success &&
		value == Bool.True;
#endif

	/// <summary>
	/// True if body tracking is enabled, otherwise false.
	/// </summary>
	public static bool bodyTrackingEnabled =>
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		false;
#else
		version >= OVRP_1_78_0.version &&
		OVRP_1_78_0.ovrp_GetBodyTrackingEnabled(out var value) == Result.Success &&
		value == Bool.True;
#endif

	/// <summary>
	/// Gets the current <see cref="BodyState"/> associated with body tracking.
	/// </summary>
	/// <remarks>
	/// You can safely provide a zero-initialized <see cref="BodyState"/> as the <paramref name="bodyState"/> parameter.
	/// However, if you call this method frequently (e.g., each frame), consider reusing an existing instance of
	/// <see cref="BodyState"/> to avoid per-frame allocations of the <see cref="BodyState.JointLocations"/> array.
	/// </remarks>
	/// <param name="stepId">The <see cref="Step"/> associated with the <see cref="BodyState"/> to get. Only
	///		<see cref="Step.Render"/> is supported.</param>
	/// <param name="bodyState">On success, <paramref name="bodyState"/> will be populated with the
	///		<see cref="BodyState"/> associated <paramref name="stepId"/>. You can safely pass a zero-initialized
	///		<see cref="BodyState"/>; <see cref="BodyState.JointLocations"/> will be	reallocated	if it is `null` or its
	///		length does not match the number of joint locations.</param>
	/// <returns>Returns `true` if the body state was successfully retrieved. Returns `false` otherwise.</returns>
	public static bool GetBodyState(Step stepId, ref BodyState bodyState)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (nativeXrApi == XrApi.OpenXR && stepId == Step.Physics)
		{
			Debug.LogWarning("Step.Physics is deprecated when using OpenXR");
			stepId = Step.Render;
		}

		if (version < OVRP_1_78_0.version) return false;

		// Avoid per-frame allocations if caller provides appropriately pre-initialized BodyState
		const int jointCount = (int)BoneId.Body_End;
		if (bodyState.JointLocations?.Length != jointCount)
		{
			bodyState.JointLocations = new BodyJointLocation[jointCount];
		}

		var result = OVRP_1_78_0.ovrp_GetBodyState(stepId, -1, out var bodyStateInternal);
		if (result != Result.Success) return false;
		if (bodyStateInternal.IsActive != Bool.True) return false;

		bodyState.Confidence = bodyStateInternal.Confidence;
		bodyState.SkeletonChangedCount = bodyStateInternal.SkeletonChangedCount;
		bodyState.Time = bodyStateInternal.Time;
		bodyState.JointLocations[0] = bodyStateInternal.JointLocation_0;
		bodyState.JointLocations[1] = bodyStateInternal.JointLocation_1;
		bodyState.JointLocations[2] = bodyStateInternal.JointLocation_2;
		bodyState.JointLocations[3] = bodyStateInternal.JointLocation_3;
		bodyState.JointLocations[4] = bodyStateInternal.JointLocation_4;
		bodyState.JointLocations[5] = bodyStateInternal.JointLocation_5;
		bodyState.JointLocations[6] = bodyStateInternal.JointLocation_6;
		bodyState.JointLocations[7] = bodyStateInternal.JointLocation_7;
		bodyState.JointLocations[8] = bodyStateInternal.JointLocation_8;
		bodyState.JointLocations[9] = bodyStateInternal.JointLocation_9;
		bodyState.JointLocations[10] = bodyStateInternal.JointLocation_10;
		bodyState.JointLocations[11] = bodyStateInternal.JointLocation_11;
		bodyState.JointLocations[12] = bodyStateInternal.JointLocation_12;
		bodyState.JointLocations[13] = bodyStateInternal.JointLocation_13;
		bodyState.JointLocations[14] = bodyStateInternal.JointLocation_14;
		bodyState.JointLocations[15] = bodyStateInternal.JointLocation_15;
		bodyState.JointLocations[16] = bodyStateInternal.JointLocation_16;
		bodyState.JointLocations[17] = bodyStateInternal.JointLocation_17;
		bodyState.JointLocations[18] = bodyStateInternal.JointLocation_18;
		bodyState.JointLocations[19] = bodyStateInternal.JointLocation_19;
		bodyState.JointLocations[20] = bodyStateInternal.JointLocation_20;
		bodyState.JointLocations[21] = bodyStateInternal.JointLocation_21;
		bodyState.JointLocations[22] = bodyStateInternal.JointLocation_22;
		bodyState.JointLocations[23] = bodyStateInternal.JointLocation_23;
		bodyState.JointLocations[24] = bodyStateInternal.JointLocation_24;
		bodyState.JointLocations[25] = bodyStateInternal.JointLocation_25;
		bodyState.JointLocations[26] = bodyStateInternal.JointLocation_26;
		bodyState.JointLocations[27] = bodyStateInternal.JointLocation_27;
		bodyState.JointLocations[28] = bodyStateInternal.JointLocation_28;
		bodyState.JointLocations[29] = bodyStateInternal.JointLocation_29;
		bodyState.JointLocations[30] = bodyStateInternal.JointLocation_30;
		bodyState.JointLocations[31] = bodyStateInternal.JointLocation_31;
		bodyState.JointLocations[32] = bodyStateInternal.JointLocation_32;
		bodyState.JointLocations[33] = bodyStateInternal.JointLocation_33;
		bodyState.JointLocations[34] = bodyStateInternal.JointLocation_34;
		bodyState.JointLocations[35] = bodyStateInternal.JointLocation_35;
		bodyState.JointLocations[36] = bodyStateInternal.JointLocation_36;
		bodyState.JointLocations[37] = bodyStateInternal.JointLocation_37;
		bodyState.JointLocations[38] = bodyStateInternal.JointLocation_38;
		bodyState.JointLocations[39] = bodyStateInternal.JointLocation_39;
		bodyState.JointLocations[40] = bodyStateInternal.JointLocation_40;
		bodyState.JointLocations[41] = bodyStateInternal.JointLocation_41;
		bodyState.JointLocations[42] = bodyStateInternal.JointLocation_42;
		bodyState.JointLocations[43] = bodyStateInternal.JointLocation_43;
		bodyState.JointLocations[44] = bodyStateInternal.JointLocation_44;
		bodyState.JointLocations[45] = bodyStateInternal.JointLocation_45;
		bodyState.JointLocations[46] = bodyStateInternal.JointLocation_46;
		bodyState.JointLocations[47] = bodyStateInternal.JointLocation_47;
		bodyState.JointLocations[48] = bodyStateInternal.JointLocation_48;
		bodyState.JointLocations[49] = bodyStateInternal.JointLocation_49;
		bodyState.JointLocations[50] = bodyStateInternal.JointLocation_50;
		bodyState.JointLocations[51] = bodyStateInternal.JointLocation_51;
		bodyState.JointLocations[52] = bodyStateInternal.JointLocation_52;
		bodyState.JointLocations[53] = bodyStateInternal.JointLocation_53;
		bodyState.JointLocations[54] = bodyStateInternal.JointLocation_54;
		bodyState.JointLocations[55] = bodyStateInternal.JointLocation_55;
		bodyState.JointLocations[56] = bodyStateInternal.JointLocation_56;
		bodyState.JointLocations[57] = bodyStateInternal.JointLocation_57;
		bodyState.JointLocations[58] = bodyStateInternal.JointLocation_58;
		bodyState.JointLocations[59] = bodyStateInternal.JointLocation_59;
		bodyState.JointLocations[60] = bodyStateInternal.JointLocation_60;
		bodyState.JointLocations[61] = bodyStateInternal.JointLocation_61;
		bodyState.JointLocations[62] = bodyStateInternal.JointLocation_62;
		bodyState.JointLocations[63] = bodyStateInternal.JointLocation_63;
		bodyState.JointLocations[64] = bodyStateInternal.JointLocation_64;
		bodyState.JointLocations[65] = bodyStateInternal.JointLocation_65;
		bodyState.JointLocations[66] = bodyStateInternal.JointLocation_66;
		bodyState.JointLocations[67] = bodyStateInternal.JointLocation_67;
		bodyState.JointLocations[68] = bodyStateInternal.JointLocation_68;
		bodyState.JointLocations[69] = bodyStateInternal.JointLocation_69;

		return true;
#endif
	}

	public static bool GetMesh(MeshType meshType, out Mesh mesh)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		mesh = new Mesh();
		return false;
#else
		if (version >= OVRP_1_44_0.version)
		{
			mesh = new Mesh();
			int meshSize = Marshal.SizeOf(mesh);
			System.IntPtr meshPtr = Marshal.AllocHGlobal(meshSize);
			Result result = OVRP_1_44_0.ovrp_GetMesh(meshType, meshPtr);
			if (result == Result.Success)
			{
				Marshal.PtrToStructure(meshPtr, mesh);
			}
			Marshal.FreeHGlobal(meshPtr);

			return (result == Result.Success);
		}
		else
		{
			mesh = new Mesh();
			return false;
		}
#endif
	}


	public static bool StartKeyboardTracking(UInt64 trackedKeyboardId)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_68_0.version)
		{
			Result result;
			result = OVRP_1_68_0.ovrp_StartKeyboardTracking(trackedKeyboardId);
			return result == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool StopKeyboardTracking()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_68_0.version)
		{
			Result result;
			result = OVRP_1_68_0.ovrp_StopKeyboardTracking();
			return result == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool GetKeyboardState(Step stepId, out KeyboardState keyboardState)
	{
		keyboardState = default(KeyboardState);

#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_68_0.version)
		{
			Result result;
			result = OVRP_1_68_0.ovrp_GetKeyboardState(stepId, -1, out keyboardState);
			return result == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool GetSystemKeyboardDescription(TrackedKeyboardQueryFlags keyboardQueryFlags, out KeyboardDescription keyboardDescription)
	{
		keyboardDescription = default(KeyboardDescription);

#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_68_0.version)
		{
			Result result;
			result = OVRP_1_68_0.ovrp_GetSystemKeyboardDescription(keyboardQueryFlags, out keyboardDescription);
			return result == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}


// Virtual keyboard calls


	/// <summary>
	/// True if face tracking is enabled, otherwise false.
	/// </summary>
	public static bool faceTrackingEnabled =>
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		false;
#else
		version >= OVRP_1_78_0.version &&
		OVRP_1_78_0.ovrp_GetFaceTrackingEnabled(out var val) == Result.Success &&
		val == Bool.True;
#endif

	/// <summary>
	/// True if face tracking is supported, otherwise false.
	/// </summary>
	public static bool faceTrackingSupported =>
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		false;
#else
		version >= OVRP_1_78_0.version &&
		OVRP_1_78_0.ovrp_GetFaceTrackingSupported(out var value) == Result.Success &&
		value == Bool.True;
#endif //OVRPLUGIN_UNSUPPORTED_PLATFORM

	private static FaceStateInternal cachedFaceState = new FaceStateInternal();
	public static bool GetFaceState(Step stepId, int frameIndex, ref FaceState faceState)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (nativeXrApi == XrApi.OpenXR && stepId == Step.Physics)
		{
			Debug.LogWarning("Step.Physics is deprecated when using OpenXR");
			stepId = Step.Render;
		}

		if (version >= OVRP_1_78_0.version)
		{
			Result res = OVRP_1_78_0.ovrp_GetFaceState(stepId, frameIndex, out cachedFaceState);
			if (res == Result.Success)
			{
				// attempt to avoid allocations if client provides appropriately pre-initialized HandState
				if (faceState.ExpressionWeights == null || faceState.ExpressionWeights.Length != (int)FaceConstants.MaxFaceExpressions)
				{
					faceState.ExpressionWeights = new float[(int)FaceConstants.MaxFaceExpressions];
				}

				if (faceState.ExpressionWeightConfidences == null || faceState.ExpressionWeightConfidences.Length != (int)FaceConstants.MaxFaceRegionConfidences)
				{
					faceState.ExpressionWeightConfidences = new float[(int)FaceConstants.MaxFaceRegionConfidences];
				}

				// unrolling the arrays is necessary to avoid per-frame allocations during marshaling
				faceState.ExpressionWeights[0] = cachedFaceState.ExpressionWeights_0;
				faceState.ExpressionWeights[1] = cachedFaceState.ExpressionWeights_1;
				faceState.ExpressionWeights[2] = cachedFaceState.ExpressionWeights_2;
				faceState.ExpressionWeights[3] = cachedFaceState.ExpressionWeights_3;
				faceState.ExpressionWeights[4] = cachedFaceState.ExpressionWeights_4;
				faceState.ExpressionWeights[5] = cachedFaceState.ExpressionWeights_5;
				faceState.ExpressionWeights[6] = cachedFaceState.ExpressionWeights_6;
				faceState.ExpressionWeights[7] = cachedFaceState.ExpressionWeights_7;
				faceState.ExpressionWeights[8] = cachedFaceState.ExpressionWeights_8;
				faceState.ExpressionWeights[9] = cachedFaceState.ExpressionWeights_9;
				faceState.ExpressionWeights[10] = cachedFaceState.ExpressionWeights_10;
				faceState.ExpressionWeights[11] = cachedFaceState.ExpressionWeights_11;
				faceState.ExpressionWeights[12] = cachedFaceState.ExpressionWeights_12;
				faceState.ExpressionWeights[13] = cachedFaceState.ExpressionWeights_13;
				faceState.ExpressionWeights[14] = cachedFaceState.ExpressionWeights_14;
				faceState.ExpressionWeights[15] = cachedFaceState.ExpressionWeights_15;
				faceState.ExpressionWeights[16] = cachedFaceState.ExpressionWeights_16;
				faceState.ExpressionWeights[17] = cachedFaceState.ExpressionWeights_17;
				faceState.ExpressionWeights[18] = cachedFaceState.ExpressionWeights_18;
				faceState.ExpressionWeights[19] = cachedFaceState.ExpressionWeights_19;
				faceState.ExpressionWeights[20] = cachedFaceState.ExpressionWeights_20;
				faceState.ExpressionWeights[21] = cachedFaceState.ExpressionWeights_21;
				faceState.ExpressionWeights[22] = cachedFaceState.ExpressionWeights_22;
				faceState.ExpressionWeights[23] = cachedFaceState.ExpressionWeights_23;
				faceState.ExpressionWeights[24] = cachedFaceState.ExpressionWeights_24;
				faceState.ExpressionWeights[25] = cachedFaceState.ExpressionWeights_25;
				faceState.ExpressionWeights[26] = cachedFaceState.ExpressionWeights_26;
				faceState.ExpressionWeights[27] = cachedFaceState.ExpressionWeights_27;
				faceState.ExpressionWeights[28] = cachedFaceState.ExpressionWeights_28;
				faceState.ExpressionWeights[29] = cachedFaceState.ExpressionWeights_29;
				faceState.ExpressionWeights[30] = cachedFaceState.ExpressionWeights_30;
				faceState.ExpressionWeights[31] = cachedFaceState.ExpressionWeights_31;
				faceState.ExpressionWeights[32] = cachedFaceState.ExpressionWeights_32;
				faceState.ExpressionWeights[33] = cachedFaceState.ExpressionWeights_33;
				faceState.ExpressionWeights[34] = cachedFaceState.ExpressionWeights_34;
				faceState.ExpressionWeights[35] = cachedFaceState.ExpressionWeights_35;
				faceState.ExpressionWeights[36] = cachedFaceState.ExpressionWeights_36;
				faceState.ExpressionWeights[37] = cachedFaceState.ExpressionWeights_37;
				faceState.ExpressionWeights[38] = cachedFaceState.ExpressionWeights_38;
				faceState.ExpressionWeights[39] = cachedFaceState.ExpressionWeights_39;
				faceState.ExpressionWeights[40] = cachedFaceState.ExpressionWeights_40;
				faceState.ExpressionWeights[41] = cachedFaceState.ExpressionWeights_41;
				faceState.ExpressionWeights[42] = cachedFaceState.ExpressionWeights_42;
				faceState.ExpressionWeights[43] = cachedFaceState.ExpressionWeights_43;
				faceState.ExpressionWeights[44] = cachedFaceState.ExpressionWeights_44;
				faceState.ExpressionWeights[45] = cachedFaceState.ExpressionWeights_45;
				faceState.ExpressionWeights[46] = cachedFaceState.ExpressionWeights_46;
				faceState.ExpressionWeights[47] = cachedFaceState.ExpressionWeights_47;
				faceState.ExpressionWeights[48] = cachedFaceState.ExpressionWeights_48;
				faceState.ExpressionWeights[49] = cachedFaceState.ExpressionWeights_49;
				faceState.ExpressionWeights[50] = cachedFaceState.ExpressionWeights_50;
				faceState.ExpressionWeights[51] = cachedFaceState.ExpressionWeights_51;
				faceState.ExpressionWeights[52] = cachedFaceState.ExpressionWeights_52;
				faceState.ExpressionWeights[53] = cachedFaceState.ExpressionWeights_53;
				faceState.ExpressionWeights[54] = cachedFaceState.ExpressionWeights_54;
				faceState.ExpressionWeights[55] = cachedFaceState.ExpressionWeights_55;
				faceState.ExpressionWeights[56] = cachedFaceState.ExpressionWeights_56;
				faceState.ExpressionWeights[57] = cachedFaceState.ExpressionWeights_57;
				faceState.ExpressionWeights[58] = cachedFaceState.ExpressionWeights_58;
				faceState.ExpressionWeights[59] = cachedFaceState.ExpressionWeights_59;
				faceState.ExpressionWeights[60] = cachedFaceState.ExpressionWeights_60;
				faceState.ExpressionWeights[61] = cachedFaceState.ExpressionWeights_61;
				faceState.ExpressionWeights[62] = cachedFaceState.ExpressionWeights_62;
				faceState.ExpressionWeightConfidences[0] = cachedFaceState.ExpressionWeightConfidences_0;
				faceState.ExpressionWeightConfidences[1] = cachedFaceState.ExpressionWeightConfidences_1;
				faceState.Status = cachedFaceState.Status.ToFaceExpressionStatus();
				faceState.Time = cachedFaceState.Time;

				return true;
			}
			else
			{
				return false;
			}
		}
		else
		{
			return false;
		}
#endif
	}

	/// <summary>
	/// True if eye tracking is enabled, otherwise false.
	/// </summary>
	public static bool eyeTrackingEnabled =>
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		false;
#else
		version >= OVRP_1_78_0.version &&
		OVRP_1_78_0.ovrp_GetEyeTrackingEnabled(out var val) == Result.Success &&
		val == Bool.True;
#endif // OVRPLUGIN_UNSUPPORTED_PLATFORM

	/// <summary>
	/// True if eye tracking is supported, otherwise false.
	/// </summary>
	public static bool eyeTrackingSupported =>
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		false;
#else
		version >= OVRP_1_78_0.version &&
		OVRP_1_78_0.ovrp_GetEyeTrackingSupported(out var val) == Result.Success &&
		val == Bool.True;
#endif // OVRPLUGIN_UNSUPPORTED_PLATFORM

	private static EyeGazesStateInternal cachedEyeGazesState = new EyeGazesStateInternal();
	public static bool GetEyeGazesState(Step stepId, int frameIndex, ref EyeGazesState eyeGazesState)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (nativeXrApi == XrApi.OpenXR && stepId == Step.Physics)
		{
			Debug.LogWarning("Step.Physics is deprecated when using OpenXR");
			stepId = Step.Render;
		}

		if (version >= OVRP_1_78_0.version)
		{
			Result res = OVRP_1_78_0.ovrp_GetEyeGazesState(stepId, frameIndex, out cachedEyeGazesState);
			if (res == Result.Success)
			{
				// attempt to avoid allocations if client provides appropriately pre-initialized HandState
				if (eyeGazesState.EyeGazes == null || eyeGazesState.EyeGazes.Length != (int)Eye.Count)
				{
					eyeGazesState.EyeGazes = new EyeGazeState[(int)Eye.Count];
				}

				// unrolling the arrays is necessary to avoid per-frame allocations during marshaling
				eyeGazesState.EyeGazes[0] = cachedEyeGazesState.EyeGazes_0;
				eyeGazesState.EyeGazes[1] = cachedEyeGazesState.EyeGazes_1;
				eyeGazesState.Time = cachedEyeGazesState.Time;

				return true;
			}
			else
			{
				return false;
			}
		}
		else
		{
			return false;
		}
#endif
	}


	public static bool StartEyeTracking() =>
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		false;
#else
		version >= OVRP_1_78_0.version &&
		OVRP_1_78_0.ovrp_StartEyeTracking() == Result.Success;
#endif // OVRPLUGIN_UNSUPPORTED_PLATFORM

	public static bool StopEyeTracking() =>
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		false;
#else
		version >= OVRP_1_78_0.version &&
		OVRP_1_78_0.ovrp_StopEyeTracking() == Result.Success;
#endif // OVRPLUGIN_UNSUPPORTED_PLATFORM

	public static bool StartFaceTracking() =>
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		false;
#else
		version >= OVRP_1_78_0.version &&
		OVRP_1_78_0.ovrp_StartFaceTracking() == Result.Success;
#endif // OVRPLUGIN_UNSUPPORTED_PLATFORM

	public static bool StopFaceTracking() =>
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		false;
#else
		version >= OVRP_1_78_0.version &&
		OVRP_1_78_0.ovrp_StopFaceTracking() == Result.Success;
#endif // OVRPLUGIN_UNSUPPORTED_PLATFORM

	public static bool StartBodyTracking() =>
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		false;
#else
		version >= OVRP_1_78_0.version &&
		OVRP_1_78_0.ovrp_StartBodyTracking() == Result.Success;
#endif // OVRPLUGIN_UNSUPPORTED_PLATFORM

	public static bool StopBodyTracking() =>
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		false;
#else
		version >= OVRP_1_78_0.version &&
		OVRP_1_78_0.ovrp_StopBodyTracking() == Result.Success;
#endif // OVRPLUGIN_UNSUPPORTED_PLATFORM

	public static int GetLocalTrackingSpaceRecenterCount()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return 0;
#else
		if (version >= OVRP_1_44_0.version)
		{
			int recenterCount = 0;
			Result res = OVRP_1_44_0.ovrp_GetLocalTrackingSpaceRecenterCount(ref recenterCount);
			if (res == Result.Success)
			{
				return recenterCount;
			}

			return 0;
		}
		else
		{
			return 0;
		}
#endif
	}

	public static bool GetSystemHmd3DofModeEnabled()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_45_0.version)
		{
			Bool val = Bool.False;
			Result res = OVRP_1_45_0.ovrp_GetSystemHmd3DofModeEnabled(ref val);
			if (res == Result.Success)
			{
				return val == Bool.True;
			}

			return false;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool SetClientColorDesc(ColorSpace colorSpace)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_49_0.version)
		{
			if (colorSpace == ColorSpace.Unknown)
			{
				Debug.LogWarning("A color gamut of Unknown is not supported. Defaulting to DCI-P3 color space instead.");
				colorSpace = ColorSpace.P3;
			}
			return OVRP_1_49_0.ovrp_SetClientColorDesc(colorSpace) == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static ColorSpace GetHmdColorDesc()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return ColorSpace.Unknown;
#else
		ColorSpace colorSpace = ColorSpace.Unknown;
		if (version >= OVRP_1_49_0.version)
		{
			Result res = OVRP_1_49_0.ovrp_GetHmdColorDesc(ref colorSpace);
			if(res != Result.Success)
			{
				Debug.LogError("GetHmdColorDesc: Failed to get Hmd color description");
			}
			return colorSpace;
		}
		else
		{
			Debug.LogError("GetHmdColorDesc: Not supported on this version of OVRPlugin");
			return colorSpace;
		}
#endif
	}

	public static bool PollEvent(ref EventDataBuffer eventDataBuffer)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		eventDataBuffer = default(EventDataBuffer);
		return false;
#else
		if (version >= OVRP_1_55_1.version)
		{
			IntPtr DataPtr = IntPtr.Zero;
			if(eventDataBuffer.EventData == null)
			{
				eventDataBuffer.EventData = new byte[EventDataBufferSize];
			}
			Result result = OVRP_1_55_1.ovrp_PollEvent2(ref eventDataBuffer.EventType, ref DataPtr);

			if (result != Result.Success || DataPtr == IntPtr.Zero)
				return false;

			Marshal.Copy(DataPtr, eventDataBuffer.EventData, 0, EventDataBufferSize);
			return true;
		}
		else if (version >= OVRP_1_55_0.version)
		{
			return OVRP_1_55_0.ovrp_PollEvent(ref eventDataBuffer) == Result.Success;
		}
		else
		{
			eventDataBuffer = default(EventDataBuffer);
			return false;
		}
#endif
	}



	public static UInt64 GetNativeOpenXRInstance()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0;
#else
		if (version >= OVRP_1_55_0.version)
		{
			UInt64 instance, session;
			if (OVRP_1_55_0.ovrp_GetNativeOpenXRHandles(out instance, out session) == Result.Success)
			{
				return instance;
			}
		}
		return 0;
#endif
	}

	public static UInt64 GetNativeOpenXRSession()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0;
#else
		if (version >= OVRP_1_55_0.version)
		{
			UInt64 instance, session;
			if (OVRP_1_55_0.ovrp_GetNativeOpenXRHandles(out instance, out session) == Result.Success)
			{
				return session;
			}
		}
		return 0;
#endif
	}

	public static bool SetKeyboardOverlayUV(Vector2f uv)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_57_0.version)
		{
			Result result = OVRP_1_57_0.ovrp_SetKeyboardOverlayUV(uv);
			return (result == Result.Success);
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool CreateSpatialAnchor(SpatialAnchorCreateInfo createInfo, out UInt64 requestId) {
		requestId = 0;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_72_0.version)
		{
			Result result = OVRP_1_72_0.ovrp_CreateSpatialAnchor(ref createInfo, out requestId);
			return (result == Result.Success);
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool SetSpaceComponentStatus(UInt64 space, SpaceComponentType componentType, bool enable, double timeout, out UInt64 requestId) {
		requestId = 0;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_72_0.version)
		{
			Result result = OVRP_1_72_0.ovrp_SetSpaceComponentStatus(ref space, componentType, ToBool(enable), timeout, out requestId);
			return (result == Result.Success);
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool GetSpaceComponentStatus(UInt64 space, SpaceComponentType componentType, out bool enabled, out bool changePending) {
		enabled = false;
		changePending = false;

#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_72_0.version)
		{
			Bool isEnabled, isPending;
			Result result = OVRP_1_72_0.ovrp_GetSpaceComponentStatus(ref space, componentType, out isEnabled, out isPending);
			enabled = isEnabled == Bool.True;
			changePending = isPending == Bool.True;
			return (result == Result.Success);
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool EnumerateSpaceSupportedComponents(UInt64 space, out uint numSupportedComponents, SpaceComponentType[] supportedComponents) {
		numSupportedComponents = 0;

#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_72_0.version)
		{
			Result result = OVRP_1_72_0.ovrp_EnumerateSpaceSupportedComponents(ref space, (uint)supportedComponents.Length, out numSupportedComponents, supportedComponents);
			return (result == Result.Success);
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool SaveSpace(UInt64 space, SpaceStorageLocation location, SpaceStoragePersistenceMode mode, out UInt64 requestId) {
		requestId = 0;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_72_0.version)
		{
			return OVRP_1_72_0.ovrp_SaveSpace(ref space, location, mode, out requestId) == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool EraseSpace(UInt64 space, SpaceStorageLocation location, out UInt64 requestId) {
		requestId = 0;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_72_0.version)
		{
			return OVRP_1_72_0.ovrp_EraseSpace(ref space, location, out requestId) == Result.Success;
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool GetSpaceUuid(UInt64 space, out Guid uuid) {
		uuid = default;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		return version >= OVRP_1_74_0.version &&
		       OVRP_1_74_0.ovrp_GetSpaceUuid(in space, out uuid) == Result.Success;
#endif
	}

	public static bool QuerySpaces(SpaceQueryInfo queryInfo, out UInt64 requestId) {
		requestId = 0;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_72_0.version)
		{
			if (queryInfo.FilterType == SpaceQueryFilterType.Ids)
			{
				if (queryInfo.IdInfo.Ids?.Length > SpaceFilterInfoIdsMaxSize)
				{
					Debug.LogError("QuerySpaces attempted to query more uuids than the maximum number supported: " + SpaceFilterInfoIdsMaxSize);
					return false;
				}
			}
			else if (queryInfo.FilterType == SpaceQueryFilterType.Components)
			{
				if (queryInfo.ComponentsInfo.Components?.Length > SpaceFilterInfoComponentsMaxSize)
				{
					Debug.LogError("QuerySpaces attempted to query more components than the maximum number supported: " + SpaceFilterInfoComponentsMaxSize);
					return false;
				}
			}

			// The array size must be exactly SpaceFilterInfoIdsMaxSize or else the data marshaling will fail
			if (queryInfo.IdInfo.Ids?.Length != SpaceFilterInfoIdsMaxSize)
			{
				Array.Resize(ref queryInfo.IdInfo.Ids, SpaceFilterInfoIdsMaxSize);
			}

			// The array size must be exactly SpaceFilterInfoComponentsMaxSize or else the data marshaling will fail
			if (queryInfo.ComponentsInfo.Components?.Length != SpaceFilterInfoComponentsMaxSize)
			{
				Array.Resize(ref queryInfo.ComponentsInfo.Components, SpaceFilterInfoComponentsMaxSize);
			}

			Result result = OVRP_1_72_0.ovrp_QuerySpaces(ref queryInfo, out requestId);
			return (result == Result.Success);
		}
		else
		{
			return false;
		}
#endif
	}

	public static unsafe bool RetrieveSpaceQueryResults(UInt64 requestId,
		out NativeArray<SpaceQueryResult> results, Allocator allocator)
	{
		results = default;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
        return false;
#else

		if (version < OVRP_1_72_0.version) return false;

		uint count = 0;
		if (OVRP_1_72_0.ovrp_RetrieveSpaceQueryResults(ref requestId, default,
			    ref count, default) != Result.Success)
		{
			return false;
		}

		results = new NativeArray<SpaceQueryResult>((int)count, allocator);
		if (OVRP_1_72_0.ovrp_RetrieveSpaceQueryResults(ref requestId,
			    (uint)results.Length, ref count, new IntPtr(results.GetUnsafePtr()))
		    != Result.Success)
		{
			results.Dispose();
			return false;
		}

		return true;
#endif
	}

	public static bool RetrieveSpaceQueryResults(UInt64 requestId, out SpaceQueryResult[] results) {
		results = null;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_72_0.version)
		{
			Result result;
			IntPtr nullResultsPtr = new IntPtr(0);
			UInt32 resultCountOutput = 0;
			result = OVRP_1_72_0.ovrp_RetrieveSpaceQueryResults(ref requestId, 0, ref resultCountOutput, nullResultsPtr);
			if (result != Result.Success)
			{
				return false;
			}

			int spaceQueryResultSize = Marshal.SizeOf(typeof(SpaceQueryResult));
			int resultsSizeInBytes = (int)resultCountOutput * spaceQueryResultSize;
			IntPtr resultsPtr = Marshal.AllocHGlobal(resultsSizeInBytes);
			result = OVRP_1_72_0.ovrp_RetrieveSpaceQueryResults(ref requestId, resultCountOutput, ref resultCountOutput, resultsPtr);
			if (result != Result.Success)
			{
				Marshal.FreeHGlobal(resultsPtr);
				return false;
			}

			results = new SpaceQueryResult[resultCountOutput];
			for (int i = 0; i < resultCountOutput; i++)
			{
				SpaceQueryResult r = (SpaceQueryResult)Marshal.PtrToStructure(resultsPtr + (i * spaceQueryResultSize), typeof(SpaceQueryResult));
				results[i] = r;
			}

			Marshal.FreeHGlobal(resultsPtr);
			return true;
		}
		else
		{
			return false;
		}
#endif
	}

	public static OVRPlugin.Result SaveSpaceList(NativeArray<ulong> spaces, SpaceStorageLocation location, out UInt64 requestId)
	{
		requestId = 0;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return Result.Failure_Unsupported;
#else
		if (version >= OVRP_1_79_0.version)
		{
			unsafe
			{
				Result result = OVRP_1_79_0.ovrp_SaveSpaceList((ulong*)spaces.GetUnsafeReadOnlyPtr(),
					(uint)spaces.Length, location, out requestId);
				return result;
			}
		}
		else
		{
			return Result.Failure_Unsupported;
		}
#endif
	}

	public static bool GetSpaceUserId(UInt64 spaceUserHandle, out UInt64 spaceUserId)
	{
		spaceUserId = 0;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		return version >= OVRP_1_79_0.version && OVRP_1_79_0.ovrp_GetSpaceUserId(in spaceUserHandle, out spaceUserId) == Result.Success;
#endif
	}

	public static bool CreateSpaceUser(UInt64 spaceUserId, out UInt64 spaceUserHandle)
	{
		spaceUserHandle = 0;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
        return false;
#else
		return version >= OVRP_1_79_0.version && OVRP_1_79_0.ovrp_CreateSpaceUser(in spaceUserId, out spaceUserHandle) == Result.Success;
#endif
	}

	public static bool DestroySpaceUser(UInt64 spaceUserHandle)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
        return false;
#else
		return version >= OVRP_1_79_0.version && OVRP_1_79_0.ovrp_DestroySpaceUser(in spaceUserHandle) == Result.Success;
#endif
	}

	public static unsafe Result ShareSpaces(NativeArray<ulong> spaces, NativeArray<ulong> userHandles, out UInt64 requestId)
	{
		requestId = 0;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return Result.Failure_Unsupported;
#else
		if (version >= OVRP_1_79_0.version)
		{
			return OVRP_1_79_0.ovrp_ShareSpaces((ulong*)spaces.GetUnsafeReadOnlyPtr(), (uint)spaces.Length,
				(ulong*)userHandles.GetUnsafeReadOnlyPtr(), (uint)userHandles.Length, out requestId);
		}
		else
		{
			return Result.Failure_Unsupported;
		}
#endif
	}

	public static bool TryLocateSpace(UInt64 space, TrackingOrigin baseOrigin, out Posef pose)
	{
		using var profiler = new OVRProfilerScope(nameof(TryLocateSpace));

		pose = Posef.identity;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		return version >= OVRP_1_64_0.version &&
		       OVRP_1_64_0.ovrp_LocateSpace(ref pose, ref space, baseOrigin) == Result.Success;
#endif // OVRPLUGIN_UNSUPPORTED_PLATFORM
	}

	[Obsolete(nameof(LocateSpace) + " unconditionally returns a pose, even if the underlying OpenXR function fails. Instead, use TryLocateSpace, which indicates failure.")]
	public static Posef LocateSpace(UInt64 space, TrackingOrigin baseOrigin) =>
		TryLocateSpace(space, baseOrigin, out var pose) ? pose : Posef.identity;

	public static bool TryLocateSpace(UInt64 space, TrackingOrigin baseOrigin, out Posef pose,
		out SpaceLocationFlags locationFlags)
	{
		pose = Posef.identity;
		locationFlags = default;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
        return false;
#else
		if (version >= OVRP_1_79_0.version &&
		    OVRP_1_79_0.ovrp_LocateSpace2(out var spaceLocation, in space, baseOrigin) ==
		    Result.Success)
		{
			pose = spaceLocation.pose;
			locationFlags = spaceLocation.locationFlags;
			return true;
		}

		return false;
#endif // OVRPLUGIN_UNSUPPORTED_PLATFORM
	}

	public static bool DestroySpace(UInt64 space)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_65_0.version)
		{
			Result result = OVRP_1_65_0.ovrp_DestroySpace(ref space);
			return (result == Result.Success);
		}
		else
		{
			return false;
		}
#endif
	}

	[StructLayout(LayoutKind.Sequential)]
	private struct SpaceContainerInternal
	{
		public int uuidCapacityInput;
		public int uuidCountOutput;
		public IntPtr uuids;
	}

	[StructLayout(LayoutKind.Sequential)]
	private struct SpaceSemanticLabelInternal
	{
		public int byteCapacityInput;
		public int byteCountOutput;
		public IntPtr labels;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct RoomLayout
	{
		public Guid floorUuid;
		public Guid ceilingUuid;
		public Guid[] wallUuids;
	}

	[StructLayout(LayoutKind.Sequential)]
	private struct RoomLayoutInternal
	{
		public Guid floorUuid;
		public Guid ceilingUuid;
		public int wallUuidCapacityInput;
		public int wallUuidCountOutput;
		public IntPtr wallUuids;
	}

	[StructLayout(LayoutKind.Sequential)]
	private struct PolygonalBoundary2DInternal
	{
		public int vertexCapacityInput;
		public int vertexCountOutput;
		public IntPtr vertices;
	}

	[StructLayout(LayoutKind.Sequential)]
	private struct SceneCaptureRequestInternal
	{
		public int requestByteCount;
		[MarshalAs(UnmanagedType.LPStr)]
		public string request;
	}

	private struct PinnedArray<T> : IDisposable where T : unmanaged
	{
		GCHandle _handle;
		public PinnedArray(T[] array) =>_handle = GCHandle.Alloc(array, GCHandleType.Pinned);
		public void Dispose() => _handle.Free();
		public static implicit operator IntPtr(PinnedArray<T> pinnedArray) => pinnedArray._handle.AddrOfPinnedObject();
	}

	public static bool GetSpaceContainer(UInt64 space, out Guid[] containerUuids) {
		containerUuids = Array.Empty<Guid>();
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version < OVRP_1_72_0.version) return false;

		var containerInternal = default(SpaceContainerInternal);
		if (OVRP_1_72_0.ovrp_GetSpaceContainer(ref space, ref containerInternal) != Result.Success) return false;

		var uuids = new Guid[containerInternal.uuidCountOutput];
		using (var pinnedArray = new PinnedArray<Guid>(uuids)) {
			containerInternal.uuidCapacityInput = containerInternal.uuidCountOutput;
			containerInternal.uuids = pinnedArray;
			if (OVRP_1_72_0.ovrp_GetSpaceContainer(ref space, ref containerInternal) != Result.Success) return false;
		}

		containerUuids = uuids;
		return true;
#endif
	}

	public static bool GetSpaceBoundingBox2D(UInt64 space, out Rectf rect) {
		rect = new Rectf();
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_72_0.version)
		{
			Result result = OVRP_1_72_0.ovrp_GetSpaceBoundingBox2D(ref space, out rect);
			return (result == Result.Success);
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool GetSpaceBoundingBox3D(UInt64 space, out Boundsf bounds) {
		bounds = new Boundsf();
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_72_0.version)
		{
			Result result = OVRP_1_72_0.ovrp_GetSpaceBoundingBox3D(ref space, out bounds);
			return (result == Result.Success);
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool GetSpaceSemanticLabels(UInt64 space, out string labels) {
		labels = "";
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_72_0.version) {
			var labelsInternal = new SpaceSemanticLabelInternal {
				byteCapacityInput = 0,
				byteCountOutput = 0,
			};
			Result result = OVRP_1_72_0.ovrp_GetSpaceSemanticLabels(ref space, ref labelsInternal);
			if (result == Result.Success) {
				labelsInternal.byteCapacityInput = labelsInternal.byteCountOutput;
				labelsInternal.labels = Marshal.AllocHGlobal(sizeof(byte) * labelsInternal.byteCountOutput);
				result = OVRP_1_72_0.ovrp_GetSpaceSemanticLabels(ref space, ref labelsInternal);
				labels = Marshal.PtrToStringAnsi(labelsInternal.labels, labelsInternal.byteCountOutput);
				Marshal.FreeHGlobal(labelsInternal.labels);
			}
			return (result == Result.Success);
		} else {
			return false;
		}
#endif
	}

	public static bool GetSpaceRoomLayout(UInt64 space, out RoomLayout roomLayout) {
		roomLayout = new RoomLayout();
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version < OVRP_1_72_0.version) return false;

		var roomLayoutInternal = default(RoomLayoutInternal);
		if (OVRP_1_72_0.ovrp_GetSpaceRoomLayout(ref space, ref roomLayoutInternal) != Result.Success) return false;

		var uuids = new Guid[roomLayoutInternal.wallUuidCountOutput];
		using (var uuidBuffer = new PinnedArray<Guid>(uuids)) {
			roomLayoutInternal.wallUuidCapacityInput = roomLayoutInternal.wallUuidCountOutput;
			roomLayoutInternal.wallUuids = uuidBuffer;
			if (OVRP_1_72_0.ovrp_GetSpaceRoomLayout(ref space, ref roomLayoutInternal) != Result.Success) return false;
		}

		roomLayout.ceilingUuid = roomLayoutInternal.ceilingUuid;
		roomLayout.floorUuid = roomLayoutInternal.floorUuid;
		roomLayout.wallUuids = uuids;
		return true;
#endif
	}

	public static bool GetSpaceBoundary2DCount(UInt64 space, out int count)
	{
		count = 0;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
        return false;
#else
		if (version < OVRP_1_72_0.version) return false;

		var boundaryInternal = default(PolygonalBoundary2DInternal);
		var result = OVRP_1_72_0.ovrp_GetSpaceBoundary2D(ref space, ref boundaryInternal);
		count = boundaryInternal.vertexCountOutput;
		return result == Result.Success;
#endif
	}

	public static unsafe bool GetSpaceBoundary2D(UInt64 space, NativeArray<Vector2> boundary)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
        return false;
#else
		if (version < OVRP_1_72_0.version) return false;

		var boundaryInternal = new PolygonalBoundary2DInternal
		{
			vertexCapacityInput = boundary.Length,
			vertices = new IntPtr(boundary.GetUnsafePtr()),
		};

		return OVRP_1_72_0.ovrp_GetSpaceBoundary2D(ref space, ref boundaryInternal) == Result.Success;
#endif
	}


	public static NativeArray<Vector2> GetSpaceBoundary2D(UInt64 space, Allocator allocator)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return default;
#else
		if (version < OVRP_1_72_0.version) return default;

		var boundaryInternal = new PolygonalBoundary2DInternal
		{
			vertexCapacityInput = 0,
			vertexCountOutput = 0,
		};

		// Two call idiom: first call just gets the number of vertices
		var result = OVRP_1_72_0.ovrp_GetSpaceBoundary2D(ref space, ref boundaryInternal);
		if (result != Result.Success) return default;

		var boundary = new NativeArray<Vector2>(boundaryInternal.vertexCountOutput, allocator);

		unsafe
		{
			boundaryInternal.vertices = new IntPtr(boundary.GetUnsafePtr());
			boundaryInternal.vertexCapacityInput = boundary.Length;
		}

		// Two call idiom: second call populates the array
		if (OVRP_1_72_0.ovrp_GetSpaceBoundary2D(ref space, ref boundaryInternal) == Result.Success)
		{
			return boundary;
		}

		boundary.Dispose();
		return default;
#endif
	}

	[Obsolete("This method allocates managed arrays. Use GetSpaceBoundary2D(UInt64, Allocator) to avoid managed allocations.")]
	public static bool GetSpaceBoundary2D(UInt64 space, out Vector2[] boundary) {
		boundary = Array.Empty<Vector2>();
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_72_0.version)
		{
			var boundaryInternal = new PolygonalBoundary2DInternal {
				vertexCapacityInput = 0,
				vertexCountOutput = 0,
			};
			Result result = OVRP_1_72_0.ovrp_GetSpaceBoundary2D(ref space, ref boundaryInternal);
			if (result == Result.Success) {
				boundaryInternal.vertexCapacityInput = boundaryInternal.vertexCountOutput;
				int size = Marshal.SizeOf(typeof(Vector2));
				boundaryInternal.vertices = Marshal.AllocHGlobal(boundaryInternal.vertexCountOutput * size);
				result = OVRP_1_72_0.ovrp_GetSpaceBoundary2D(ref space, ref boundaryInternal);
				if (result == Result.Success) {
					boundary = new Vector2[boundaryInternal.vertexCountOutput];

					IntPtr LongPtr = boundaryInternal.vertices;
					for(int i = 0; i < boundaryInternal.vertexCountOutput; i++){
						IntPtr tempPtr = new IntPtr(size);
						tempPtr = LongPtr;
						LongPtr += size;
						boundary[i] = Marshal.PtrToStructure<Vector2>(tempPtr);
					}

					Marshal.FreeHGlobal(boundaryInternal.vertices);
				}
			}
			return (result == Result.Success);
		}
		else
		{
			return false;
		}
#endif
	}

	public static bool RequestSceneCapture(string requestString, out UInt64 requestId) {
		requestId = 0;
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		if (version >= OVRP_1_72_0.version)
		{
			var sceneCaptureRequest = new SceneCaptureRequestInternal {
				requestByteCount = requestString == null ? 0 : System.Text.Encoding.ASCII.GetByteCount(requestString),
				request = requestString,
			};
			Result result = OVRP_1_72_0.ovrp_RequestSceneCapture(ref sceneCaptureRequest, out requestId);
			return (result == Result.Success);
		}
		else
		{
			return false;
		}
#endif
	}


	public static string[] GetRenderModelPaths()
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return null;
#else
		if (version >= OVRP_1_68_0.version)
		{
			uint index = 0;
			List<string> paths = new List<string>();
			IntPtr pathPtr = Marshal.AllocHGlobal(sizeof(byte) * OVRP_1_68_0.OVRP_RENDER_MODEL_MAX_PATH_LENGTH);
			while (OVRP_1_68_0.ovrp_GetRenderModelPaths(index, pathPtr) == Result.Success)
			{
				paths.Add(Marshal.PtrToStringAnsi(pathPtr));
				index++;
			}
			Marshal.FreeHGlobal(pathPtr);
			return paths.ToArray();
		}
		else
		{
			return null;
		}
#endif
	}

	public static bool GetRenderModelProperties(string modelPath, ref RenderModelProperties modelProperties)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return false;
#else
		Result result;
		RenderModelPropertiesInternal props;
		if (version >= OVRP_1_74_0.version)
		{
			result = OVRP_1_74_0.ovrp_GetRenderModelProperties2(
				modelPath,
				RenderModelFlags.SupportsGltf20Subset2,
				out props);
		}
		else if (version >= OVRP_1_68_0.version)
		{
			result = OVRP_1_68_0.ovrp_GetRenderModelProperties(modelPath, out props);
		}
		else
		{
			return false;
		}

		if (result != Result.Success)
			return false;

		modelProperties.ModelName = System.Text.Encoding.Default.GetString(props.ModelName);
		modelProperties.ModelKey = props.ModelKey;
		modelProperties.VendorId = props.VendorId;
		modelProperties.ModelVersion = props.ModelVersion;
		return true;
#endif
	}

	public static byte[] LoadRenderModel(UInt64 modelKey)
	{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
		return null;
#else
		if (version >= OVRP_1_68_0.version)
		{
			uint bufferSize = 0;
			if (OVRP_1_68_0.ovrp_LoadRenderModel(modelKey, 0, ref bufferSize, IntPtr.Zero) != Result.Success)
			{
				return null;
			}

			if (bufferSize == 0)
			{
				return null;
			}

			IntPtr bufferPtr = Marshal.AllocHGlobal((int)bufferSize);
			if (OVRP_1_68_0.ovrp_LoadRenderModel(modelKey, bufferSize, ref bufferSize, bufferPtr) != Result.Success)
			{
				Marshal.FreeHGlobal(bufferPtr);
				return null;
			}

			byte[] bufferData = new byte[bufferSize];
			Marshal.Copy(bufferPtr, bufferData, 0, (int)bufferSize);
			Marshal.FreeHGlobal(bufferPtr);

			return bufferData;
		}
		else
		{
			return null;
		}
#endif
	}

	public class Ktx
	{
		public static IntPtr LoadKtxFromMemory(IntPtr dataPtr, uint length)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return IntPtr.Zero;
#else
			if (nativeXrApi != XrApi.OpenXR)
			{
				Debug.LogWarning("KTX features are only supported in OpenXR.");
				return IntPtr.Zero;
			}

			if (version >= OVRP_1_65_0.version)
			{
				IntPtr texture = IntPtr.Zero;
				OVRP_1_65_0.ovrp_KtxLoadFromMemory(ref dataPtr, length, ref texture);
				return texture;
			}
			return IntPtr.Zero;
#endif
		}

		public static uint GetKtxTextureWidth(IntPtr texture)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0;
#else
			if (nativeXrApi != XrApi.OpenXR)
			{
				Debug.LogWarning("KTX features are only supported in OpenXR.");
				return 0;
			}

			if (version >= OVRP_1_65_0.version)
			{
				uint width = 0;
				OVRP_1_65_0.ovrp_KtxTextureWidth(texture, ref width);
				return width;
			}
			return 0;
#endif
		}

		public static uint GetKtxTextureHeight(IntPtr texture)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0;
#else
			if (nativeXrApi != XrApi.OpenXR)
			{
				Debug.LogWarning("KTX features are only supported in OpenXR.");
				return 0;
			}

			if (version >= OVRP_1_65_0.version)
			{
				uint height = 0;
				OVRP_1_65_0.ovrp_KtxTextureHeight(texture, ref height);
				return height;
			}
			return 0;
#endif
		}

		public static bool TranscodeKtxTexture(IntPtr texture, uint format)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (nativeXrApi != XrApi.OpenXR)
			{
				Debug.LogWarning("KTX features are only supported in OpenXR.");
				return false;
			}

			if (version >= OVRP_1_65_0.version)
			{
				return OVRP_1_65_0.ovrp_KtxTranscode(texture, format) == Result.Success;
			}
			return false;
#endif
		}

		public static uint GetKtxTextureSize(IntPtr texture)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return 0;
#else
			if (nativeXrApi != XrApi.OpenXR)
			{
				Debug.LogWarning("KTX features are only supported in OpenXR.");
				return 0;
			}

			if (version >= OVRP_1_65_0.version)
			{
				uint size = 0;
				OVRP_1_65_0.ovrp_KtxTextureSize(texture, ref size);
				return size;
			}
			return 0;
#endif
		}

		public static bool GetKtxTextureData(IntPtr texture, IntPtr textureData, uint bufferSize)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (nativeXrApi != XrApi.OpenXR)
			{
				Debug.LogWarning("KTX features are only supported in OpenXR.");
				return false;
			}

			if (version >= OVRP_1_65_0.version)
			{
				return OVRP_1_65_0.ovrp_KtxGetTextureData(texture, textureData, bufferSize) == Result.Success;
			}
			return false;
#endif
		}

		public static bool DestroyKtxTexture(IntPtr texture)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (nativeXrApi != XrApi.OpenXR)
			{
				Debug.LogWarning("KTX features are only supported in OpenXR.");
				return false;
			}

			if (version >= OVRP_1_65_0.version)
			{
				return OVRP_1_65_0.ovrp_KtxDestroy(texture) == Result.Success;
			}
			return false;
#endif
		}
	}






	public class UnityOpenXR
	{
		public static bool Enabled = false; // OculusXRFeature will set it to true when being used

		public static void SetClientVersion()
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			// do nothing
#else
			if (version >= OVRP_1_71_0.version)
			{
				OVRP_1_71_0.ovrp_UnityOpenXR_SetClientVersion(wrapperVersion.Major, wrapperVersion.Minor, wrapperVersion.Build);
			}
#endif
		}

		public static IntPtr HookGetInstanceProcAddr(IntPtr func)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return func;
#else
			if (version >= OVRP_1_71_0.version)
			{
				return OVRP_1_71_0.ovrp_UnityOpenXR_HookGetInstanceProcAddr(func);
			}
			else
			{
				return func;
			}
#endif
		}

		public static bool OnInstanceCreate(UInt64 xrInstance)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
			return false;
#else
			if (version >= OVRP_1_71_0.version)
			{
				Result result = OVRP_1_71_0.ovrp_UnityOpenXR_OnInstanceCreate(xrInstance);
				return result == Result.Success;
			}
			else
			{
				return false;
			}
#endif
		}

		public static void OnInstanceDestroy(UInt64 xrInstance)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
#else
			if (version >= OVRP_1_71_0.version)
			{
				OVRP_1_71_0.ovrp_UnityOpenXR_OnInstanceDestroy(xrInstance);
			}
#endif
		}

		public static void OnSessionCreate(UInt64 xrSession)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
#else
			if (version >= OVRP_1_71_0.version)
			{
				OVRP_1_71_0.ovrp_UnityOpenXR_OnSessionCreate(xrSession);
			}
#endif
		}

		public static void OnAppSpaceChange(UInt64 xrSpace)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
#else
			if (version >= OVRP_1_71_0.version)
			{
				OVRP_1_71_0.ovrp_UnityOpenXR_OnAppSpaceChange(xrSpace);
			}
#endif
		}

		public static void OnSessionStateChange(int oldState, int newState)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
#else
			if (version >= OVRP_1_71_0.version)
			{
				OVRP_1_71_0.ovrp_UnityOpenXR_OnSessionStateChange(oldState, newState);
			}
#endif
		}

		public static void OnSessionBegin(UInt64 xrSession)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
#else
			if (version >= OVRP_1_71_0.version)
			{
				OVRP_1_71_0.ovrp_UnityOpenXR_OnSessionBegin(xrSession);
			}
#endif
		}

		public static void OnSessionEnd(UInt64 xrSession)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
#else
			if (version >= OVRP_1_71_0.version)
			{
				OVRP_1_71_0.ovrp_UnityOpenXR_OnSessionEnd(xrSession);
			}
#endif
		}

		public static void OnSessionExiting(UInt64 xrSession)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
#else
			if (version >= OVRP_1_71_0.version)
			{
				OVRP_1_71_0.ovrp_UnityOpenXR_OnSessionExiting(xrSession);
			}
#endif
		}

		public static void OnSessionDestroy(UInt64 xrSession)
		{
#if OVRPLUGIN_UNSUPPORTED_PLATFORM
#else
			if (version >= OVRP_1_71_0.version)
			{
				OVRP_1_71_0.ovrp_UnityOpenXR_OnSessionDestroy(xrSession);
			}
#endif
		}
	}


	private const string pluginName = "OVRPlugin";
	private static System.Version _versionZero = new System.Version(0, 0, 0);

	// Disable all the DllImports when the platform is not supported
#if !OVRPLUGIN_UNSUPPORTED_PLATFORM

	private static class OVRP_0_1_0
	{
		public static readonly System.Version version = new System.Version(0, 1, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Sizei ovrp_GetEyeTextureSize(Eye eyeId);
	}

	private static class OVRP_0_1_1
	{
		public static readonly System.Version version = new System.Version(0, 1, 1);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetOverlayQuad2(Bool onTop, Bool headLocked, IntPtr texture, IntPtr device, Posef pose, Vector3f scale);
	}

	private static class OVRP_0_1_2
	{
		public static readonly System.Version version = new System.Version(0, 1, 2);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Posef ovrp_GetNodePose(Node nodeId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetControllerVibration(uint controllerMask, float frequency, float amplitude);
	}

	private static class OVRP_0_1_3
	{
		public static readonly System.Version version = new System.Version(0, 1, 3);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Posef ovrp_GetNodeVelocity(Node nodeId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Posef ovrp_GetNodeAcceleration(Node nodeId);
	}

	private static class OVRP_0_5_0
	{
		public static readonly System.Version version = new System.Version(0, 5, 0);
	}

	private static class OVRP_1_0_0
	{
		public static readonly System.Version version = new System.Version(1, 0, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern TrackingOrigin ovrp_GetTrackingOriginType();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetTrackingOriginType(TrackingOrigin originType);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Posef ovrp_GetTrackingCalibratedOrigin();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_RecenterTrackingOrigin(uint flags);
	}

	private static class OVRP_1_1_0
	{
		public static readonly System.Version version = new System.Version(1, 1, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetInitialized();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "ovrp_GetVersion")]
		private static extern IntPtr _ovrp_GetVersion();
		public static string ovrp_GetVersion() { return Marshal.PtrToStringAnsi(OVRP_1_1_0._ovrp_GetVersion()); }

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "ovrp_GetNativeSDKVersion")]
		private static extern IntPtr _ovrp_GetNativeSDKVersion();
		public static string ovrp_GetNativeSDKVersion() { return Marshal.PtrToStringAnsi(OVRP_1_1_0._ovrp_GetNativeSDKVersion()); }

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern IntPtr ovrp_GetAudioOutId();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern IntPtr ovrp_GetAudioInId();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern float ovrp_GetEyeTextureScale();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetEyeTextureScale(float value);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetTrackingOrientationSupported();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetTrackingOrientationEnabled();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetTrackingOrientationEnabled(Bool value);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetTrackingPositionSupported();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetTrackingPositionEnabled();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetTrackingPositionEnabled(Bool value);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetNodePresent(Node nodeId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetNodeOrientationTracked(Node nodeId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetNodePositionTracked(Node nodeId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Frustumf ovrp_GetNodeFrustum(Node nodeId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern ControllerState ovrp_GetControllerState(uint controllerMask);

		[System.Obsolete("Deprecated. Replaced by ovrp_GetSuggestedCpuPerformanceLevel", false)]
		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern int ovrp_GetSystemCpuLevel();

		[System.Obsolete("Deprecated. Replaced by ovrp_SetSuggestedCpuPerformanceLevel", false)]
		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetSystemCpuLevel(int value);

		[System.Obsolete("Deprecated. Replaced by ovrp_GetSuggestedGpuPerformanceLevel", false)]
		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern int ovrp_GetSystemGpuLevel();

		[System.Obsolete("Deprecated. Replaced by ovrp_SetSuggestedGpuPerformanceLevel", false)]
		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetSystemGpuLevel(int value);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetSystemPowerSavingMode();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern float ovrp_GetSystemDisplayFrequency();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern int ovrp_GetSystemVSyncCount();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern float ovrp_GetSystemVolume();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern BatteryStatus ovrp_GetSystemBatteryStatus();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern float ovrp_GetSystemBatteryLevel();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern float ovrp_GetSystemBatteryTemperature();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "ovrp_GetSystemProductName")]
		private static extern IntPtr _ovrp_GetSystemProductName();
		public static string ovrp_GetSystemProductName() { return Marshal.PtrToStringAnsi(OVRP_1_1_0._ovrp_GetSystemProductName()); }

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_ShowSystemUI(PlatformUI ui);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetAppMonoscopic();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetAppMonoscopic(Bool value);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetAppHasVrFocus();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetAppShouldQuit();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetAppShouldRecenter();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "ovrp_GetAppLatencyTimings")]
		private static extern IntPtr _ovrp_GetAppLatencyTimings();
		public static string ovrp_GetAppLatencyTimings() { return Marshal.PtrToStringAnsi(OVRP_1_1_0._ovrp_GetAppLatencyTimings()); }

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetUserPresent();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern float ovrp_GetUserIPD();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetUserIPD(float value);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern float ovrp_GetUserEyeDepth();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetUserEyeDepth(float value);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern float ovrp_GetUserEyeHeight();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetUserEyeHeight(float value);
	}

	private static class OVRP_1_2_0
	{
		public static readonly System.Version version = new System.Version(1, 2, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetSystemVSyncCount(int vsyncCount);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrpi_SetTrackingCalibratedOrigin();
	}

	private static class OVRP_1_3_0
	{
		public static readonly System.Version version = new System.Version(1, 3, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetEyeOcclusionMeshEnabled();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetEyeOcclusionMeshEnabled(Bool value);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetSystemHeadphonesPresent();
	}

	private static class OVRP_1_5_0
	{
		public static readonly System.Version version = new System.Version(1, 5, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern SystemRegion ovrp_GetSystemRegion();
	}

	private static class OVRP_1_6_0
	{
		public static readonly System.Version version = new System.Version(1, 6, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetTrackingIPDEnabled();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetTrackingIPDEnabled(Bool value);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern HapticsDesc ovrp_GetControllerHapticsDesc(uint controllerMask);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern HapticsState ovrp_GetControllerHapticsState(uint controllerMask);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetControllerHaptics(uint controllerMask, HapticsBuffer hapticsBuffer);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetOverlayQuad3(uint flags, IntPtr textureLeft, IntPtr textureRight, IntPtr device, Posef pose, Vector3f scale, int layerIndex);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern float ovrp_GetEyeRecommendedResolutionScale();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern float ovrp_GetAppCpuStartToGpuEndTime();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern int ovrp_GetSystemRecommendedMSAALevel();
	}

	private static class OVRP_1_7_0
	{
		public static readonly System.Version version = new System.Version(1, 7, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetAppChromaticCorrection();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetAppChromaticCorrection(Bool value);
	}

	private static class OVRP_1_8_0
	{
		public static readonly System.Version version = new System.Version(1, 8, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetBoundaryConfigured();

		[System.Obsolete("Deprecated. This function will not be supported in OpenXR", false)]
		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern BoundaryTestResult ovrp_TestBoundaryNode(Node nodeId, BoundaryType boundaryType);

		[System.Obsolete("Deprecated. This function will not be supported in OpenXR", false)]
		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern BoundaryTestResult ovrp_TestBoundaryPoint(Vector3f point, BoundaryType boundaryType);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern BoundaryGeometry ovrp_GetBoundaryGeometry(BoundaryType boundaryType);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Vector3f ovrp_GetBoundaryDimensions(BoundaryType boundaryType);

		[System.Obsolete("Deprecated. This function will not be supported in OpenXR", false)]
		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetBoundaryVisible();

		[System.Obsolete("Deprecated. This function will not be supported in OpenXR", false)]
		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetBoundaryVisible(Bool value);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_Update2(int stateId, int frameIndex, double predictionSeconds);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Posef ovrp_GetNodePose2(int stateId, Node nodeId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Posef ovrp_GetNodeVelocity2(int stateId, Node nodeId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Posef ovrp_GetNodeAcceleration2(int stateId, Node nodeId);
	}

	private static class OVRP_1_9_0
	{
		public static readonly System.Version version = new System.Version(1, 9, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern SystemHeadset ovrp_GetSystemHeadsetType();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Controller ovrp_GetActiveController();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Controller ovrp_GetConnectedControllers();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetBoundaryGeometry2(BoundaryType boundaryType, IntPtr points, ref int pointsCount);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern AppPerfStats ovrp_GetAppPerfStats();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_ResetAppPerfStats();
	}

	private static class OVRP_1_10_0
	{
		public static readonly System.Version version = new System.Version(1, 10, 0);
	}

	private static class OVRP_1_11_0
	{
		public static readonly System.Version version = new System.Version(1, 11, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_SetDesiredEyeTextureFormat(EyeTextureFormat value);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern EyeTextureFormat ovrp_GetDesiredEyeTextureFormat();
	}

	private static class OVRP_1_12_0
	{
		public static readonly System.Version version = new System.Version(1, 12, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern float ovrp_GetAppFramerate();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern PoseStatef ovrp_GetNodePoseState(Step stepId, Node nodeId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern ControllerState2 ovrp_GetControllerState2(uint controllerMask);
	}

	private static class OVRP_1_15_0
	{
		public static readonly System.Version version = new System.Version(1, 15, 0);

		public const int OVRP_EXTERNAL_CAMERA_NAME_SIZE = 32;

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_InitializeMixedReality();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_ShutdownMixedReality();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetMixedRealityInitialized();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_UpdateExternalCamera();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetExternalCameraCount(out int cameraCount);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetExternalCameraName(int cameraId, [MarshalAs(UnmanagedType.LPArray, SizeConst = OVRP_EXTERNAL_CAMERA_NAME_SIZE)] char[] cameraName);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetExternalCameraIntrinsics(int cameraId, out CameraIntrinsics cameraIntrinsics);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetExternalCameraExtrinsics(int cameraId, out CameraExtrinsics cameraExtrinsics);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_CalculateLayerDesc(OverlayShape shape, LayerLayout layout, ref Sizei textureSize,
			int mipLevels, int sampleCount, EyeTextureFormat format, int layerFlags, ref LayerDescInternal layerDesc);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_EnqueueSetupLayer(ref LayerDescInternal desc, IntPtr layerId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_EnqueueDestroyLayer(IntPtr layerId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetLayerTextureStageCount(int layerId, ref int layerTextureStageCount);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetLayerTexturePtr(int layerId, int stage, Eye eyeId, ref IntPtr textureHandle);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_EnqueueSubmitLayer(uint flags, IntPtr textureLeft, IntPtr textureRight, int layerId, int frameIndex, ref Posef pose, ref Vector3f scale, int layerIndex);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetNodeFrustum2(Node nodeId, out Frustumf2 nodeFrustum);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetEyeTextureArrayEnabled();
	}

	private static class OVRP_1_16_0
	{
		public static readonly System.Version version = new System.Version(1, 16, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_UpdateCameraDevices();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_IsCameraDeviceAvailable(CameraDevice cameraDevice);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetCameraDevicePreferredColorFrameSize(CameraDevice cameraDevice, Sizei preferredColorFrameSize);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_OpenCameraDevice(CameraDevice cameraDevice);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_CloseCameraDevice(CameraDevice cameraDevice);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_HasCameraDeviceOpened(CameraDevice cameraDevice);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_IsCameraDeviceColorFrameAvailable(CameraDevice cameraDevice);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetCameraDeviceColorFrameSize(CameraDevice cameraDevice, out Sizei colorFrameSize);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetCameraDeviceColorFrameBgraPixels(CameraDevice cameraDevice, out IntPtr colorFrameBgraPixels, out int colorFrameRowPitch);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetControllerState4(uint controllerMask, ref ControllerState4 controllerState);
	}

	private static class OVRP_1_17_0
	{
		public static readonly System.Version version = new System.Version(1, 17, 0);

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || OVRPLUGIN_EDITOR_MOCK_ENABLED
		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetExternalCameraPose(CameraDevice camera, out Posef cameraPose);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_ConvertPoseToCameraSpace(CameraDevice camera, ref Posef trackingSpacePose, out Posef cameraSpacePose);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetCameraDeviceIntrinsicsParameters(CameraDevice camera, out Bool supportIntrinsics, out CameraDeviceIntrinsicsParameters intrinsicsParameters);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_DoesCameraDeviceSupportDepth(CameraDevice camera, out Bool supportDepth);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetCameraDeviceDepthSensingMode(CameraDevice camera, out CameraDeviceDepthSensingMode depthSensoringMode);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetCameraDeviceDepthSensingMode(CameraDevice camera, CameraDeviceDepthSensingMode depthSensoringMode);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetCameraDevicePreferredDepthQuality(CameraDevice camera, out CameraDeviceDepthQuality depthQuality);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetCameraDevicePreferredDepthQuality(CameraDevice camera, CameraDeviceDepthQuality depthQuality);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_IsCameraDeviceDepthFrameAvailable(CameraDevice camera, out Bool available);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetCameraDeviceDepthFrameSize(CameraDevice camera, out Sizei depthFrameSize);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetCameraDeviceDepthFramePixels(CameraDevice cameraDevice, out IntPtr depthFramePixels, out int depthFrameRowPitch);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetCameraDeviceDepthConfidencePixels(CameraDevice cameraDevice, out IntPtr depthConfidencePixels, out int depthConfidenceRowPitch);
#endif
	}

	private static class OVRP_1_18_0
	{
		public static readonly System.Version version = new System.Version(1, 18, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetHandNodePoseStateLatency(double latencyInSeconds);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetHandNodePoseStateLatency(out double latencyInSeconds);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetAppHasInputFocus(out Bool appHasInputFocus);
	}

	private static class OVRP_1_19_0
	{
		public static readonly System.Version version = new System.Version(1, 19, 0);
	}

	private static class OVRP_1_21_0
	{
		public static readonly System.Version version = new System.Version(1, 21, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetTiledMultiResSupported(out Bool foveationSupported);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetTiledMultiResLevel(out FoveatedRenderingLevel level);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetTiledMultiResLevel(FoveatedRenderingLevel level);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetGPUUtilSupported(out Bool gpuUtilSupported);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetGPUUtilLevel(out float gpuUtil);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetSystemDisplayFrequency2(out float systemDisplayFrequency);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetSystemDisplayAvailableFrequencies(IntPtr systemDisplayAvailableFrequencies, ref int numFrequencies);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetSystemDisplayFrequency(float requestedFrequency);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetAppAsymmetricFov(out Bool useAsymmetricFov);
	}

	private static class OVRP_1_28_0
	{
		public static readonly System.Version version = new System.Version(1, 28, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetDominantHand(out Handedness dominantHand);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SendEvent(string name, string param);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_EnqueueSetupLayer2(ref LayerDescInternal desc, int compositionDepth, IntPtr layerId);

	}

	private static class OVRP_1_29_0
	{
		public static readonly System.Version version = new System.Version(1, 29, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetLayerAndroidSurfaceObject(int layerId, ref IntPtr surfaceObject);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetHeadPoseModifier(ref Quatf relativeRotation, ref Vector3f relativeTranslation);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetHeadPoseModifier(out Quatf relativeRotation, out Vector3f relativeTranslation);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetNodePoseStateRaw(Step stepId, int frameIndex, Node nodeId, out PoseStatef nodePoseState);
	}

	private static class OVRP_1_30_0
	{
		public static readonly System.Version version = new System.Version(1, 30, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetCurrentTrackingTransformPose(out Posef trackingTransformPose);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetTrackingTransformRawPose(out Posef trackingTransformRawPose);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SendEvent2(string name, string param, string source);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_IsPerfMetricsSupported(PerfMetrics perfMetrics, out Bool isSupported);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetPerfMetricsFloat(PerfMetrics perfMetrics, out float value);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetPerfMetricsInt(PerfMetrics perfMetrics, out int value);
	}

	private static class OVRP_1_31_0
	{
		public static readonly System.Version version = new System.Version(1, 31, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetTimeInSeconds(out double value);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetColorScaleAndOffset(Vector4 colorScale, Vector4 colorOffset, Bool applyToAllLayers);

	}

	private static class OVRP_1_32_0
	{
		public static readonly System.Version version = new System.Version(1, 32, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_AddCustomMetadata(string name, string param);
	}

	private static class OVRP_1_34_0
	{
		public static readonly System.Version version = new System.Version(1, 34, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_EnqueueSubmitLayer2(uint flags, IntPtr textureLeft, IntPtr textureRight, int layerId, int frameIndex, ref Posef pose, ref Vector3f scale, int layerIndex,
			Bool overrideTextureRectMatrix, ref TextureRectMatrixf textureRectMatrix, Bool overridePerLayerColorScaleAndOffset, ref Vector4 colorScale, ref Vector4 colorOffset);

	}

	private static class OVRP_1_35_0
	{
		public static readonly System.Version version = new System.Version(1, 35, 0);
	}

	private static class OVRP_1_36_0
	{
		public static readonly System.Version version = new System.Version(1, 36, 0);
	}

	private static class OVRP_1_37_0
	{
		public static readonly System.Version version = new System.Version(1, 37, 0);
	}

	private static class OVRP_1_38_0
	{
		public static readonly System.Version version = new System.Version(1, 38, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetTrackingTransformRelativePose(ref Posef trackingTransformRelativePose, TrackingOrigin trackingOrigin);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_Initialize();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_Shutdown();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_GetInitialized(out Bool initialized);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_Update();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_GetMrcActivationMode(out Media.MrcActivationMode activationMode);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_SetMrcActivationMode(Media.MrcActivationMode activationMode);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_IsMrcEnabled(out Bool mrcEnabled);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_IsMrcActivated(out Bool mrcActivated);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_UseMrcDebugCamera(out Bool useMrcDebugCamera);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_SetMrcInputVideoBufferType(Media.InputVideoBufferType inputVideoBufferType);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_GetMrcInputVideoBufferType(ref Media.InputVideoBufferType inputVideoBufferType);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_SetMrcFrameSize(int frameWidth, int frameHeight);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_GetMrcFrameSize(ref int frameWidth, ref int frameHeight);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_SetMrcAudioSampleRate(int sampleRate);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_GetMrcAudioSampleRate(ref int sampleRate);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_SetMrcFrameImageFlipped(Bool flipped);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_GetMrcFrameImageFlipped(ref Bool flipped);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_EncodeMrcFrame(System.IntPtr rawBuffer, System.IntPtr audioDataPtr, int audioDataLen, int audioChannels, double timestamp, ref int outSyncId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_EncodeMrcFrameWithDualTextures(System.IntPtr backgroundTextureHandle, System.IntPtr foregroundTextureHandle, System.IntPtr audioData, int audioDataLen, int audioChannels, double timestamp, ref int outSyncId);


		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_SyncMrcFrame(int syncId);


		//[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		//public static extern Result ovrp_GetExternalCameraCalibrationRawPose(int cameraId, out Posef rawPose);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetDeveloperMode(Bool active);


		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetNodeOrientationValid(Node nodeId, ref Bool nodeOrientationValid);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetNodePositionValid(Node nodeId, ref Bool nodePositionValid);
	}

	private static class OVRP_1_39_0
	{
		public static readonly System.Version version = new System.Version(1, 39, 0);
	}

	private static class OVRP_1_40_0
	{
		public static readonly System.Version version = new System.Version(1, 40, 0);
	}

	private static class OVRP_1_41_0
	{
		public static readonly System.Version version = new System.Version(1, 41, 0);
	}

	private static class OVRP_1_42_0
	{
		public static readonly System.Version version = new System.Version(1, 42, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetAdaptiveGpuPerformanceScale2(ref float adaptiveGpuPerformanceScale);
	}

	private static class OVRP_1_43_0
	{
		public static readonly System.Version version = new System.Version(1, 43, 0);
	}

	private static class OVRP_1_44_0
	{
		public static readonly System.Version version = new System.Version(1, 44, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetHandTrackingEnabled(ref Bool handTrackingEnabled);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetHandState(Step stepId, Hand hand, out HandStateInternal handState);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetSkeleton(SkeletonType skeletonType, out Skeleton skeleton);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetMesh(MeshType meshType, System.IntPtr meshPtr);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_OverrideExternalCameraFov(int cameraId, Bool useOverriddenFov, ref Fovf fov);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetUseOverriddenExternalCameraFov(int cameraId, out Bool useOverriddenFov);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_OverrideExternalCameraStaticPose(int cameraId, Bool useOverriddenPose, ref Posef poseInStageOrigin);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetUseOverriddenExternalCameraStaticPose(int cameraId, out Bool useOverriddenStaticPose);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_ResetDefaultExternalCamera();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetDefaultExternalCamera(string cameraName, ref CameraIntrinsics cameraIntrinsics, ref CameraExtrinsics cameraExtrinsics);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetLocalTrackingSpaceRecenterCount(ref int recenterCount);
	}

	private static class OVRP_1_45_0
	{
		public static readonly System.Version version = new System.Version(1, 45, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetSystemHmd3DofModeEnabled(ref Bool enabled);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_SetAvailableQueueIndexVulkan(uint queueIndexVk);
	}

	private static class OVRP_1_46_0
	{
		public static readonly System.Version version = new System.Version(1, 46, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetTiledMultiResDynamic(out Bool isDynamic);


		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetTiledMultiResDynamic(Bool isDynamic);
	}

	private static class OVRP_1_47_0
	{
		public static readonly System.Version version = new System.Version(1, 47, 0);
	}

	private static class OVRP_1_48_0
	{
		public static readonly System.Version version = new System.Version(1, 48, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetExternalCameraProperties(string cameraName, ref CameraIntrinsics cameraIntrinsics, ref CameraExtrinsics cameraExtrinsics);

	}

	private static class OVRP_1_49_0
	{
		public static readonly System.Version version = new System.Version(1, 49, 0);

		public const int OVRP_ANCHOR_NAME_SIZE = 32;

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetClientColorDesc(ColorSpace colorSpace);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetHmdColorDesc(ref ColorSpace colorSpace);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_EncodeMrcFrameWithPoseTime(IntPtr rawBuffer, IntPtr audioDataPtr, int audioDataLen, int audioChannels, double timestamp, double poseTime, ref int outSyncId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_EncodeMrcFrameDualTexturesWithPoseTime(IntPtr backgroundTextureHandle, IntPtr foregroundTextureHandle, IntPtr audioData, int audioDataLen, int audioChannels, double timestamp, double poseTime, ref int outSyncId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_SetHeadsetControllerPose(Posef headsetPose, Posef leftControllerPose, Posef rightControllerPose);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_EnumerateCameraAnchorHandles(ref int anchorCount, ref IntPtr CameraAnchorHandle);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_GetCurrentCameraAnchorHandle(ref IntPtr anchorHandle);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_GetCameraAnchorName(IntPtr anchorHandle, [MarshalAs(UnmanagedType.LPArray, SizeConst = OVRP_ANCHOR_NAME_SIZE)] char[] cameraName);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_GetCameraAnchorHandle(IntPtr anchorName, ref IntPtr anchorHandle);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_GetCameraAnchorType(IntPtr anchorHandle, ref CameraAnchorType anchorType);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_CreateCustomCameraAnchor(IntPtr anchorName, ref IntPtr anchorHandle);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_DestroyCustomCameraAnchor(IntPtr anchorHandle);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_GetCustomCameraAnchorPose(IntPtr anchorHandle, ref Posef pose);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_SetCustomCameraAnchorPose(IntPtr anchorHandle, Posef pose);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_GetCameraMinMaxDistance(IntPtr anchorHandle, ref double minDistance, ref double maxDistance);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_SetCameraMinMaxDistance(IntPtr anchorHandle, double minDistance, double maxDistance);
	}

	private static class OVRP_1_50_0
	{
		public static readonly System.Version version = new System.Version(1, 50, 0);
	}

	private static class OVRP_1_51_0
	{
		public static readonly System.Version version = new System.Version(1, 51, 0);
	}

	private static class OVRP_1_52_0
	{
		public static readonly System.Version version = new System.Version(1, 52, 0);
	}

	private static class OVRP_1_53_0
	{
		public static readonly System.Version version = new System.Version(1, 53, 0);
	}

	private static class OVRP_1_54_0
	{
		public static readonly System.Version version = new System.Version(1, 54, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_SetPlatformInitialized();
	}

	private static class OVRP_1_55_0
	{
		public static readonly System.Version version = new System.Version(1, 55, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetSkeleton2(SkeletonType skeletonType, out Skeleton2Internal skeleton);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_PollEvent(ref EventDataBuffer eventDataBuffer);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetNativeXrApiType(out XrApi xrApi);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetNativeOpenXRHandles(out UInt64 xrInstance, out UInt64 xrSession);
	}

	private static class OVRP_1_55_1
	{
		public static readonly System.Version version = new System.Version(1, 55, 1);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_PollEvent2(ref EventType eventType, ref IntPtr eventData);
	}

	private static class OVRP_1_56_0
	{
		public static readonly System.Version version = new System.Version(1, 56, 0);
	}

	private static class OVRP_1_57_0
	{
		public static readonly System.Version version = new System.Version(1, 57, 0);


		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_GetPlatformCameraMode(out Media.PlatformCameraMode platformCameraMode);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_SetPlatformCameraMode(Media.PlatformCameraMode platformCameraMode);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetEyeFovPremultipliedAlphaMode(Bool enabled);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetEyeFovPremultipliedAlphaMode(ref Bool enabled);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetKeyboardOverlayUV(Vector2f uv);
	}

	private static class OVRP_1_58_0
	{
		public static readonly System.Version version = new System.Version(1, 58, 0);
	}

	private static class OVRP_1_59_0
	{
		public static readonly System.Version version = new System.Version(1, 59, 0);
	}

	private static class OVRP_1_60_0
	{
		public static readonly System.Version version = new System.Version(1, 60, 0);

	}

	private static class OVRP_1_61_0
	{
		public static readonly System.Version version = new System.Version(1, 61, 0);
	}

	private static class OVRP_1_62_0
	{
		public static readonly System.Version version = new System.Version(1, 62, 0);
	}

	private static class OVRP_1_63_0
	{
		public static readonly System.Version version = new System.Version(1, 63, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_InitializeInsightPassthrough();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_ShutdownInsightPassthrough();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Bool ovrp_GetInsightPassthroughInitialized();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetInsightPassthroughStyle(int layerId, InsightPassthroughStyle style);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_CreateInsightTriangleMesh(
			int layerId, IntPtr vertices, int vertexCount, IntPtr triangles, int triangleCount, out ulong meshHandle);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_DestroyInsightTriangleMesh(ulong meshHandle);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_AddInsightPassthroughSurfaceGeometry(int layerId, ulong meshHandle, Matrix4x4 T_world_model, out ulong geometryInstanceHandle);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_DestroyInsightPassthroughGeometryInstance(ulong geometryInstanceHandle);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_UpdateInsightPassthroughGeometryTransform(ulong geometryInstanceHandle, Matrix4x4 T_world_model);
	}
#endif // !OVRPLUGIN_UNSUPPORTED_PLATFORM

    private static class OVRP_1_64_0
	{
		public static readonly System.Version version = new System.Version(1, 64, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_LocateSpace(ref Posef location, ref UInt64 space, TrackingOrigin trackingOrigin);
	}

	private static class OVRP_1_65_0
	{
		public static readonly System.Version version = new System.Version(1, 65, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_KtxLoadFromMemory(ref IntPtr data, uint length, ref System.IntPtr texture);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_KtxTextureWidth(IntPtr texture, ref uint width);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_KtxTextureHeight(IntPtr texture, ref uint height);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_KtxTranscode(IntPtr texture, uint format);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_KtxGetTextureData(IntPtr texture, IntPtr data, uint bufferSize);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_KtxTextureSize(IntPtr texture, ref uint size);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_KtxDestroy(IntPtr texture);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_DestroySpace(ref UInt64 space);
	}

	private static class OVRP_1_66_0
	{
		public static readonly System.Version version = new System.Version(1, 66, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetInsightPassthroughInitializationState();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_Media_IsCastingToRemoteClient(out Bool isCasting);
	}

	private static class OVRP_1_67_0
	{
		public static readonly System.Version version = new System.Version(1, 67, 0);
	}

	private static class OVRP_1_68_0
	{
		public static readonly System.Version version = new System.Version(1, 68, 0);

		public const int OVRP_RENDER_MODEL_MAX_PATH_LENGTH = 256;
		public const int OVRP_RENDER_MODEL_MAX_NAME_LENGTH = 64;

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_LoadRenderModel(UInt64 modelKey, uint bufferInputCapacity, ref uint bufferCountOutput, IntPtr buffer);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetRenderModelPaths(uint index, IntPtr path);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetRenderModelProperties(string path, out RenderModelPropertiesInternal properties);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetInsightPassthroughKeyboardHandsIntensity(int layerId, InsightPassthroughKeyboardHandsIntensity intensity);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_StartKeyboardTracking(UInt64 trackedKeyboardId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_StopKeyboardTracking();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetSystemKeyboardDescription(TrackedKeyboardQueryFlags keyboardQueryFlags, out KeyboardDescription keyboardDescription);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetKeyboardState(Step stepId, int frameIndex, out KeyboardState keyboardState);
	}

	private static class OVRP_1_69_0
	{
		public static readonly System.Version version = new System.Version(1, 69, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetNodePoseStateImmediate(Node nodeId, out PoseStatef nodePoseState);

	}

	private static class OVRP_1_70_0
	{
		public static readonly System.Version version = new System.Version(1, 70, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetLogCallback2(LogCallback2DelegateType logCallback);
	}

	private static class OVRP_1_71_0
	{
		public static readonly System.Version version = new System.Version(1, 71, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_IsInsightPassthroughSupported(ref Bool supported);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern void ovrp_UnityOpenXR_SetClientVersion(int majorVersion, int minorVersion, int patchVersion);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern IntPtr ovrp_UnityOpenXR_HookGetInstanceProcAddr(IntPtr func);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_UnityOpenXR_OnInstanceCreate(UInt64 xrInstance);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern void ovrp_UnityOpenXR_OnInstanceDestroy(UInt64 xrInstance);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern void ovrp_UnityOpenXR_OnSessionCreate(UInt64 xrSession);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern void ovrp_UnityOpenXR_OnAppSpaceChange(UInt64 xrSpace);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern void ovrp_UnityOpenXR_OnSessionStateChange(int oldState, int newState);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern void ovrp_UnityOpenXR_OnSessionBegin(UInt64 xrSession);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern void ovrp_UnityOpenXR_OnSessionEnd(UInt64 xrSession);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern void ovrp_UnityOpenXR_OnSessionExiting(UInt64 xrSession);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern void ovrp_UnityOpenXR_OnSessionDestroy(UInt64 xrSession);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetSuggestedCpuPerformanceLevel(ProcessorPerformanceLevel perfLevel);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetSuggestedCpuPerformanceLevel(out ProcessorPerformanceLevel perfLevel);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetSuggestedGpuPerformanceLevel(ProcessorPerformanceLevel perfLevel);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetSuggestedGpuPerformanceLevel(out ProcessorPerformanceLevel perfLevel);


	}

	private static class OVRP_1_72_0
	{
		public static readonly System.Version version = new System.Version(1, 72, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_CreateSpatialAnchor(ref SpatialAnchorCreateInfo createInfo, out UInt64 requestId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetSpaceComponentStatus(ref UInt64 space, SpaceComponentType componentType, Bool enable, double timeout, out UInt64 requestId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetSpaceComponentStatus(ref UInt64 space, SpaceComponentType componentType, out Bool enabled, out Bool changePending);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_EnumerateSpaceSupportedComponents(ref UInt64 space, uint componentTypesCapacityInput, out uint componentTypesCountOutput, [MarshalAs(UnmanagedType.LPArray), In, Out] SpaceComponentType[] componentTypes);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SaveSpace(ref UInt64 space, SpaceStorageLocation location, SpaceStoragePersistenceMode mode, out UInt64 requestId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_QuerySpaces(ref SpaceQueryInfo queryInfo, out UInt64 requestId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_RetrieveSpaceQueryResults(ref UInt64 requestId, UInt32 resultCapacityInput, ref UInt32 resultCountOutput, System.IntPtr results);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_EraseSpace(ref UInt64 space, SpaceStorageLocation location, out UInt64 requestId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetSpaceContainer(ref UInt64 space, ref SpaceContainerInternal containerInternal);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetSpaceBoundingBox2D(ref UInt64 space, out Rectf rect);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetSpaceBoundingBox3D(ref UInt64 space, out Boundsf bounds);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result  ovrp_GetSpaceSemanticLabels(ref UInt64 space, ref SpaceSemanticLabelInternal labelsInternal);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetSpaceRoomLayout(ref UInt64 space, ref RoomLayoutInternal roomLayoutInternal);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetSpaceBoundary2D(ref UInt64 space, ref PolygonalBoundary2DInternal boundaryInternal);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_RequestSceneCapture(ref SceneCaptureRequestInternal request, out UInt64 requestId);
	}

	private static class OVRP_1_73_0
	{
		public static readonly System.Version version = new System.Version(1, 73, 0);

	}

	private static class OVRP_1_74_0
	{
		public static readonly System.Version version = new System.Version(1, 74, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetSpaceUuid(in UInt64 space, out Guid uuid);


		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetRenderModelProperties2(string path, RenderModelFlags flags, out RenderModelPropertiesInternal properties);
	}

	private static class OVRP_1_75_0
	{
		public static readonly System.Version version = new System.Version(1, 75, 0);
	}

	private static class OVRP_1_76_0
	{
		public static readonly System.Version version = new System.Version(1, 76, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetNodePoseStateAtTime(double time, Node nodeId, out PoseStatef nodePoseState);

	}

	private static class OVRP_1_78_0
	{
		public static readonly System.Version version = new System.Version(1, 78, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetPassthroughCapabilityFlags(ref PassthroughCapabilityFlags capabilityFlags);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetFoveationEyeTrackedSupported(out Bool foveationSupported);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetFoveationEyeTracked(out Bool isEyeTrackedFoveation);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetFoveationEyeTracked(Bool isEyeTrackedFoveation);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_StartFaceTracking();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_StopFaceTracking();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_StartBodyTracking();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_StopBodyTracking();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_StartEyeTracking();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_StopEyeTracking();

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetEyeTrackingSupported(out Bool eyeTrackingSupported);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetFaceTrackingSupported(out Bool faceTrackingSupported);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetBodyTrackingEnabled(out Bool value);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetBodyTrackingSupported(out Bool value);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetBodyState(Step stepId, int frameIndex, out BodyStateInternal bodyState);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetFaceTrackingEnabled(out Bool faceTrackingEnabled);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetFaceState(Step stepId, int frameIndex, out FaceStateInternal faceState);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetEyeTrackingEnabled(out Bool eyeTrackingEnabled);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetEyeGazesState(Step stepId, int frameIndex, out EyeGazesStateInternal eyeGazesState);


		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetControllerState5(uint controllerMask, ref ControllerState5 controllerState);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetControllerLocalizedVibration(Controller controllerMask, HapticsLocation hapticsLocationMask, float frequency, float amplitude);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetLocalDimmingSupported(out Bool localDimmingSupported);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetLocalDimming(Bool localDimmingMode);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetLocalDimming(out Bool localDimmingMode);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetCurrentInteractionProfile(Hand hand, out InteractionProfile interactionProfile);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetControllerHapticsAmplitudeEnvelope(
			Controller controllerMask,
			HapticsAmplitudeEnvelopeVibration hapticsVibration);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_SetControllerHapticsPcm(
			Controller controllerMask,
			HapticsPcmVibration hapticsVibration);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetControllerSampleRateHz(Controller controller, out float sampleRateHz);
	}

	private static class OVRP_1_79_0
	{
		public static readonly System.Version version = new System.Version(1, 79, 0);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern unsafe Result ovrp_ShareSpaces(UInt64* spaces, UInt32 numSpaces, ulong* userHandles,
			UInt32 numUsers, out UInt64 requestId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern unsafe Result ovrp_SaveSpaceList(UInt64* spaces, UInt32 numSpaces,
			SpaceStorageLocation location, out UInt64 requestId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_GetSpaceUserId(in UInt64 spaceUserHandle, out UInt64 spaceUserId);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_CreateSpaceUser(in UInt64 spaceUserId, out UInt64 spaceUserHandle);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_DestroySpaceUser(in UInt64 userHandle);

		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_LocateSpace2(out SpaceLocationf location, in UInt64 space, TrackingOrigin trackingOrigin);



		[DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
		public static extern Result ovrp_DeclareUser(in UInt64 userId, out UInt64 userHandle);



	}

	private static class OVRP_1_81_0
	{
		public static readonly System.Version version = new System.Version(1, 81, 0);

	}
	/* INSERT NEW OVRP CLASS ABOVE THIS LINE */
}
