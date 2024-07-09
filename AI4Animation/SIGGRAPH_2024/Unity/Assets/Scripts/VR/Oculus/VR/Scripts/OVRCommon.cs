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
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

#if USING_XR_SDK
using UnityEngine.XR;
using UnityEngine.Experimental.XR;
#endif

using InputTracking = UnityEngine.XR.InputTracking;
using Node = UnityEngine.XR.XRNode;
using NodeState = UnityEngine.XR.XRNodeState;
using Device = UnityEngine.XR.XRDevice;

/// <summary>
/// Miscellaneous extension methods that any script can use.
/// </summary>
public static partial class OVRExtensions
{
	/// <summary>
	/// Converts the given world-space transform to an OVRPose in tracking space.
	/// </summary>
	public static OVRPose ToTrackingSpacePose(this Transform transform, Camera camera)
	{
		// Initializing to identity, but for all Oculus headsets, down below the pose will be initialized to the runtime's pose value, so identity will never be returned.
		OVRPose headPose = OVRPose.identity;

		Vector3 pos;
		Quaternion rot;
		if (OVRNodeStateProperties.GetNodeStatePropertyVector3(Node.Head, NodeStatePropertyType.Position, OVRPlugin.Node.Head, OVRPlugin.Step.Render, out pos))
			headPose.position = pos;
		if (OVRNodeStateProperties.GetNodeStatePropertyQuaternion(Node.Head, NodeStatePropertyType.Orientation, OVRPlugin.Node.Head, OVRPlugin.Step.Render, out rot))
			headPose.orientation = rot;

		var ret = headPose * transform.ToHeadSpacePose(camera);

		return ret;
	}

	/// <summary>
	/// Converts the given pose from tracking-space to world-space.
	/// </summary>
	[Obsolete("ToWorldSpacePose should be invoked with an explicit mainCamera parameter")]
	public static OVRPose ToWorldSpacePose(this OVRPose trackingSpacePose)
	{
		return ToWorldSpacePose(trackingSpacePose, Camera.main);
	}

	/// <summary>
	/// Converts the given pose from tracking-space to world-space.
	/// </summary>
	public static OVRPose ToWorldSpacePose(this OVRPose trackingSpacePose, Camera mainCamera)
	{
		// Transform from tracking-Space to head-Space
		OVRPose poseInHeadSpace = trackingSpacePose.ToHeadSpacePose();

		// Transform from head space to world space
		var cameraTransform = mainCamera.transform.localToWorldMatrix;
		var headSpaceTransform = Matrix4x4.TRS(
			poseInHeadSpace.position, poseInHeadSpace.orientation, Vector3.one);
		var worldSpaceTransform = cameraTransform * headSpaceTransform;
		return new OVRPose
		{
			position = worldSpaceTransform.GetColumn(3),
			orientation = worldSpaceTransform.rotation
		};
	}

	/// <summary>
	/// Converts the given pose from tracking-space to head-space.
	/// </summary>
	public static OVRPose ToHeadSpacePose(this OVRPose trackingSpacePose)
	{
		OVRPose headPose = OVRPose.identity;

		Vector3 pos;
		Quaternion rot;
		if (OVRNodeStateProperties.GetNodeStatePropertyVector3(UnityEngine.XR.XRNode.Head, NodeStatePropertyType.Position, OVRPlugin.Node.Head, OVRPlugin.Step.Render, out pos))
			headPose.position = pos;
		if (OVRNodeStateProperties.GetNodeStatePropertyQuaternion(UnityEngine.XR.XRNode.Head, NodeStatePropertyType.Orientation, OVRPlugin.Node.Head, OVRPlugin.Step.Render, out rot))
			headPose.orientation = rot;

		OVRPose poseInHeadSpace = headPose.Inverse() * trackingSpacePose;

		return poseInHeadSpace;
	}

	/// <summary>
	/// Converts the given world-space transform to an OVRPose in head space.
	/// </summary>
	public static OVRPose ToHeadSpacePose(this Transform transform, Camera camera)
	{
		return camera.transform.ToOVRPose().Inverse() * transform.ToOVRPose();
	}

	public static OVRPose ToOVRPose(this Transform t, bool isLocal = false)
	{
		OVRPose pose;
		pose.orientation = (isLocal) ? t.localRotation : t.rotation;
		pose.position = (isLocal) ? t.localPosition : t.position;
		return pose;
	}

	public static void FromOVRPose(this Transform t, OVRPose pose, bool isLocal = false)
	{
		if (isLocal)
		{
			t.localRotation = pose.orientation;
			t.localPosition = pose.position;
		}
		else
		{
			t.rotation = pose.orientation;
			t.position = pose.position;
		}
	}

	public static OVRPose ToOVRPose(this OVRPlugin.Posef p)
	{
		return new OVRPose()
		{
			position = new Vector3(p.Position.x, p.Position.y, -p.Position.z),
			orientation = new Quaternion(-p.Orientation.x, -p.Orientation.y, p.Orientation.z, p.Orientation.w)
		};
	}

	public static OVRTracker.Frustum ToFrustum(this OVRPlugin.Frustumf f)
	{
		return new OVRTracker.Frustum()
		{
			nearZ = f.zNear,
			farZ = f.zFar,

			fov = new Vector2()
			{
				x = Mathf.Rad2Deg * f.fovX,
				y = Mathf.Rad2Deg * f.fovY
			}
		};
	}

	public static Color FromColorf(this OVRPlugin.Colorf c)
	{
		return new Color() { r = c.r, g = c.g, b = c.b, a = c.a };
	}

	public static OVRPlugin.Colorf ToColorf(this Color c)
	{
		return new OVRPlugin.Colorf() { r = c.r, g = c.g, b = c.b, a = c.a };
	}

	public static Vector3 FromVector3f(this OVRPlugin.Vector3f v)
	{
		return new Vector3() { x = v.x, y = v.y, z = v.z };
	}

	public static Vector3 FromFlippedXVector3f(this OVRPlugin.Vector3f v)
	{
		return new Vector3() { x = -v.x, y = v.y, z = v.z };
	}

	public static Vector3 FromFlippedZVector3f(this OVRPlugin.Vector3f v)
	{
		return new Vector3() { x = v.x, y = v.y, z = -v.z };
	}

	public static OVRPlugin.Vector3f ToVector3f(this Vector3 v)
	{
		return new OVRPlugin.Vector3f() { x = v.x, y = v.y, z = v.z };
	}

	public static OVRPlugin.Vector3f ToFlippedXVector3f(this Vector3 v)
	{
		return new OVRPlugin.Vector3f() { x = -v.x, y = v.y, z = v.z };
	}

	public static OVRPlugin.Vector3f ToFlippedZVector3f(this Vector3 v)
	{
		return new OVRPlugin.Vector3f() { x = v.x, y = v.y, z = -v.z };
	}

	public static Vector4 FromVector4f(this OVRPlugin.Vector4f v)
	{
		return new Vector4() { x = v.x, y = v.y, z = v.z, w = v.w };
	}

	public static OVRPlugin.Vector4f ToVector4f(this Vector4 v)
	{
		return new OVRPlugin.Vector4f() { x = v.x, y = v.y, z = v.z, w = v.w };
	}

	public static Quaternion FromQuatf(this OVRPlugin.Quatf q)
	{
		return new Quaternion() { x = q.x, y = q.y, z = q.z, w = q.w };
	}

	public static Quaternion FromFlippedXQuatf(this OVRPlugin.Quatf q)
	{
		return new Quaternion() { x = q.x, y = -q.y, z = -q.z, w = q.w };
	}

	public static Quaternion FromFlippedZQuatf(this OVRPlugin.Quatf q)
	{
		return new Quaternion() { x = -q.x, y = -q.y, z = q.z, w = q.w };
	}

	public static OVRPlugin.Quatf ToQuatf(this Quaternion q)
	{
		return new OVRPlugin.Quatf() { x = q.x, y = q.y, z = q.z, w = q.w };
	}

	public static OVRPlugin.Quatf ToFlippedXQuatf(this Quaternion q)
	{
		return new OVRPlugin.Quatf() { x = q.x, y = -q.y, z = -q.z, w = q.w };
	}

	public static OVRPlugin.Quatf ToFlippedZQuatf(this Quaternion q)
	{
		return new OVRPlugin.Quatf() { x = -q.x, y = -q.y, z = q.z, w = q.w };
	}

	public static OVR.OpenVR.HmdMatrix34_t ConvertToHMDMatrix34(this Matrix4x4 m)
	{
		OVR.OpenVR.HmdMatrix34_t pose = new OVR.OpenVR.HmdMatrix34_t();

		pose.m0 = m[0, 0];
		pose.m1 = m[0, 1];
		pose.m2 = -m[0, 2];
		pose.m3 = m[0, 3];

		pose.m4 = m[1, 0];
		pose.m5 = m[1, 1];
		pose.m6 = -m[1, 2];
		pose.m7 = m[1, 3];

		pose.m8 = -m[2, 0];
		pose.m9 = -m[2, 1];
		pose.m10 = m[2, 2];
		pose.m11 = -m[2, 3];

		return pose;
	}

	public static Transform FindChildRecursive(this Transform parent, string name)
	{
		for (int i = 0; i < parent.childCount; i++)
		{
			var child = parent.GetChild(i);
			if (child.name.Contains(name))
				return child;

			var result = child.FindChildRecursive(name);
			if (result != null)
				return result;
		}
		return null;
	}

	public static bool Equals(this Gradient gradient, Gradient otherGradient)
	{
		if (gradient.colorKeys.Length != otherGradient.colorKeys.Length || gradient.alphaKeys.Length != otherGradient.alphaKeys.Length)
			return false;

		for (int i = 0; i < gradient.colorKeys.Length; i++)
		{
			GradientColorKey key = gradient.colorKeys[i];
			GradientColorKey otherKey = otherGradient.colorKeys[i];
			if (key.color != otherKey.color || key.time != otherKey.time)
				return false;
		}

		for (int i = 0; i < gradient.alphaKeys.Length; i++)
		{
			GradientAlphaKey key = gradient.alphaKeys[i];
			GradientAlphaKey otherKey = otherGradient.alphaKeys[i];
			if (key.alpha != otherKey.alpha || key.time != otherKey.time)
				return false;
		}

		return true;
	}

	public static void CopyFrom(this Gradient gradient, Gradient otherGradient)
	{
		GradientColorKey[] colorKeys = new GradientColorKey[otherGradient.colorKeys.Length];
		for (int i = 0; i < colorKeys.Length; i++)
		{
			Color col = otherGradient.colorKeys[i].color;
			colorKeys[i].color = new Color(col.r, col.g, col.b, col.a);
			colorKeys[i].time = otherGradient.colorKeys[i].time;
		}

		GradientAlphaKey[] alphaKeys = new GradientAlphaKey[otherGradient.alphaKeys.Length];
		for (int i = 0; i < alphaKeys.Length; i++)
		{
			alphaKeys[i].alpha = otherGradient.alphaKeys[i].alpha;
			alphaKeys[i].time = otherGradient.alphaKeys[i].time;
		}

		gradient.SetKeys(colorKeys, alphaKeys);
	}
}

//4 types of node state properties that can be queried with UnityEngine.XR
public enum NodeStatePropertyType
{
	Acceleration,
	AngularAcceleration,
	Velocity,
	AngularVelocity,
	Position,
	Orientation
}

public static class OVRNodeStateProperties
{
	private static List<NodeState> nodeStateList = new List<NodeState>();

	public static bool IsHmdPresent()
	{
		if (OVRManager.OVRManagerinitialized && OVRManager.loadedXRDevice == OVRManager.XRDevice.Oculus)
			return OVRPlugin.hmdPresent;
#if USING_XR_SDK
		XRDisplaySubsystem currentDisplaySubsystem = OVRManager.GetCurrentDisplaySubsystem();
		if (currentDisplaySubsystem != null)
			return currentDisplaySubsystem.running;				//In 2019.3, this should be changed to currentDisplaySubsystem.isConnected, but this is a fine placeholder for now.
		return false;
#elif REQUIRES_XR_SDK
		return false;
#else
		return Device.isPresent;
#endif
	}

	public static bool GetNodeStatePropertyVector3(Node nodeType, NodeStatePropertyType propertyType, OVRPlugin.Node ovrpNodeType, OVRPlugin.Step stepType, out Vector3 retVec)
	{
		retVec = Vector3.zero;
		switch (propertyType)
		{
			case NodeStatePropertyType.Acceleration:
				if (OVRManager.loadedXRDevice == OVRManager.XRDevice.Oculus)
				{
					retVec = OVRPlugin.GetNodeAcceleration(ovrpNodeType, stepType).FromFlippedZVector3f();
					return true;
				}
				if (GetUnityXRNodeStateVector3(nodeType, NodeStatePropertyType.Acceleration, out retVec))
					return true;
				break;

			case NodeStatePropertyType.AngularAcceleration:
				if (OVRManager.loadedXRDevice == OVRManager.XRDevice.Oculus)
				{
					retVec = OVRPlugin.GetNodeAngularAcceleration(ovrpNodeType, stepType).FromFlippedZVector3f();
					return true;
				}
				if (GetUnityXRNodeStateVector3(nodeType, NodeStatePropertyType.AngularAcceleration, out retVec))
					return true;
				break;

			case NodeStatePropertyType.Velocity:
				if (OVRManager.loadedXRDevice == OVRManager.XRDevice.Oculus)
				{
					retVec = OVRPlugin.GetNodeVelocity(ovrpNodeType, stepType).FromFlippedZVector3f();
					return true;
				}
				if (GetUnityXRNodeStateVector3(nodeType, NodeStatePropertyType.Velocity, out retVec))
					return true;
				break;

			case NodeStatePropertyType.AngularVelocity:
				if (OVRManager.loadedXRDevice == OVRManager.XRDevice.Oculus)
				{
					retVec = OVRPlugin.GetNodeAngularVelocity(ovrpNodeType, stepType).FromFlippedZVector3f();
					return true;
				}
				if (GetUnityXRNodeStateVector3(nodeType, NodeStatePropertyType.AngularVelocity, out retVec))
					return true;
				break;

			case NodeStatePropertyType.Position:
				if (OVRManager.loadedXRDevice == OVRManager.XRDevice.Oculus)
				{
					retVec = OVRPlugin.GetNodePose(ovrpNodeType, stepType).ToOVRPose().position;
					return true;
				}
				if (GetUnityXRNodeStateVector3(nodeType, NodeStatePropertyType.Position, out retVec))
					return true;
				break;
		}

		return false;
	}

	public static bool GetNodeStatePropertyQuaternion(Node nodeType, NodeStatePropertyType propertyType, OVRPlugin.Node ovrpNodeType, OVRPlugin.Step stepType, out Quaternion retQuat)
	{
		retQuat = Quaternion.identity;
		switch (propertyType)
		{
			case NodeStatePropertyType.Orientation:
				if (OVRManager.loadedXRDevice == OVRManager.XRDevice.Oculus)
				{
					retQuat = OVRPlugin.GetNodePose(ovrpNodeType, stepType).ToOVRPose().orientation;
					return true;
				}
				if (GetUnityXRNodeStateQuaternion(nodeType, NodeStatePropertyType.Orientation, out retQuat))
					return true;
				break;
		}
		return false;
	}

	private static bool ValidateProperty(Node nodeType, ref NodeState requestedNodeState)
	{
		InputTracking.GetNodeStates(nodeStateList);

		if (nodeStateList.Count == 0)
			return false;

		bool nodeStateFound = false;
		requestedNodeState = nodeStateList[0];

		for (int i = 0; i < nodeStateList.Count; i++)
		{
			if (nodeStateList[i].nodeType == nodeType)
			{
				requestedNodeState = nodeStateList[i];
				nodeStateFound = true;
				break;
			}
		}

		return nodeStateFound;
	}

	private static bool GetUnityXRNodeStateVector3(Node nodeType, NodeStatePropertyType propertyType, out Vector3 retVec)
	{
		retVec = Vector3.zero;

		NodeState requestedNodeState = default(NodeState);

		if (!ValidateProperty(nodeType, ref requestedNodeState))
			return false;

		if (propertyType == NodeStatePropertyType.Acceleration)
		{
			if (requestedNodeState.TryGetAcceleration(out retVec))
			{
				return true;
			}
		}
		else if (propertyType == NodeStatePropertyType.AngularAcceleration)
		{
			if (requestedNodeState.TryGetAngularAcceleration(out retVec))
			{
				return true;
			}
		}
		else if (propertyType == NodeStatePropertyType.Velocity)
		{
			if (requestedNodeState.TryGetVelocity(out retVec))
			{
				return true;
			}
		}
		else if (propertyType == NodeStatePropertyType.AngularVelocity)
		{
			if (requestedNodeState.TryGetAngularVelocity(out retVec))
			{
				return true;
			}
		}
		else if (propertyType == NodeStatePropertyType.Position)
		{
			if (requestedNodeState.TryGetPosition(out retVec))
			{
				return true;
			}
		}

		return false;
	}

	private static bool GetUnityXRNodeStateQuaternion(Node nodeType, NodeStatePropertyType propertyType, out Quaternion retQuat)
	{
		retQuat = Quaternion.identity;

		NodeState requestedNodeState = default(NodeState);

		if (!ValidateProperty(nodeType, ref requestedNodeState))
			return false;

		if (propertyType == NodeStatePropertyType.Orientation)
		{
			if (requestedNodeState.TryGetRotation(out retQuat))
			{
				return true;
			}
		}

		return false;
	}

}

/// <summary>
/// An affine transformation built from a Unity position and orientation.
/// </summary>
[System.Serializable]
public struct OVRPose
{
	/// <summary>
	/// A pose with no translation or rotation.
	/// </summary>
	public static OVRPose identity
	{
		get {
			return new OVRPose()
			{
				position = Vector3.zero,
				orientation = Quaternion.identity
			};
		}
	}

	public override bool Equals(System.Object obj)
	{
		return obj is OVRPose && this == (OVRPose)obj;
	}

	public override int GetHashCode()
	{
		return position.GetHashCode() ^ orientation.GetHashCode();
	}

	public static bool operator ==(OVRPose x, OVRPose y)
	{
		return x.position == y.position && x.orientation == y.orientation;
	}

	public static bool operator !=(OVRPose x, OVRPose y)
	{
		return !(x == y);
	}

	/// <summary>
	/// The position.
	/// </summary>
	public Vector3 position;

	/// <summary>
	/// The orientation.
	/// </summary>
	public Quaternion orientation;

	/// <summary>
	/// Multiplies two poses.
	/// </summary>
	public static OVRPose operator*(OVRPose lhs, OVRPose rhs)
	{
		var ret = new OVRPose();
		ret.position = lhs.position + lhs.orientation * rhs.position;
		ret.orientation = lhs.orientation * rhs.orientation;
		return ret;
	}

	/// <summary>
	/// Computes the inverse of the given pose.
	/// </summary>
	public OVRPose Inverse()
	{
		OVRPose ret;
		ret.orientation = Quaternion.Inverse(orientation);
		ret.position = ret.orientation * -position;
		return ret;
	}

	/// <summary>
	/// Converts the pose from left- to right-handed or vice-versa.
	/// </summary>
	public OVRPose flipZ()
	{
		var ret = this;
		ret.position.z = -ret.position.z;
		ret.orientation.z = -ret.orientation.z;
		ret.orientation.w = -ret.orientation.w;
		return ret;
	}

	// Warning: this function is not a strict reverse of OVRPlugin.Posef.ToOVRPose(), even after flipZ()
	public OVRPlugin.Posef ToPosef_Legacy()
	{
		return new OVRPlugin.Posef()
		{
			Position = position.ToVector3f(),
			Orientation = orientation.ToQuatf()
		};
	}

	public OVRPlugin.Posef ToPosef()
	{
		OVRPlugin.Posef result = new OVRPlugin.Posef();
		result.Position.x = position.x;
		result.Position.y = position.y;
		result.Position.z = -position.z;
		result.Orientation.x = -orientation.x;
		result.Orientation.y = -orientation.y;
		result.Orientation.z = orientation.z;
		result.Orientation.w = orientation.w;
		return result;
	}

	public OVRPose Rotate180AlongX()
	{
		var ret = this;
		ret.orientation *= Quaternion.Euler(180, 0, 0);
		return ret;
	}
}

/// <summary>
/// Encapsulates an 8-byte-aligned of unmanaged memory.
/// </summary>
public class OVRNativeBuffer : IDisposable
{
	private bool disposed = false;
	private int m_numBytes = 0;
	private IntPtr m_ptr = IntPtr.Zero;

	/// <summary>
	/// Creates a buffer of the specified size.
	/// </summary>
	public OVRNativeBuffer(int numBytes)
	{
		Reallocate(numBytes);
	}

	/// <summary>
	/// Releases unmanaged resources and performs other cleanup operations before the <see cref="OVRNativeBuffer"/> is
	/// reclaimed by garbage collection.
	/// </summary>
	~OVRNativeBuffer()
	{
		Dispose(false);
	}

	/// <summary>
	/// Reallocates the buffer with the specified new size.
	/// </summary>
	public void Reset(int numBytes)
	{
		Reallocate(numBytes);
	}

	/// <summary>
	/// The current number of bytes in the buffer.
	/// </summary>
	public int GetCapacity()
	{
		return m_numBytes;
	}

	/// <summary>
	/// A pointer to the unmanaged memory in the buffer, starting at the given offset in bytes.
	/// </summary>
	public IntPtr GetPointer(int byteOffset = 0)
	{
		if (byteOffset < 0 || byteOffset >= m_numBytes)
			return IntPtr.Zero;
		return (byteOffset == 0) ? m_ptr : new IntPtr(m_ptr.ToInt64() + byteOffset);
	}

	/// <summary>
	/// Releases all resource used by the <see cref="OVRNativeBuffer"/> object.
	/// </summary>
	/// <remarks>Call <see cref="Dispose"/> when you are finished using the <see cref="OVRNativeBuffer"/>. The <see cref="Dispose"/>
	/// method leaves the <see cref="OVRNativeBuffer"/> in an unusable state. After calling <see cref="Dispose"/>, you must
	/// release all references to the <see cref="OVRNativeBuffer"/> so the garbage collector can reclaim the memory that
	/// the <see cref="OVRNativeBuffer"/> was occupying.</remarks>
	public void Dispose()
	{
		Dispose(true);
		GC.SuppressFinalize(this);
	}

	private void Dispose(bool disposing)
	{
		if (disposed)
			return;

		if (disposing)
		{
			// dispose managed resources
		}

		// dispose unmanaged resources
		Release();

		disposed = true;
	}

	private void Reallocate(int numBytes)
	{
		Release();

		if (numBytes > 0)
		{
			m_ptr = Marshal.AllocHGlobal(numBytes);
			m_numBytes = numBytes;
		}
		else
		{
			m_ptr = IntPtr.Zero;
			m_numBytes = 0;
		}
	}

	private void Release()
	{
		if (m_ptr != IntPtr.Zero)
		{
			Marshal.FreeHGlobal(m_ptr);
			m_ptr = IntPtr.Zero;
			m_numBytes = 0;
		}
	}
}
