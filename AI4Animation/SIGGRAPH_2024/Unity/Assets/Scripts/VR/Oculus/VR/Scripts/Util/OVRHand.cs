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

public class OVRHand : MonoBehaviour,
	OVRSkeleton.IOVRSkeletonDataProvider,
	OVRSkeletonRenderer.IOVRSkeletonRendererDataProvider,
	OVRMesh.IOVRMeshDataProvider,
	OVRMeshRenderer.IOVRMeshRendererDataProvider
{
	public enum Hand
	{
		None      = OVRPlugin.Hand.None,
		HandLeft  = OVRPlugin.Hand.HandLeft,
		HandRight = OVRPlugin.Hand.HandRight,
	}

	public enum HandFinger
	{
		Thumb  = OVRPlugin.HandFinger.Thumb,
		Index  = OVRPlugin.HandFinger.Index,
		Middle = OVRPlugin.HandFinger.Middle,
		Ring   = OVRPlugin.HandFinger.Ring,
		Pinky  = OVRPlugin.HandFinger.Pinky,
		Max    = OVRPlugin.HandFinger.Max,
	}

	public enum TrackingConfidence
	{
		Low  = OVRPlugin.TrackingConfidence.Low,
		High = OVRPlugin.TrackingConfidence.High
	}

	[SerializeField]
	private Hand HandType = Hand.None;
	[SerializeField]
	private Transform _pointerPoseRoot = null;
	private GameObject _pointerPoseGO;
	private OVRPlugin.HandState _handState = new OVRPlugin.HandState();

	public bool IsDataValid { get; private set; }
	public bool IsDataHighConfidence { get; private set; }
	public bool IsTracked { get; private set; }
	public bool IsSystemGestureInProgress { get; private set; }
	public bool IsPointerPoseValid { get; private set; }
	public Transform PointerPose { get; private set; }
	public float HandScale { get; private set; }
	public TrackingConfidence HandConfidence { get; private set; }
	public bool IsDominantHand { get; private set; }

	private void Awake()
	{
		_pointerPoseGO = new GameObject();
		PointerPose = _pointerPoseGO.transform;
		if (_pointerPoseRoot != null)
		{
			PointerPose.SetParent(_pointerPoseRoot, false);
		}

		GetHandState(OVRPlugin.Step.Render);
	}

	private void Update()
	{
		GetHandState(OVRPlugin.Step.Render);
	}

	private void FixedUpdate()
	{
		if (OVRPlugin.nativeXrApi != OVRPlugin.XrApi.OpenXR)
		{
			GetHandState(OVRPlugin.Step.Physics);
		}
	}

	private void GetHandState(OVRPlugin.Step step)
	{
		if (OVRPlugin.GetHandState(step, (OVRPlugin.Hand)HandType, ref _handState))
		{
			IsTracked = (_handState.Status & OVRPlugin.HandStatus.HandTracked) != 0;
			IsSystemGestureInProgress = (_handState.Status & OVRPlugin.HandStatus.SystemGestureInProgress) != 0;
			IsPointerPoseValid = (_handState.Status & OVRPlugin.HandStatus.InputStateValid) != 0;
			IsDominantHand = (_handState.Status & OVRPlugin.HandStatus.DominantHand) != 0;
			PointerPose.localPosition = _handState.PointerPose.Position.FromFlippedZVector3f();
			PointerPose.localRotation = _handState.PointerPose.Orientation.FromFlippedZQuatf();
			HandScale = _handState.HandScale;
			HandConfidence = (TrackingConfidence)_handState.HandConfidence;

			IsDataValid = true;
			IsDataHighConfidence = IsTracked && HandConfidence == TrackingConfidence.High;
		}
		else
		{
			IsTracked = false;
			IsSystemGestureInProgress = false;
			IsPointerPoseValid = false;
			PointerPose.localPosition = Vector3.zero;
			PointerPose.localRotation = Quaternion.identity;
			HandScale = 1.0f;
			HandConfidence = TrackingConfidence.Low;

			IsDataValid = false;
			IsDataHighConfidence = false;
		}
	}

	public bool GetFingerIsPinching(HandFinger finger)
	{
		return IsDataValid && (((int)_handState.Pinches & (1 << (int)finger)) != 0);
	}

	public float GetFingerPinchStrength(HandFinger finger)
	{
		if (IsDataValid
			&& _handState.PinchStrength != null
			&& _handState.PinchStrength.Length == (int)OVRPlugin.HandFinger.Max)
		{
			return _handState.PinchStrength[(int)finger];
		}

		return 0.0f;
	}

	public TrackingConfidence GetFingerConfidence(HandFinger finger)
	{
		if (IsDataValid
			&& _handState.FingerConfidences != null
			&& _handState.FingerConfidences.Length == (int)OVRPlugin.HandFinger.Max)
		{
			return (TrackingConfidence)_handState.FingerConfidences[(int)finger];
		}

		return TrackingConfidence.Low;
	}

	OVRSkeleton.SkeletonType OVRSkeleton.IOVRSkeletonDataProvider.GetSkeletonType()
	{
		switch (HandType)
		{
		case Hand.HandLeft:
			return OVRSkeleton.SkeletonType.HandLeft;
		case Hand.HandRight:
			return OVRSkeleton.SkeletonType.HandRight;
		case Hand.None:
		default:
			return OVRSkeleton.SkeletonType.None;
		}
	}

	OVRSkeleton.SkeletonPoseData OVRSkeleton.IOVRSkeletonDataProvider.GetSkeletonPoseData()
	{
		var data = new OVRSkeleton.SkeletonPoseData();

		data.IsDataValid = IsDataValid;
		if (IsDataValid)
		{
			data.RootPose = _handState.RootPose;
			data.RootScale = _handState.HandScale;
			data.BoneRotations = _handState.BoneRotations;
			data.IsDataHighConfidence = IsTracked && HandConfidence == TrackingConfidence.High;
		}

		return data;
	}

    OVRSkeletonRenderer.SkeletonRendererData OVRSkeletonRenderer.IOVRSkeletonRendererDataProvider.GetSkeletonRendererData()
	{
		var data = new OVRSkeletonRenderer.SkeletonRendererData();

		data.IsDataValid = IsDataValid;
		if (IsDataValid)
		{
			data.RootScale = _handState.HandScale;
			data.IsDataHighConfidence = IsTracked && HandConfidence == TrackingConfidence.High;
			data.ShouldUseSystemGestureMaterial = IsSystemGestureInProgress;
		}

		return data;
	}


	OVRMesh.MeshType OVRMesh.IOVRMeshDataProvider.GetMeshType()
	{
		switch (HandType)
		{
		case Hand.None:
			return OVRMesh.MeshType.None;
		case Hand.HandLeft:
			return OVRMesh.MeshType.HandLeft;
		case Hand.HandRight:
			return OVRMesh.MeshType.HandRight;
		default:
			return OVRMesh.MeshType.None;
		}
	}

	OVRMeshRenderer.MeshRendererData OVRMeshRenderer.IOVRMeshRendererDataProvider.GetMeshRendererData()
	{
		var data = new OVRMeshRenderer.MeshRendererData();

		data.IsDataValid = IsDataValid;
		if (IsDataValid)
		{
			data.IsDataHighConfidence = IsTracked && HandConfidence == TrackingConfidence.High;
			data.ShouldUseSystemGestureMaterial = IsSystemGestureInProgress;
		}

		return data;
	}
}
