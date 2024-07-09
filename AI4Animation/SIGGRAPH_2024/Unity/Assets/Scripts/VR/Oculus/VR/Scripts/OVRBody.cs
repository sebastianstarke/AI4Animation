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
using UnityEngine;

/// <summary>
/// Manages data related to body tracking.
/// </summary>
/// <remarks>
/// Typically, you would use this in conjunction with an <see cref="OVRSkeleton"/> and/or
/// <see cref="OVRSkeletonRenderer"/>.
/// </remarks>
public class OVRBody : MonoBehaviour,
    OVRSkeleton.IOVRSkeletonDataProvider,
    OVRSkeletonRenderer.IOVRSkeletonRendererDataProvider
{
    private OVRPlugin.BodyState _bodyState;

    private OVRPlugin.Quatf[] _boneRotations;

    private OVRPlugin.Vector3f[] _boneTranslations;

    private bool _dataChangedSinceLastQuery;

    private bool _hasData;

    private const OVRPermissionsRequester.Permission BodyTrackingPermission = OVRPermissionsRequester.Permission.BodyTracking;
    private Action<string> _onPermissionGranted;
    private static int _trackingInstanceCount;

    /// <summary>
    /// The raw <see cref="BodyState"/> data used to populate the <see cref="OVRSkeleton"/>.
    /// </summary>
    public OVRPlugin.BodyState? BodyState => _hasData ? _bodyState : default(OVRPlugin.BodyState?);

    private void Awake()
    {
        _onPermissionGranted = OnPermissionGranted;
    }

    private void OnEnable()
    {
        _trackingInstanceCount++;
        _dataChangedSinceLastQuery = false;
        _hasData = false;

        if (!StartBodyTracking())
        {
            enabled = false;
            return;
        }

        if (OVRPlugin.nativeXrApi == OVRPlugin.XrApi.OpenXR)
        {
            GetBodyState(OVRPlugin.Step.Render);
        }
        else
        {
            enabled = false;
            Debug.LogWarning($"[{nameof(OVRBody)}] Body tracking is only supported by OpenXR and is unavailable.");
        }
    }

    private void OnPermissionGranted(string permissionId)
    {
        if (permissionId == OVRPermissionsRequester.GetPermissionId(BodyTrackingPermission))
        {
            OVRPermissionsRequester.PermissionGranted -= _onPermissionGranted;
            enabled = true;
        }
    }

    private bool StartBodyTracking()
    {
        if (!OVRPermissionsRequester.IsPermissionGranted(BodyTrackingPermission))
        {
            OVRPermissionsRequester.PermissionGranted -= _onPermissionGranted;
            OVRPermissionsRequester.PermissionGranted += _onPermissionGranted;
            return false;
        }

        if (!OVRPlugin.StartBodyTracking())
        {
            Debug.LogWarning($"[{nameof(OVRBody)}] Failed to start body tracking.");
            return false;
        }

        return true;
    }

    private void OnDisable()
    {
        if (--_trackingInstanceCount == 0)
        {
            OVRPlugin.StopBodyTracking();
        }
    }

    private void OnDestroy()
    {
        OVRPermissionsRequester.PermissionGranted -= _onPermissionGranted;
    }

    private void Update() => GetBodyState(OVRPlugin.Step.Render);

    private void GetBodyState(OVRPlugin.Step step)
    {
        if (OVRPlugin.GetBodyState(step, ref _bodyState))
        {
            _hasData = true;
            _dataChangedSinceLastQuery = true;
        }
        else
        {
            _hasData = false;
        }
    }

    OVRSkeleton.SkeletonType OVRSkeleton.IOVRSkeletonDataProvider.GetSkeletonType() => OVRSkeleton.SkeletonType.Body;

    OVRSkeleton.SkeletonPoseData OVRSkeleton.IOVRSkeletonDataProvider.GetSkeletonPoseData()
    {
        if (!_hasData) return default;

        if (_dataChangedSinceLastQuery)
        {
            // Make sure arrays have been allocated
            Array.Resize(ref _boneRotations, _bodyState.JointLocations.Length);
            Array.Resize(ref _boneTranslations, _bodyState.JointLocations.Length);

            // Copy joint poses into bone arrays
            for (var i = 0; i < _bodyState.JointLocations.Length; i++)
            {
                var jointLocation = _bodyState.JointLocations[i];
                if (jointLocation.OrientationValid)
                {
                    _boneRotations[i] = jointLocation.Pose.Orientation;
                }

                if (jointLocation.PositionValid)
                {
                    _boneTranslations[i] = jointLocation.Pose.Position;
                }
            }

            _dataChangedSinceLastQuery = false;
        }

        return new OVRSkeleton.SkeletonPoseData
        {
            IsDataValid = true,
            IsDataHighConfidence = _bodyState.Confidence > .5f,
            RootPose = _bodyState.JointLocations[(int)OVRPlugin.BoneId.Body_Root].Pose,
            RootScale = 1.0f,
            BoneRotations = _boneRotations,
            BoneTranslations = _boneTranslations,
            SkeletonChangedCount = (int)_bodyState.SkeletonChangedCount,
        };
    }

    OVRSkeletonRenderer.SkeletonRendererData
        OVRSkeletonRenderer.IOVRSkeletonRendererDataProvider.GetSkeletonRendererData() => _hasData
    ? new OVRSkeletonRenderer.SkeletonRendererData
    {
        RootScale = 1.0f,
        IsDataValid = true,
        IsDataHighConfidence = true,
        ShouldUseSystemGestureMaterial = false,
    }
    : default;
}
