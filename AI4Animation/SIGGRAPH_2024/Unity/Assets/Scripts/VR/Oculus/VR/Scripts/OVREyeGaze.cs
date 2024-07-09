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
/// This class updates the transform of the GameObject to point toward an eye direction.
/// </summary>
/// <remarks>
/// See <see cref="OVRPlugin.EyeGazeState"/> structure for list of eye state parameters.
/// </remarks>
public class OVREyeGaze : MonoBehaviour
{
    /// <summary>
    /// True if eye tracking is enabled, otherwise false.
    /// </summary>
    public bool EyeTrackingEnabled => OVRPlugin.eyeTrackingEnabled;

    /// <summary>
    /// GameObject will automatically change position and rotate according to the selected eye.
    /// </summary>
    public EyeId Eye;

    /// <summary>
    /// A confidence value ranging from 0..1 indicating the reliability of the eye tracking data.
    /// </summary>
    public float Confidence { get; private set; }

    /// <summary>
    /// GameObject will not change if detected eye state confidence is below this threshold.
    /// </summary>
    [Range(0f, 1f)]
    public float ConfidenceThreshold = 0.5f;

    /// <summary>
    /// GameObject will automatically change position.
    /// </summary>
    public bool ApplyPosition = true;

    /// <summary>
    /// GameObject will automatically rotate.
    /// </summary>
    public bool ApplyRotation = true;

    private OVRPlugin.EyeGazesState _currentEyeGazesState;

    /// <summary>
    /// Reference frame for eye. If it's null, then world reference frame will be used.
    /// </summary>
    public Transform ReferenceFrame;

    /// <summary>
    /// HeadSpace: Track eye relative to head space.
    /// WorldSpace: Track eye in world space.
    /// TrackingSpace: Track eye relative to OVRCameraRig.
    /// </summary>
    public EyeTrackingMode TrackingMode;

    private Quaternion _initialRotationOffset;
    private Transform _viewTransform;
    private const OVRPermissionsRequester.Permission EyeTrackingPermission = OVRPermissionsRequester.Permission.EyeTracking;
    private Action<string> _onPermissionGranted;
    private static int _trackingInstanceCount;

    private void Awake()
    {
        _onPermissionGranted = OnPermissionGranted;
    }

    private void Start()
    {
        PrepareHeadDirection();
    }

    private void OnEnable()
    {
        _trackingInstanceCount++;

        if (!StartEyeTracking())
        {
            enabled = false;
        }
    }

    private void OnPermissionGranted(string permissionId)
    {
        if (permissionId == OVRPermissionsRequester.GetPermissionId(EyeTrackingPermission))
        {
            OVRPermissionsRequester.PermissionGranted -= _onPermissionGranted;
            enabled = true;
        }
    }

    private bool StartEyeTracking()
    {
        if (!OVRPermissionsRequester.IsPermissionGranted(EyeTrackingPermission))
        {
            OVRPermissionsRequester.PermissionGranted -= _onPermissionGranted;
            OVRPermissionsRequester.PermissionGranted += _onPermissionGranted;
            return false;
        }

        if (!OVRPlugin.StartEyeTracking())
        {
            Debug.LogWarning($"[{nameof(OVREyeGaze)}] Failed to start eye tracking.");
            return false;
        }

        return true;
    }

    private void OnDisable()
    {
        if (--_trackingInstanceCount == 0)
        {
            OVRPlugin.StopEyeTracking();
        }
    }

    private void OnDestroy()
    {
        OVRPermissionsRequester.PermissionGranted -= _onPermissionGranted;
    }

    private void Update()
    {
        if (!OVRPlugin.GetEyeGazesState(OVRPlugin.Step.Render, -1, ref _currentEyeGazesState))
            return;

        var eyeGaze = _currentEyeGazesState.EyeGazes[(int)Eye];

        if (!eyeGaze.IsValid)
            return;

        Confidence = eyeGaze.Confidence;
        if (Confidence < ConfidenceThreshold)
            return;

        var pose = eyeGaze.Pose.ToOVRPose();
        switch (TrackingMode)
        {
            case EyeTrackingMode.HeadSpace:
                pose = pose.ToHeadSpacePose();
                break;
            case EyeTrackingMode.WorldSpace:
                pose = pose.ToWorldSpacePose(Camera.main);
                break;
        }

        if (ApplyPosition)
        {
            transform.position = pose.position;
        }

        if (ApplyRotation)
        {
            transform.rotation = CalculateEyeRotation(pose.orientation);
        }
    }

    private Quaternion CalculateEyeRotation(Quaternion eyeRotation)
    {
        var eyeRotationWorldSpace = _viewTransform.rotation * eyeRotation;
        var lookDirection = eyeRotationWorldSpace * Vector3.forward;
        var targetRotation = Quaternion.LookRotation(lookDirection, _viewTransform.up);

        return targetRotation * _initialRotationOffset;
    }

    private void PrepareHeadDirection()
    {
        string transformName = "HeadLookAtDirection";

        _viewTransform = new GameObject(transformName).transform;

        if (ReferenceFrame)
        {
            _viewTransform.SetPositionAndRotation(ReferenceFrame.position, ReferenceFrame.rotation);
        }
        else
        {
            _viewTransform.SetPositionAndRotation(transform.position, Quaternion.identity);
        }

        _viewTransform.parent = transform.parent;
        _initialRotationOffset = Quaternion.Inverse(_viewTransform.rotation) * transform.rotation;
    }


    /// <summary>
    /// List of eyes
    /// </summary>
    public enum EyeId
    {
        Left = OVRPlugin.Eye.Left,
        Right = OVRPlugin.Eye.Right
    }

    public enum EyeTrackingMode
    {
        HeadSpace,
        WorldSpace,
        TrackingSpace
    }
}
