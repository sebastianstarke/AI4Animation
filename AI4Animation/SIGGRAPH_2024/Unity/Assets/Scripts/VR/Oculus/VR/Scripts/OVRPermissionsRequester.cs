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
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Android;

/// <summary>
/// This class handles android permission requests for the capabilities listed in <see cref = "Permission"/>.
/// </summary>
internal static class OVRPermissionsRequester
{
    /// <summary>
    /// Occurs when a <see cref="Permission"/> is granted.
    /// </summary>
    public static event Action<string> PermissionGranted;

    /// <summary>
    /// Enum listing the capabilities this class can request permission for.
    /// </summary>
    public enum Permission
    {
        /// <summary>
        /// Represents the Face Tracking capability.
        /// </summary>
        FaceTracking,

        /// <summary>
        /// Represents the Body Tracking capability.
        /// </summary>
        BodyTracking,

        /// <summary>
        /// Represents the Eye Tracking capability.
        /// </summary>
        EyeTracking
    }

    private const string FaceTrackingPermission = "com.oculus.permission.FACE_TRACKING";
    private const string EyeTrackingPermission  = "com.oculus.permission.EYE_TRACKING";
    private const string BodyTrackingPermission = "com.oculus.permission.BODY_TRACKING";

    /// <summary>
    /// Returns the permission ID of the given <see cref="Permission"/> to be requested from the user.
    /// </summary>
    /// <param name="permission">The <see cref="Permission"/> to get the ID of.</param>
    /// <returns></returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when an invalid <see cref="Permission"/> is used.</exception>
    public static string GetPermissionId(Permission permission)
    {
        return permission switch
        {
            Permission.FaceTracking => FaceTrackingPermission,
            Permission.BodyTracking => BodyTrackingPermission,
            Permission.EyeTracking => EyeTrackingPermission,
            _ => throw new ArgumentOutOfRangeException(nameof(permission), permission, null)
        };
    }

    private static bool IsPermissionSupportedByPlatform(Permission permission)
    {
        return permission switch
        {
            Permission.FaceTracking => OVRPlugin.faceTrackingSupported,
            Permission.BodyTracking => OVRPlugin.bodyTrackingSupported,
            Permission.EyeTracking => OVRPlugin.eyeTrackingSupported,
            _ => throw new ArgumentOutOfRangeException(nameof(permission), permission, null)
        };
    }

    /// <summary>
    /// Returns whether the <see cref="permission"/> has been granted.
    /// </summary>
    /// <param name="permission"><see cref="Permission"/> to be checked.</param>
    public static bool IsPermissionGranted(Permission permission)
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        return UnityEngine.Android.Permission.HasUserAuthorizedPermission(GetPermissionId(permission));
#else
        return true;
#endif
    }

    /// <summary>
    /// Requests the listed <see cref="permissions"/>.
    /// </summary>
    /// <param name="permissions">Set of <see cref="Permission"/> to be requested.</param>
    public static void Request(IEnumerable<Permission> permissions)
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        var permissionIdsToRequest = new List<string>();

        foreach (var permission in permissions)
        {
            if (ShouldRequestPermission(permission))
            {
                permissionIdsToRequest.Add(GetPermissionId(permission));
            }
        }

        if (permissionIdsToRequest.Count > 0)
        {
            UnityEngine.Android.Permission.RequestUserPermissions(permissionIdsToRequest.ToArray(),
                BuildPermissionCallbacks());
        }
#endif
    }

    private static bool ShouldRequestPermission(Permission permission)
    {
        if (!IsPermissionSupportedByPlatform(permission))
        {
            Debug.LogWarning(
                $"[[{nameof(OVRPermissionsRequester)}] Permission {permission} is not supported by the platform and can't be requested.");
            return false;
        }

        return !IsPermissionGranted(permission);
    }

    private static PermissionCallbacks BuildPermissionCallbacks()
    {
        var permissionCallbacks = new PermissionCallbacks();
        permissionCallbacks.PermissionDenied += permissionId =>
        {
            Debug.LogWarning($"[{nameof(OVRPermissionsRequester)}] Permission {permissionId} was denied.");
        };
        permissionCallbacks.PermissionDeniedAndDontAskAgain += permissionId =>
        {
            Debug.LogWarning(
                $"[{nameof(OVRPermissionsRequester)}] Permission {permissionId} was denied and blocked from being requested again.");
        };
        permissionCallbacks.PermissionGranted += permissionId =>
        {
            Debug.Log($"[{nameof(OVRPermissionsRequester)}] Permission {permissionId} was granted.");
            PermissionGranted?.Invoke(permissionId);
        };
        return permissionCallbacks;
    }
}
