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
using UnityEngine;

/// <summary>
/// This class manages the face expressions data.
/// </summary>
/// <remarks>
/// Refers to the <see cref="OVRFaceExpressions.FaceExpression"/> enum for the list of face expressions.
/// </remarks>
public class OVRFaceExpressions : MonoBehaviour, IReadOnlyCollection<float>
{
    /// <summary>
    /// True if face tracking is enabled, otherwise false.
    /// </summary>
    public bool FaceTrackingEnabled => OVRPlugin.faceTrackingEnabled;

    /// <summary>
    /// True if the facial expressions are valid, otherwise false.
    /// </summary>
    /// <remarks>
    /// This value gets updated in every frame. You should check this
    /// value before querying for face expressions.
    /// </remarks>
    public bool ValidExpressions { get; private set; }

    /// <summary>
    /// True if the eye look-related blend shapes are valid, otherwise false.
    /// </summary>
    /// <remarks>
    /// This property affects the behavior of two sets of blend shapes.
    ///
    /// **EyesLook:**
    /// - <see cref="FaceExpression.EyesLookDownL"/>
    /// - <see cref="FaceExpression.EyesLookDownR"/>
    /// - <see cref="FaceExpression.EyesLookLeftL"/>
    /// - <see cref="FaceExpression.EyesLookLeftR"/>
    /// - <see cref="FaceExpression.EyesLookRightL"/>
    /// - <see cref="FaceExpression.EyesLookRightR"/>
    /// - <see cref="FaceExpression.EyesLookUpL"/>
    /// - <see cref="FaceExpression.EyesLookUpR"/>
    ///
    /// **EyesClosed:**
    /// - <see cref="FaceExpression.EyesClosedL"/>
    /// - <see cref="FaceExpression.EyesClosedR"/>
    ///
    /// **When <see cref="EyeFollowingBlendshapesValid"/> is `false`:**
    /// - The `EyesLook` blend shapes are set to zero.
    /// - The `EyesClosed` blend shapes range from 0..1, and represent the true state of the eyelids.
    ///
    /// **When <see cref="EyeFollowingBlendshapesValid"/> is `true`:**
    /// - The `EyesLook` blend shapes are valid.
    /// - The `EyesClosed` blend shapes are modified so that the sum of the `EyesClosedX` and `EyesLookDownX` blend shapes
    ///   range from 0..1. This helps avoid double deformation of the avatar's eye lids when they may be driven by both
    ///   the `EyesClosed` and `EyesLookDown` blend shapes. To recover the true `EyesClosed` values, add the
    ///   minimum of `EyesLookDownL` and `EyesLookDownR` blend shapes back using the following formula:<br />
    ///   `EyesClosedL` += min(`EyesLookDownL`, `EyesLookDownR`)<br />
    ///   `EyesClosedR` += min(`EyesLookDownL`, `EyesLookDownR`)
    /// </remarks>
    public bool EyeFollowingBlendshapesValid { get; private set; }

    private OVRPlugin.FaceState _currentFaceState;
    private const OVRPermissionsRequester.Permission FaceTrackingPermission = OVRPermissionsRequester.Permission.FaceTracking;
    private Action<string> _onPermissionGranted;
    private static int _trackingInstanceCount;

    private void Awake()
    {
        _onPermissionGranted = OnPermissionGranted;
    }

    private void OnEnable()
    {
        _trackingInstanceCount++;

        if (!StartFaceTracking())
        {
            enabled = false;
        }
    }

    private void OnPermissionGranted(string permissionId)
    {
        if (permissionId == OVRPermissionsRequester.GetPermissionId(FaceTrackingPermission))
        {
            OVRPermissionsRequester.PermissionGranted -= _onPermissionGranted;
            enabled = true;
        }
    }

    private bool StartFaceTracking()
    {
        if (!OVRPermissionsRequester.IsPermissionGranted(FaceTrackingPermission))
        {
            OVRPermissionsRequester.PermissionGranted -= _onPermissionGranted;
            OVRPermissionsRequester.PermissionGranted += _onPermissionGranted;
            return false;
        }

        if (!OVRPlugin.StartFaceTracking())
        {
            Debug.LogWarning($"[{nameof(OVRFaceExpressions)}] Failed to start face tracking.");
            return false;
        }

        return true;
    }

    private void OnDisable()
    {
        if (--_trackingInstanceCount == 0)
        {
            OVRPlugin.StopFaceTracking();
        }
    }

    private void OnDestroy()
    {
        OVRPermissionsRequester.PermissionGranted -= _onPermissionGranted;
    }

    private void Update()
    {
        ValidExpressions = OVRPlugin.GetFaceState(OVRPlugin.Step.Render, -1, ref _currentFaceState) &&
                           _currentFaceState.Status.IsValid;

        EyeFollowingBlendshapesValid = ValidExpressions && _currentFaceState.Status.IsEyeFollowingBlendshapesValid;
    }


    /// <summary>
    /// This will return the weight of the given expression.
    /// </summary>
    /// <returns>Returns weight of expression ranged between 0.0 to 100.0.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when <see cref="OVRFaceExpressions.ValidExpressions"/> is false.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="expression"/> value is not in range.
    /// </exception>
    public float this[FaceExpression expression]
    {
        get
        {
            CheckValidity();

            if (expression < 0 || expression >= FaceExpression.Max)
            {
                throw new ArgumentOutOfRangeException(nameof(expression),
                    expression,
                    $"Value must be between 0 to {(int)FaceExpression.Max}");
            }

            return _currentFaceState.ExpressionWeights[(int)expression];
        }
    }

    /// <summary>
    /// This method tries to gets the weight of the given expression if it's available.
    /// </summary>
    /// <param name="expression" cref="FaceExpression">The expression to get the weight of.</param>
    /// <param name="weight">The output argument that will contain the expression weight or 0.0 if it's not available.</param>
    /// <returns>Returns true if the expression weight is available, false otherwise</returns>
    public bool TryGetFaceExpressionWeight(FaceExpression expression, out float weight)
    {
        if (!ValidExpressions || expression < 0 || expression >= FaceExpression.Max)
        {
            weight = 0;
            return false;
        }

        weight = _currentFaceState.ExpressionWeights[(int)expression];
        return true;
    }

    /// <summary>
    /// List of face parts used for getting the face tracking confidence weight in <see cref="TryGetWeightConfidence"/>.
    /// </summary>
    public enum FaceRegionConfidence
    {
        /// <summary>
        /// Represents the lower part of the face. It includes the mouth, chin and a portion of the nose and cheek.
        /// </summary>
        Lower = OVRPlugin.FaceRegionConfidence.Lower,
        /// <summary>
        /// Represents the upper part of the face. It includes the eyes, eye brows and a portion of the nose and cheek.
        /// </summary>
        Upper = OVRPlugin.FaceRegionConfidence.Upper,
        /// <summary>
        /// Used to determine the size of the <see cref="FaceRegionConfidence"/> enum.
        /// </summary>
        Max = OVRPlugin.FaceRegionConfidence.Max
    }

    /// <summary>
    /// This method tries to gets the confidence weight of the given face part if it's available.
    /// </summary>
    /// <param name="region" cref="FaceRegionConfidence">The part of the face to get the confidence weight of.</param>
    /// <param name="weightConfidence">The output argument that will contain the weight confidence or 0.0 if it's not available.</param>
    /// <returns>Returns true if the weight confidence is available, false otherwise</returns>
    public bool TryGetWeightConfidence(FaceRegionConfidence region, out float weightConfidence)
    {
        if (!ValidExpressions || region < 0 || region >= FaceRegionConfidence.Max)
        {
            weightConfidence = 0;
            return false;
        }

        weightConfidence = _currentFaceState.ExpressionWeightConfidences[(int)region];
        return true;
    }

    private void CheckValidity()
    {
        if (!ValidExpressions)
        {
            throw new InvalidOperationException(
                $"Face expressions are not valid at this time. Use {nameof(ValidExpressions)} to check for validity.");
        }
    }

    /// <summary>
    /// Copies expression weights to a pre-allocated array.
    /// </summary>
    /// <param name="array">Pre-allocated destination array for expression weights</param>
    /// <param name="startIndex">Starting index in the destination array</param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="array"/> is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when there is not enough capacity to copy weights to <paramref name="array"/> at <paramref name="startIndex"/> index.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="startIndex"/> value is out of <paramref name="array"/> bounds.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when <see cref="OVRFaceExpressions.ValidExpressions"/> is false.
    /// </exception>
    public void CopyTo(float[] array, int startIndex = 0)
    {
        if (array == null)
        {
            throw new ArgumentNullException(nameof(array));
        }

        if (startIndex < 0 || startIndex >= array.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(startIndex),
                startIndex,
                $"Value must be between 0 to {array.Length - 1}");
        }

        if (array.Length - startIndex < (int)FaceExpression.Max)
        {
            throw new ArgumentException(
                $"Capacity is too small - required {(int)FaceExpression.Max}, available {array.Length - startIndex}.",
                nameof(array));
        }

        CheckValidity();
        for (int i = 0; i < (int)FaceExpression.Max; i++)
        {
            array[i + startIndex] = _currentFaceState.ExpressionWeights[i];
        }
    }

    /// <summary>
    /// Allocates a float array and copies expression weights to it.
    /// </summary>
    public float[] ToArray()
    {
        var array = new float[(int)OVRFaceExpressions.FaceExpression.Max];
        this.CopyTo(array);
        return array;
    }

    /// <summary>
    /// List of face expressions.
    /// </summary>
    public enum FaceExpression
    {
        BrowLowererL = OVRPlugin.FaceExpression.Brow_Lowerer_L,
        BrowLowererR = OVRPlugin.FaceExpression.Brow_Lowerer_R,
        CheekPuffL = OVRPlugin.FaceExpression.Cheek_Puff_L,
        CheekPuffR = OVRPlugin.FaceExpression.Cheek_Puff_R,
        CheekRaiserL = OVRPlugin.FaceExpression.Cheek_Raiser_L,
        CheekRaiserR = OVRPlugin.FaceExpression.Cheek_Raiser_R,
        CheekSuckL = OVRPlugin.FaceExpression.Cheek_Suck_L,
        CheekSuckR = OVRPlugin.FaceExpression.Cheek_Suck_R,
        ChinRaiserB = OVRPlugin.FaceExpression.Chin_Raiser_B,
        ChinRaiserT = OVRPlugin.FaceExpression.Chin_Raiser_T,
        DimplerL = OVRPlugin.FaceExpression.Dimpler_L,
        DimplerR = OVRPlugin.FaceExpression.Dimpler_R,
        EyesClosedL = OVRPlugin.FaceExpression.Eyes_Closed_L,
        EyesClosedR = OVRPlugin.FaceExpression.Eyes_Closed_R,
        EyesLookDownL = OVRPlugin.FaceExpression.Eyes_Look_Down_L,
        EyesLookDownR = OVRPlugin.FaceExpression.Eyes_Look_Down_R,
        EyesLookLeftL = OVRPlugin.FaceExpression.Eyes_Look_Left_L,
        EyesLookLeftR = OVRPlugin.FaceExpression.Eyes_Look_Left_R,
        EyesLookRightL = OVRPlugin.FaceExpression.Eyes_Look_Right_L,
        EyesLookRightR = OVRPlugin.FaceExpression.Eyes_Look_Right_R,
        EyesLookUpL = OVRPlugin.FaceExpression.Eyes_Look_Up_L,
        EyesLookUpR = OVRPlugin.FaceExpression.Eyes_Look_Up_R,
        InnerBrowRaiserL = OVRPlugin.FaceExpression.Inner_Brow_Raiser_L,
        InnerBrowRaiserR = OVRPlugin.FaceExpression.Inner_Brow_Raiser_R,
        JawDrop = OVRPlugin.FaceExpression.Jaw_Drop,
        JawSidewaysLeft = OVRPlugin.FaceExpression.Jaw_Sideways_Left,
        JawSidewaysRight = OVRPlugin.FaceExpression.Jaw_Sideways_Right,
        JawThrust = OVRPlugin.FaceExpression.Jaw_Thrust,
        LidTightenerL = OVRPlugin.FaceExpression.Lid_Tightener_L,
        LidTightenerR = OVRPlugin.FaceExpression.Lid_Tightener_R,
        LipCornerDepressorL = OVRPlugin.FaceExpression.Lip_Corner_Depressor_L,
        LipCornerDepressorR = OVRPlugin.FaceExpression.Lip_Corner_Depressor_R,
        LipCornerPullerL = OVRPlugin.FaceExpression.Lip_Corner_Puller_L,
        LipCornerPullerR = OVRPlugin.FaceExpression.Lip_Corner_Puller_R,
        LipFunnelerLB = OVRPlugin.FaceExpression.Lip_Funneler_LB,
        LipFunnelerLT = OVRPlugin.FaceExpression.Lip_Funneler_LT,
        LipFunnelerRB = OVRPlugin.FaceExpression.Lip_Funneler_RB,
        LipFunnelerRT = OVRPlugin.FaceExpression.Lip_Funneler_RT,
        LipPressorL = OVRPlugin.FaceExpression.Lip_Pressor_L,
        LipPressorR = OVRPlugin.FaceExpression.Lip_Pressor_R,
        LipPuckerL = OVRPlugin.FaceExpression.Lip_Pucker_L,
        LipPuckerR = OVRPlugin.FaceExpression.Lip_Pucker_R,
        LipStretcherL = OVRPlugin.FaceExpression.Lip_Stretcher_L,
        LipStretcherR = OVRPlugin.FaceExpression.Lip_Stretcher_R,
        LipSuckLB = OVRPlugin.FaceExpression.Lip_Suck_LB,
        LipSuckLT = OVRPlugin.FaceExpression.Lip_Suck_LT,
        LipSuckRB = OVRPlugin.FaceExpression.Lip_Suck_RB,
        LipSuckRT = OVRPlugin.FaceExpression.Lip_Suck_RT,
        LipTightenerL = OVRPlugin.FaceExpression.Lip_Tightener_L,
        LipTightenerR = OVRPlugin.FaceExpression.Lip_Tightener_R,
        LipsToward = OVRPlugin.FaceExpression.Lips_Toward,
        LowerLipDepressorL = OVRPlugin.FaceExpression.Lower_Lip_Depressor_L,
        LowerLipDepressorR = OVRPlugin.FaceExpression.Lower_Lip_Depressor_R,
        MouthLeft = OVRPlugin.FaceExpression.Mouth_Left,
        MouthRight = OVRPlugin.FaceExpression.Mouth_Right,
        NoseWrinklerL = OVRPlugin.FaceExpression.Nose_Wrinkler_L,
        NoseWrinklerR = OVRPlugin.FaceExpression.Nose_Wrinkler_R,
        OuterBrowRaiserL = OVRPlugin.FaceExpression.Outer_Brow_Raiser_L,
        OuterBrowRaiserR = OVRPlugin.FaceExpression.Outer_Brow_Raiser_R,
        UpperLidRaiserL = OVRPlugin.FaceExpression.Upper_Lid_Raiser_L,
        UpperLidRaiserR = OVRPlugin.FaceExpression.Upper_Lid_Raiser_R,
        UpperLipRaiserL = OVRPlugin.FaceExpression.Upper_Lip_Raiser_L,
        UpperLipRaiserR = OVRPlugin.FaceExpression.Upper_Lip_Raiser_R,
        [InspectorName("None")]
        Max = OVRPlugin.FaceExpression.Max
    }


    #region Face expressions enumerator

    public FaceExpressionsEnumerator GetEnumerator() =>
        new FaceExpressionsEnumerator(_currentFaceState.ExpressionWeights);

    IEnumerator<float> IEnumerable<float>.GetEnumerator() => GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    public int Count => _currentFaceState.ExpressionWeights?.Length ?? 0;

    public struct FaceExpressionsEnumerator : IEnumerator<float>
    {
        private float[] _faceExpressions;

        private int _index;

        private int _count;

        internal FaceExpressionsEnumerator(float[] array)
        {
            _faceExpressions = array;
            _index = -1;
            _count = _faceExpressions?.Length ?? 0;
        }

        public bool MoveNext() => ++_index < _count;

        public float Current => _faceExpressions[_index];

        object IEnumerator.Current => Current;

        public void Reset() => _index = -1;

        public void Dispose()
        {
        }
    }

    #endregion
}
