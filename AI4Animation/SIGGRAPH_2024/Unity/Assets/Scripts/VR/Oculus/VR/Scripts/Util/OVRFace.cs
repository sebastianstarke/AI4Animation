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

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

/// <summary>
/// OVR Component to drive blend shapes on a <c>SkinnedMeshRenderer</c> based on Face Tracking provided by <see cref="OVRFaceExpressions"/>.
/// </summary>
/// <remarks>
/// Intended to be used as a base type that is inherited from, in order to provide mapping logic from blend shape indices.
/// The mapping of <see cref="OVRFaceExpressions.FaceExpression"/> to blend shapes is accomplished by overriding <see cref="OVRFace.GetFaceExpression(int)"/>.
/// Needs to be linked to an <see cref="OVRFaceExpressions"/> component to fetch tracking data from.
/// </remarks>
[RequireComponent(typeof(SkinnedMeshRenderer))]
public class OVRFace : MonoBehaviour
{
    /// <summary>
    /// Start this instance.
    /// Will validate that all properties are set correctly
    /// </summary>
    public OVRFaceExpressions FaceExpressions
    {
        get => _faceExpressions;
        set => _faceExpressions = value;
    }

    public float BlendShapeStrengthMultiplier
    {
        get => _blendShapeStrengthMultiplier;
        set => _blendShapeStrengthMultiplier = value;
    }

    [SerializeField]
    [Tooltip("The OVRFaceExpressions Component to fetch the Face Tracking weights from that are to be applied")]
    protected internal OVRFaceExpressions _faceExpressions;

    [SerializeField]
    [Tooltip("A multiplier to the weights read from the OVRFaceExpressions to exaggerate facial expressions")]
    protected internal float _blendShapeStrengthMultiplier = 1.0f;

    private SkinnedMeshRenderer _skinnedMeshRenderer;

    /// <summary>
    /// Start this instance.
    /// Will validate that all properties are set correctly
    /// </summary>
    protected virtual void Start()
    {
        Assert.IsNotNull(_faceExpressions);
        _skinnedMeshRenderer = GetComponent<SkinnedMeshRenderer>();
        Assert.IsNotNull(_skinnedMeshRenderer);
        Assert.IsNotNull(_skinnedMeshRenderer.sharedMesh);
    }

    /// <summary>
    /// Update this instance.
    /// Will update the blend shape weights on the SkinnedMeshRenderer
    /// </summary>
    protected virtual void Update()
    {
        if (!_faceExpressions.FaceTrackingEnabled || !_faceExpressions.enabled)
        {
            return;
        }

        if ( _faceExpressions.ValidExpressions)
        {
            int numBlendshapes = _skinnedMeshRenderer.sharedMesh.blendShapeCount;

            for (int blendShapeIndex = 0; blendShapeIndex < numBlendshapes; ++blendShapeIndex)
            {
                OVRFaceExpressions.FaceExpression blendShapeToFaceExpression = GetFaceExpression(blendShapeIndex);
                if (blendShapeToFaceExpression >= OVRFaceExpressions.FaceExpression.Max || blendShapeToFaceExpression < 0)
                {
                    continue;
                }
                float currentWeight = _faceExpressions[blendShapeToFaceExpression];
                _skinnedMeshRenderer.SetBlendShapeWeight(blendShapeIndex, currentWeight * _blendShapeStrengthMultiplier);
            }
        }
    }

    /// <summary>
    /// Fetches the <see cref="OVRFaceExpressions.FaceExpression"/> for a given blend shape index on the shared mesh of the <c>SkinnedMeshRenderer</c> on the same component
    /// </summary>
    /// <remarks>
    /// Override this function to provide the mapping between blend shapes and face expressions
    /// </remarks>
    /// <param name="blendShapeIndex">The index of the blend shape, will be in-between 0 and the number of blend shapes on the shared mesh.</param>
    /// <returns>Returns the <see cref="OVRFaceExpressions.FaceExpression"/> to drive the bland shape identified by <paramref name="blendShapeIndex"/>.</returns>
    protected virtual OVRFaceExpressions.FaceExpression GetFaceExpression(int blendShapeIndex) => OVRFaceExpressions.FaceExpression.Max;
}
