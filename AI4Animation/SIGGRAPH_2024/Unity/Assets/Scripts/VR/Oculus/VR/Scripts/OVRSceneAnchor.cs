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

/// <summary>
/// Represents a scene anchor.
/// </summary>
/// <remarks>
/// A scene anchor is a type of anchor that is provided by the system. It represents an item in the physical
/// environment, such as a plane or volume. Scene anchors are created by the <see cref="OVRSceneManager"/>.
/// </remarks>
/// <seealso cref="OVRScenePlane"/>
/// <seealso cref="OVRSceneVolume"/>
/// <seealso cref="OVRSemanticClassification"/>
[DisallowMultipleComponent]
public sealed class OVRSceneAnchor : MonoBehaviour
{
    /// <summary>
    /// The runtime handle of this scene anchor.
    /// </summary>
    public OVRSpace Space { get; private set; }

    /// <summary>
    /// The universally unique identifier for this scene anchor.
    /// </summary>
    public Guid Uuid { get; private set; }

    /// <summary>
    /// Indicates whether this anchor is tracked by the system.
    /// </summary>
    public bool IsTracked { get; internal set; }

    private static readonly Quaternion RotateY180 = Quaternion.Euler(0, 180, 0);
    private OVRPlugin.Posef? _pose = null;

    private bool IsComponentEnabled(OVRPlugin.SpaceComponentType spaceComponentType) =>
        OVRPlugin.GetSpaceComponentStatus(Space, spaceComponentType, out var componentEnabled, out _)
        && componentEnabled;

    private void SyncComponent<T>(OVRPlugin.SpaceComponentType spaceComponentType)
        where T : MonoBehaviour, IOVRSceneComponent
    {
        if (!IsComponentEnabled(spaceComponentType)) return;

        var component = GetComponent<T>();
        if (component)
        {
            // If the component already exists, then it means it was added before this component was valid, so we need
            // to initialize it.
            component.Initialize();
        }
        else
        {
            gameObject.AddComponent<T>();
        }
    }

    internal void ClearPoseCache()
    {
        _pose = null;
    }

    internal void Initialize(OVRSpace space, Guid uuid)
    {
        if (Space.Valid)
            throw new InvalidOperationException($"[{uuid}] {nameof(OVRSceneAnchor)} has already been initialized.");

        if (!space.Valid)
            throw new ArgumentException($"[{uuid}] {nameof(space)} must be valid.", nameof(space));

        Space = space;
        Uuid = uuid;

        ClearPoseCache();

        SceneAnchors[this.Uuid] = this;
        SceneAnchorsList.Add(this);

        AnchorReferenceCountDictionary.TryGetValue(Space, out var referenceCount);
        AnchorReferenceCountDictionary[Space] = referenceCount + 1;

        if (!IsComponentEnabled(OVRPlugin.SpaceComponentType.Locatable))
        {
            OVRSceneManager.Development.LogError(nameof(OVRSceneAnchor),
                $"[{uuid}] Is missing the {nameof(OVRPlugin.SpaceComponentType.Locatable)} component.");
        }

        // Generally, we want to set the transform as soon as possible, but there is a valid use case where we want to
        // disable this component as soon as its added to override the transform.
        if (enabled)
        {
            var updateTransformSucceeded = TryUpdateTransform(false);

            // This should work; so add some development-only logs so we know if something is wrong here.
            if (updateTransformSucceeded)
            {
                IsTracked = true;
                OVRSceneManager.Development.Log(nameof(OVRSceneAnchor), $"[{uuid}] Initial transform set.");
            }
            else
            {
                OVRSceneManager.Development.LogWarning(nameof(OVRSceneAnchor),
                    $"[{uuid}] {nameof(OVRPlugin.TryLocateSpace)} failed. The entity may have the wrong initial transform.");
            }

        }

        SyncComponent<OVRSemanticClassification>(OVRPlugin.SpaceComponentType.SemanticLabels);
        SyncComponent<OVRSceneVolume>(OVRPlugin.SpaceComponentType.Bounded3D);
        SyncComponent<OVRScenePlane>(OVRPlugin.SpaceComponentType.Bounded2D);
    }

    /// <summary>
    /// Initializes this scene anchor from an existing scene anchor.
    /// </summary>
    /// <param name="other">An existing <see cref="OVRSceneAnchor"/> from which to initialize this scene anchor.</param>
    /// <exception cref="ArgumentNullException">Thrown if <paramref name="other"/> is `null`.</exception>
    /// <exception cref="InvalidOperationException">Thrown if this <see cref="OVRSceneAnchor"/> is already associated with a scene anchor.</exception>
    public void InitializeFrom(OVRSceneAnchor other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Initialize(other.Space, other.Uuid);
    }

    /// <summary>
    /// Get the list of all scene anchors.
    /// </summary>
    /// <param name="anchors">A list of <see cref="OVRSceneAnchor"/> to populate.</param>
    /// <exception cref="ArgumentNullException">Thrown if <paramref name="anchors"/> is `null`.</exception>
    public static void GetSceneAnchors(List<OVRSceneAnchor> anchors)
    {
        if (anchors == null)
            throw new ArgumentNullException(nameof(anchors));

        anchors.Clear();
        anchors.AddRange(SceneAnchorsList);
    }

    internal bool TryUpdateTransform(bool useCache)
    {
        if (!Space.Valid || !enabled) return false;

        if (!useCache || _pose == null)
        {
            if (!OVRPlugin.TryLocateSpace(Space, OVRPlugin.GetTrackingOriginType(), out var pose))
            {
                return false;
            }
            _pose = pose;
        }

        // NOTE: This transformation performs the following steps:
        // 1. Flip Z to convert from OpenXR's right-handed to Unity's left-handed coordinate system.
        //    OpenXR             Unity
        //       | y          y |  / z
        //       |              | /
        //       +----> x       +----> x
        //      /
        //    z/ (normal)
        //
        // 2. (1) means that Z now points in the opposite direction from OpenXR. However, the design is such that a
        //    plane's normal should coincide with +Z, so we rotate 180 degrees around the +Y axis to make Z now point
        //    in the intended direction.
        //    OpenXR           Unity
        //       | y           y |
        //       |               |
        //       +---->  x  <----+
        //      /               /
        //    z/             z / (normal)
        //
        // 3. Convert from tracking space to world space.
        var worldSpacePose = new OVRPose
        {
            position = _pose.Value.Position.FromFlippedZVector3f(),
            orientation = _pose.Value.Orientation.FromFlippedZQuatf() * RotateY180
        }.ToWorldSpacePose(Camera.main);
        transform.SetPositionAndRotation(worldSpacePose.position, worldSpacePose.orientation);
        return true;
    }

    private void OnDestroy()
    {
        SceneAnchors.Remove(this.Uuid);
        SceneAnchorsList.Remove(this);

        // Anchor destroyed. Adding it to an ignore list.
        DestroyedSceneAnchors.Add(this.Uuid);

        if (!Space.Valid)
        {
            return;
        }

        if (!AnchorReferenceCountDictionary.TryGetValue(Space, out var referenceCount))
        {
            OVRSceneManager.Development.LogError(nameof(OVRSceneAnchor),
                $"[Anchor {Space.Handle}] has not been found, can't find it for deletion");
            return;
        }

        if (referenceCount == 1)
        {
            // last reference to this anchor, delete it
            if (Space.Valid)
            {
                OVRPlugin.DestroySpace(Space);
            }

            // remove instead of decrement to not waste memory
            AnchorReferenceCountDictionary.Remove(Space);
        }
        else
        {
            AnchorReferenceCountDictionary[Space] = referenceCount - 1;
        }
    }

    private static readonly Dictionary<OVRSpace, int> AnchorReferenceCountDictionary =
        new Dictionary<OVRSpace, int>();

    internal static readonly Dictionary<Guid, OVRSceneAnchor> SceneAnchors = new Dictionary<Guid, OVRSceneAnchor>();
    internal static readonly List<OVRSceneAnchor> SceneAnchorsList = new List<OVRSceneAnchor>();
    internal static readonly HashSet<Guid> DestroyedSceneAnchors = new HashSet<Guid>();
}

internal interface IOVRSceneComponent
{
    void Initialize();
}

