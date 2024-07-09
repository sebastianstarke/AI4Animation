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
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

/// <summary>
/// A <see cref="OVRSceneAnchor"/> that has a 2D bounds associated with it.
/// </summary>
[DisallowMultipleComponent]
[RequireComponent(typeof(OVRSceneAnchor))]
public class OVRScenePlane : MonoBehaviour, IOVRSceneComponent
{
    /// <summary>
    /// The plane's width (in the local X-direction), in meters.
    /// </summary>
    public float Width { get; private set; }

    /// <summary>
    /// The plane's height (in the local Y-direction), in meters.
    /// </summary>
    public float Height { get; private set; }

    /// <summary>
    /// The dimensions of the plane.
    /// </summary>
    /// <remarks>
    /// This property corresponds to a Vector whose components are
    /// (<see cref="Width"/>, <see cref="Height"/>).
    /// </remarks>
    public Vector2 Dimensions => new Vector2(Width, Height);

    /// <summary>
    /// The vertices of the 2D plane boundary.
    /// </summary>
    /// <remarks>
    /// The vertices are provided in clockwise order and in plane-space (relative to the
    /// plane's local space). The X and Y coordinates of the 2D coordinates are the same as the 3D
    /// coordinates. To map the 2D vertices (x, y) to 3D, set the Z coordinate to zero: (x, y, 0).
    /// </remarks>
    public IReadOnlyList<Vector2> Boundary => _boundary;

    /// <summary>
    /// Whether the child transforms will be scaled according to the dimensions of this plane.
    /// </summary>
    /// <remarks>If set to True, all the child transforms will be scaled to the dimensions of this plane immediately.
    /// And, if it's set to False, dimensions of this plane will no longer affect the child transforms, and child
    /// transforms will retain their current scale.</remarks>
    public bool ScaleChildren
    {
        get => _scaleChildren;
        set {
            _scaleChildren = value;
            if(_scaleChildren && _sceneAnchor.Space.Valid)
            {
                SetChildScale(transform, Width, Height);
            }
        }
    }

    [Tooltip("When enabled, scales the child transforms according to the dimensions of this plane")]
    [SerializeField]
    private bool _scaleChildren = true;

    internal JobHandle? _jobHandle;

    private NativeArray<Vector2> _previousBoundary;

    private NativeArray<int> _boundaryLength;

    private NativeArray<Vector2> _boundaryBuffer;

    private bool _boundaryRequested;

    private OVRSceneAnchor _sceneAnchor;

    private readonly List<Vector2> _boundary = new List<Vector2>();

    private static void SetChildScale(Transform parentTransform, float width, float height)
    {
        for (var i = 0; i < parentTransform.childCount; i++)
        {
            var child = parentTransform.GetChild(i);
            var scale = new Vector3(width, height, child.localScale.z);
            child.localScale = scale;
        }
    }

    private void Awake()
    {
        _sceneAnchor = GetComponent<OVRSceneAnchor>();
        if (_sceneAnchor.Space.Valid)
        {
            ((IOVRSceneComponent)this).Initialize();
        }
    }

    private void Start()
    {
        RequestBoundary();
    }

    void IOVRSceneComponent.Initialize()
    {
        if (OVRPlugin.GetSpaceBoundingBox2D(GetComponent<OVRSceneAnchor>().Space, out var rect))
        {
            Width = rect.Size.w;
            Height = rect.Size.h;

            // The volume component will also set the scale
            if (!GetComponent<OVRSceneVolume>() && ScaleChildren)
            {
                SetChildScale(transform, Width, Height);
            }
        }
        else
        {
            OVRSceneManager.Development.LogError(nameof(OVRScenePlane),
                $"[{GetComponent<OVRSceneAnchor>().Uuid}] Could not obtain 2D bounds.");
        }
    }

    internal void ScheduleGetLengthJob()
    {
        // Don't schedule if already running
        if (_jobHandle != null) return;

        if (!OVRPlugin.GetSpaceComponentStatus(_sceneAnchor.Space,
            OVRPlugin.SpaceComponentType.Bounded2D, out var isEnabled, out var isChangePending))
        {
            return;
        }

        if (!isEnabled || isChangePending) return;

        // Scratch buffer to store single value on the heap
        _boundaryLength = new NativeArray<int>(1, Allocator.TempJob);

        // Two-call idiom: first call gets the length
        _jobHandle = new GetBoundaryLengthJob
        {
            Length = _boundaryLength,
            Space = _sceneAnchor.Space
        }.Schedule();
        _boundaryRequested = false;
    }

    internal void RequestBoundary()
    {
        _boundaryRequested = true;
        if (enabled)
        {
            // If enabled, we can go ahead and start right away
            ScheduleGetLengthJob();
        }
    }

    void Update()
    {
        if (_jobHandle?.IsCompleted == true)
        {
            _jobHandle.Value.Complete();
            _jobHandle = null;
        }
        else
        {
            return;
        }

        if (_boundaryLength.IsCreated)
        {
            var length = _boundaryLength[0];
            _boundaryLength.Dispose();

            if (length < 3)
            {
                // This means that we failed to get the boundary length, so try again
                ScheduleGetLengthJob();
                return;
            }

            using (new OVRProfilerScope("Schedule " + nameof(GetBoundaryJob)))
            {
                _boundaryBuffer = new NativeArray<Vector2>(length, Allocator.TempJob);
                if (!_previousBoundary.IsCreated)
                {
                    _previousBoundary = new NativeArray<Vector2>(length, Allocator.Persistent);
                }
                _jobHandle = new GetBoundaryJob
                {
                    Space = _sceneAnchor.Space,
                    Boundary = _boundaryBuffer,
                    PreviousBoundary = _previousBoundary,
                }.Schedule();
            }
        }
        else if (_boundaryBuffer.IsCreated)
        {
            using (new OVRProfilerScope("Copy boundary"))
            {
                if (_previousBoundary.Length == 0 || float.IsNaN(_previousBoundary[0].x))
                {
                    if (_previousBoundary.IsCreated)
                    {
                        _previousBoundary.Dispose();
                    }

                    _previousBoundary = new NativeArray<Vector2>(_boundaryBuffer.Length,
                        Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                    _previousBoundary.CopyFrom(_boundaryBuffer);

                    // Finally, copy it to the publicly accessible list.
                    _boundary.Clear();
                    foreach (var vertex in _previousBoundary)
                    {
                        _boundary.Add(new Vector2(-vertex.x, vertex.y));
                    }
                }
            }

            _boundaryBuffer.Dispose();

            if (TryGetComponent<OVRScenePlaneMeshFilter>(out var planeMeshFilter))
            {
                // Notify mesh filter that there's a new boundary
                planeMeshFilter.RequestMeshGeneration();
            }
        }
        else if (_boundaryRequested)
        {
            ScheduleGetLengthJob();
        }
    }

    private void OnDisable()
    {
        // Job completed but we may not yet have consumed the data
        if (_boundaryLength.IsCreated) _boundaryLength.Dispose(_jobHandle ?? default);
        if (_boundaryBuffer.IsCreated) _boundaryBuffer.Dispose(_jobHandle ?? default);
        if (_previousBoundary.IsCreated) _previousBoundary.Dispose(_jobHandle ?? default);

        _previousBoundary = default;
        _boundaryBuffer = default;
        _boundaryLength = default;
        _jobHandle = null;
    }

    private struct GetBoundaryLengthJob : IJob
    {
        public OVRSpace Space;

        [WriteOnly]
        public NativeArray<int> Length;

        public void Execute() => Length[0] = OVRPlugin.GetSpaceBoundary2DCount(Space, out var count)
            ? count
            : 0;
    }

    private struct GetBoundaryJob : IJob
    {
        public OVRSpace Space;

        public NativeArray<Vector2> Boundary;

        public NativeArray<Vector2> PreviousBoundary;

        private bool HasBoundaryChanged()
        {
            if (!PreviousBoundary.IsCreated) return true;
            if (Boundary.Length != PreviousBoundary.Length) return true;

            var length = Boundary.Length;
            for (var i = 0; i < length; i++)
            {
                if (Vector2.SqrMagnitude(Boundary[i] - PreviousBoundary[i]) > 1e-6f) return true;
            }

            return false;
        }

        private static void SetNaN(NativeArray<Vector2> array)
        {
            // Set a NaN to indicate failure
            if (array.Length > 0)
            {
                array[0] = new Vector2(float.NaN, float.NaN);
            }
        }

        public void Execute()
        {
            if (OVRPlugin.GetSpaceBoundary2D(Space, Boundary) && HasBoundaryChanged())
            {
                // Invalid old boundary
                SetNaN(PreviousBoundary);
            }
            else
            {
                // Invalid boundary
                SetNaN(Boundary);
            }
        }
    }
}
