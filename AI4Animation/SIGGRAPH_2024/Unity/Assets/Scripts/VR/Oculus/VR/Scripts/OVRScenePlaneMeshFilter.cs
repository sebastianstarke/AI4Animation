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
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

/// <summary>
/// Generates a mesh that represents a plane's boundary.
/// </summary>
/// <remarks>
/// When added to a GameObject that represents a scene entity, such as a floor, ceiling, or desk, this component
/// generates a mesh from its boundary vertices.
/// </remarks>
[RequireComponent(typeof(MeshFilter))]
public class OVRScenePlaneMeshFilter : MonoBehaviour
{
    private MeshFilter _meshFilter;

    private Mesh _mesh;

    private JobHandle? _jobHandle;

    private bool _meshRequested;

    private NativeArray<Vector2> _boundary;

    private NativeArray<int> _triangles;


    private void Start()
    {
        _mesh = new Mesh();
        _meshFilter = GetComponent<MeshFilter>();
        _meshFilter.sharedMesh = _mesh;

        var sceneAnchor = GetComponent<OVRSceneAnchor>();
        _mesh.name = sceneAnchor
            ? $"{nameof(OVRScenePlaneMeshFilter)} {sceneAnchor.Uuid}"
            : $"{nameof(OVRScenePlaneMeshFilter)} (anonymous)";

        RequestMeshGeneration();
    }

    internal void ScheduleMeshGeneration()
    {
        if (_jobHandle != null) return;
        if (!TryGetComponent<OVRScenePlane>(out var plane) || plane.Boundary.Count < 3) return;

        using var profiler = new OVRProfilerScope(nameof(ScheduleMeshGeneration));

        var vertexCount = plane.Boundary.Count;
        Debug.Assert(_boundary.IsCreated == false,
            "Boundary buffer should not be allocated.");

        using (new OVRProfilerScope("Copy boundary"))
        {
            _boundary = new NativeArray<Vector2>(vertexCount, Allocator.TempJob,
                NativeArrayOptions.UninitializedMemory);

            for (var i = 0; i < plane.Boundary.Count; i++)
            {
                _boundary[i] = plane.Boundary[i];
            }
        }

        using (new OVRProfilerScope("Schedule " + nameof(TriangulateBoundaryJob)))
        {
            _triangles = new NativeArray<int>((vertexCount - 2) * 3, Allocator.TempJob);
            _jobHandle = new TriangulateBoundaryJob
            {
                Boundary = _boundary,
                Triangles = _triangles,
            }.Schedule();
        }
    }

    private void Update()
    {
        if (_jobHandle?.IsCompleted == true)
        {
            // Even though the job is complete, we have to call Complete() in order
            // to mark the shared arrays as safe to read from.
            _jobHandle.Value.Complete();
            _jobHandle = null;
        }
        else
        {
            // Otherwise, there's a job running
            return;
        }

        if (_boundary.IsCreated && _triangles.IsCreated)
        {
            try
            {
                if (_triangles[0] == 0 &&
                    _triangles[1] == 0 &&
                    _triangles[2] == 0)
                {
                    return;
                }

                using (new OVRProfilerScope("Update mesh"))
                {
                    var vertices = new NativeArray<Vector3>(_boundary.Length, Allocator.Temp,
                        NativeArrayOptions.UninitializedMemory);
                    var normals = new NativeArray<Vector3>(_boundary.Length, Allocator.Temp,
                        NativeArrayOptions.UninitializedMemory);
                    var uvs = new NativeArray<Vector2>(_boundary.Length, Allocator.Temp,
                        NativeArrayOptions.UninitializedMemory);

                    using (new OVRProfilerScope("Prepare mesh data"))
                    {
                        for (var i = 0; i < _boundary.Length; i++)
                        {
                            var point = _boundary[i];
                            vertices[i] = new Vector3(point.x, point.y, 0);
                            normals[i] = new Vector3(0, 0, 1);
                            uvs[i] = new Vector2(point.x, point.y);
                        }
                    }

                    using (vertices)
                    using (normals)
                    using (uvs)
                    using (new OVRProfilerScope("Set mesh data"))
                    {
                        _mesh.Clear();
                        _mesh.SetVertices(vertices);
                        _mesh.SetIndices(_triangles, MeshTopology.Triangles, 0, calculateBounds: true);
                        _mesh.SetNormals(normals);
                        _mesh.SetUVs(0, uvs);
                    }
                }
            }
            finally
            {
                _boundary.Dispose();
                _triangles.Dispose();
            }
        }
        else if (_meshRequested)
        {
            ScheduleMeshGeneration();
        }
    }

    internal void RequestMeshGeneration()
    {
        _meshRequested = true;
        if (enabled)
        {
            ScheduleMeshGeneration();
        }
    }

    void OnDisable()
    {
        // Job completed but we may not yet have consumed the data
        if (_triangles.IsCreated)
        {
            _triangles.Dispose(_jobHandle ?? default);
        }

        _triangles = default;
        _jobHandle = null;
    }

    private struct TriangulateBoundaryJob : IJob
    {
        [ReadOnly]
        public NativeArray<Vector2> Boundary;

        [WriteOnly]
        public NativeArray<int> Triangles;

        struct NList : IDisposable
        {
            public int Count { get; private set; }

            NativeArray<int> _data;

            public NList(int capacity, Allocator allocator)
            {
                Count = capacity;
                _data = new NativeArray<int>(capacity, allocator);
                for (var i = 0; i < capacity; i++)
                {
                    _data[i] = i;
                }
            }

            public void RemoveAt(int index)
            {
                --Count;
                for (var i = index; i < Count; i++)
                {
                    _data[i] = _data[i + 1];
                }
            }

            public int GetAt(int index)
            {
                if (index >= Count)
                    return _data[index % Count];

                if (index < 0)
                    return _data[index % Count + Count];

                return _data[index];
            }

            public int this[int index] => _data[index];

            public void Dispose() => _data.Dispose();
        }

        public void Execute()
        {
            if (Boundary.Length == 0 || float.IsNaN(Boundary[0].x)) return;

            var indexList = new NList(Boundary.Length, Allocator.Temp);
            using var disposer = indexList;

            var indexListChanged = true;

            // Find a valid triangle.
            // Checks:
            // 1. Connected edges do not form a co-linear or reflex angle.
            // 2. There's no vertices inside the selected triangle area.
            var triangleCount = 0;
            while (indexList.Count > 3)
            {
                if (!indexListChanged)
                {
                    Debug.LogError($"[{nameof(OVRScenePlaneMeshFilter)}] Plane boundary triangulation failed.");

                    Triangles[0] = 0;
                    Triangles[1] = 0;
                    Triangles[2] = 0;
                    return;
                }

                indexListChanged = false;

                for (var i = 0; i < indexList.Count; i++)
                {
                    var a = indexList[i];
                    var b = indexList.GetAt(i - 1);
                    var c = indexList.GetAt(i + 1);

                    var va = Boundary[a];
                    var vb = Boundary[b];
                    var vc = Boundary[c];

                    var atob = vb - va;
                    var atoc = vc - va;

                    // reflex angle check
                    if (Cross(atob, atoc) < 0) continue;

                    var validTriangle = true;
                    for (var j = 0; j < Boundary.Length; j++)
                    {
                        if (j == a || j == b || j == c) continue;

                        if (PointInTriangle(Boundary[j], va, vb, vc))
                        {
                            validTriangle = false;
                            break;
                        }
                    }

                    // add indices to triangle list
                    if (!validTriangle) continue;

                    Triangles[triangleCount++] = c;
                    Triangles[triangleCount++] = a;
                    Triangles[triangleCount++] = b;

                    indexList.RemoveAt(i);
                    indexListChanged = true;
                    break;
                }
            }

            Triangles[triangleCount++] = indexList[2];
            Triangles[triangleCount++] = indexList[1];
            Triangles[triangleCount] = indexList[0];
        }

        private static float Cross(Vector2 a, Vector2 b) => a.x * b.y - a.y * b.x;

        private static bool PointInTriangle(Vector2 p, Vector2 a, Vector2 b, Vector2 c) =>
            Cross(b - a, p - a) >= 0 &&
            Cross(c - b, p - b) >= 0 &&
            Cross(a - c, p - c) >= 0;
    }
}
