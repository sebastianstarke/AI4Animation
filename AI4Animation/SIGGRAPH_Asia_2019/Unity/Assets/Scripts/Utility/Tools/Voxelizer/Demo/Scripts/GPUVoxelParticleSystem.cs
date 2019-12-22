using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using UnityEngine;
using UnityEngine.Rendering;
using Random = UnityEngine.Random;

namespace VoxelSystem.Demo
{

    public class GPUVoxelParticleSystem : MonoBehaviour {

        [SerializeField] protected Mesh mesh;
        [SerializeField] protected ComputeShader voxelizer, particleUpdate;
        [SerializeField] protected int count = 64;

        #region Particle properties

        [SerializeField] protected float speedScaleMin = 2.0f, speedScaleMax = 5.0f;
        [SerializeField] protected float speedLimit = 1.0f;
        [SerializeField, Range(0, 15)] protected float drag = 0.1f;
        [SerializeField] protected Vector3 gravity = Vector3.zero;
        [SerializeField] protected float speedToSpin = 60.0f;
        [SerializeField] protected float maxSpin = 20.0f;
        [SerializeField] protected float noiseAmplitude = 1.0f;
        [SerializeField] protected float noiseFrequency = 0.01f;
        [SerializeField] protected float noiseMotion = 1.0f;
        [SerializeField, Range(0f, 1f)] protected float threshold = 0f;
        protected Vector3 noiseOffset;

        #endregion

        [SerializeField] protected Material lineMat;
        Bounds bounds;

        protected GPUVoxelData data;

        protected Kernel setupKernel, updateKernel;
        protected ComputeBuffer particleBuffer, indexBuffer;

        protected new Renderer renderer;
        protected MaterialPropertyBlock block;

        #region Shader property keys

        protected const string kSetupKernelKey = "Setup", kUpdateKernelKey = "Update";

        protected const string kVoxelBufferKey = "_VoxelBuffer", kIndexBufferKey = "_IndexBuffer";
        protected const string kParticleBufferKey = "_ParticleBuffer", kParticleCountKey = "_ParticleCount";
        protected const string kWidthKey = "_Width", kHeightKey = "_Height", kDepthKey = "_Depth";
        protected const string kUnitLengthKey = "_UnitLength";

        protected const string kDTKey = "_DT";
        protected const string kSpeedKey = "_Speed";
        protected const string kDamperKey = "_Damper";
        protected const string kGravityKey = "_Gravity";
        protected const string kSpinKey = "_Spin";
        protected const string kNoiseParamsKey = "_NoiseParams", kNoiseOffsetKey = "_NoiseOffset";
        protected const string kThresholdKey = "_Threshold";

        #endregion

        #region MonoBehaviour functions

        void Start () {
			data = GPUVoxelizer.Voxelize(voxelizer, mesh, count);

            int[] indices;
            var pointMesh = BuildPoints(data, out indices);
            particleBuffer = new ComputeBuffer(pointMesh.vertexCount, Marshal.SizeOf(typeof(VParticle_t)));
            indexBuffer = new ComputeBuffer(pointMesh.vertexCount, Marshal.SizeOf(typeof(int)));
            indexBuffer.SetData(indices);

            GetComponent<MeshFilter>().sharedMesh = pointMesh;

            block = new MaterialPropertyBlock();
            renderer = GetComponent<Renderer>();
            renderer.GetPropertyBlock(block);
            bounds = mesh.bounds;

            setupKernel = new Kernel(particleUpdate, kSetupKernelKey);
            updateKernel = new Kernel(particleUpdate, kUpdateKernelKey);

            Setup();
        }
      
        void Update () {
            Compute(updateKernel, Time.deltaTime);

            block.SetBuffer(kParticleBufferKey, particleBuffer);
            renderer.SetPropertyBlock(block);

            var t = Time.timeSinceLevelLoad;
            threshold = (Mathf.Cos(t * 0.5f) + 1.0f) * 0.5f;
        }

        void OnDestroy ()
        {
            if(data != null)
            {
                data.Dispose();
                data = null;
            }

            if(particleBuffer != null)
            {
                particleBuffer.Release();
                particleBuffer = null;
            }

            if(indexBuffer != null)
            {
                indexBuffer.Release();
                indexBuffer = null;
            }
        }

        void OnRenderObject()
        {
            RenderPlane();
        }

        #endregion

        void Setup()
        {
            particleUpdate.SetBuffer(setupKernel.Index, kVoxelBufferKey, data.Buffer);
            particleUpdate.SetBuffer(setupKernel.Index, kIndexBufferKey, indexBuffer);
            particleUpdate.SetBuffer(setupKernel.Index, kParticleBufferKey, particleBuffer);
            particleUpdate.SetInt(kParticleCountKey, particleBuffer.count);
            particleUpdate.SetInt(kWidthKey, data.Width);
            particleUpdate.SetInt(kHeightKey, data.Height);
            particleUpdate.SetInt(kDepthKey, data.Depth);
            particleUpdate.SetVector(kSpeedKey, new Vector2(speedScaleMin, speedScaleMax));
            particleUpdate.SetFloat(kUnitLengthKey, data.UnitLength);

            particleUpdate.Dispatch(setupKernel.Index, particleBuffer.count / (int)setupKernel.ThreadX + 1, (int)setupKernel.ThreadY, (int)setupKernel.ThreadZ);
        }

        void Compute (Kernel kernel, float dt)
        {
            particleUpdate.SetBuffer(kernel.Index, kVoxelBufferKey, data.Buffer);
            particleUpdate.SetBuffer(kernel.Index, kIndexBufferKey, indexBuffer);
            particleUpdate.SetBuffer(kernel.Index, kParticleBufferKey, particleBuffer);
            particleUpdate.SetInt(kParticleCountKey, particleBuffer.count);

            particleUpdate.SetVector(kDTKey, new Vector2(dt, 1f / dt));
            particleUpdate.SetVector(kDamperKey, new Vector2(Mathf.Exp(-drag * dt), speedLimit));
            particleUpdate.SetVector(kGravityKey, gravity * dt);

            var pi360dt = Mathf.PI * dt / 360;
            particleUpdate.SetVector(kSpinKey, new Vector2(maxSpin * pi360dt, speedToSpin * pi360dt));

            particleUpdate.SetVector(kNoiseParamsKey, new Vector2(noiseFrequency, noiseAmplitude * dt));

            var noiseDir = (gravity == Vector3.zero) ? Vector3.up : gravity.normalized;
            noiseOffset += noiseDir * noiseMotion * dt;
            particleUpdate.SetVector(kNoiseOffsetKey, noiseOffset);

            threshold = Mathf.Clamp01(threshold);
            particleUpdate.SetInt(kThresholdKey, Mathf.FloorToInt(threshold * data.Height));

            particleUpdate.Dispatch(kernel.Index, particleBuffer.count / (int)kernel.ThreadX + 1, (int)kernel.ThreadY, (int)kernel.ThreadZ);
        }

        Mesh BuildPoints(GPUVoxelData data, out int[] vIndices)
        {
			var voxels = data.GetData();
			var vertices = new List<Vector3>();
            var indices = new List<int>();
            var vIndicesTmp = new List<int>();

            var count = 0;
			for(int i = 0, n = voxels.Length; i < n; i++) {
				var v = voxels[i];
                if (v.fill > 0)
                {
                    vertices.Add(v.position);
                    indices.Add(count++);
                    vIndicesTmp.Add(i);
                }
            }

            vIndices = vIndicesTmp.ToArray();

            var mesh = new Mesh();
			mesh.indexFormat = IndexFormat.UInt32;
            mesh.vertices = vertices.ToArray();
            mesh.SetIndices(indices.ToArray(), MeshTopology.Points, 0);
            mesh.RecalculateBounds();
            return mesh;
        }

        protected void RenderPlane()
        {
            GL.PushMatrix();
            GL.MultMatrix(transform.localToWorldMatrix);

            lineMat.SetPass(0);

            GL.Begin(GL.LINES);

            var min = bounds.min;
            var max = bounds.max;
            var h = Mathf.Lerp(min.y, max.y, threshold);

            Vector3
                v0 = new Vector3(min.x, h, min.z),
                v1 = new Vector3(min.x, h, max.z),
                v2 = new Vector3(max.x, h, max.z),
                v3 = new Vector3(max.x, h, min.z);
            
            GL.Vertex(v0); GL.Vertex(v1);
            GL.Vertex(v1); GL.Vertex(v2);
            GL.Vertex(v2); GL.Vertex(v3);
            GL.Vertex(v3); GL.Vertex(v0);

            GL.End();
            GL.PopMatrix();
        }

    }

}


