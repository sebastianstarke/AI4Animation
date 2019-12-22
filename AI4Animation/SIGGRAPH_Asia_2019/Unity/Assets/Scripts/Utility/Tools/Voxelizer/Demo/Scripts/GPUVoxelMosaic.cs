using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using UnityEngine;
using UnityEngine.Rendering;
using Random = UnityEngine.Random;

namespace VoxelSystem.Demo
{

    // Inspired by Kohei Nakama works !
    // - https://www.instagram.com/p/Bbbg0WFBF6C/?hl=ja&taken-by=kouhei_nakama

    public class GPUVoxelMosaic : MonoBehaviour {

        [SerializeField] protected Mesh mesh;
        [SerializeField] protected ComputeShader voxelizer, particleUpdate;
        [SerializeField] protected int count = 64;

        #region Particle properties

        [SerializeField] protected float speedScaleMin = 2.0f, speedScaleMax = 5.0f;

        [SerializeField, Range(10, 100)] protected int frequency = 30;
        [SerializeField, Range(0f, 1f)] protected float level = 0f;
        protected int resolutions;

        #endregion

        [SerializeField] protected Material lineMat;
        Bounds bounds;

        protected GPUVoxelData[] levels;

        protected Kernel setupKernel, updateKernel;
        protected ComputeBuffer particleBuffer;

        protected new Renderer renderer;
        protected MaterialPropertyBlock block;

        #region Shader property keys

        protected const string kSetupKernelKey = "Setup", kUpdateKernelKey = "Update";

        protected const string kVoxelBufferKey = "_VoxelBuffer";
        protected const string kParticleBufferKey = "_ParticleBuffer", kParticleCountKey = "_ParticleCount";
        protected const string kWidthKey = "_Width", kHeightKey = "_Height", kDepthKey = "_Depth";
        protected const string kUnitLengthKey = "_UnitLength";

        protected const string kDTKey = "_DT";
        protected const string kSpeedKey = "_Speed";
        protected const string kLevelKey = "_Level";
        protected const string kThresholdKey = "_Threshold";

        #endregion

        #region MonoBehaviour functions

        void Start () {
            count = GPUVoxelizer.GetNearPow2(count);
            resolutions = Mathf.FloorToInt(Mathf.Log(count, 2)) - 2;

            levels = new GPUVoxelData[resolutions];
            for(int i = 0; i < resolutions; i++)
            {
                levels[i] = GPUVoxelizer.Voxelize(voxelizer, mesh, count >> i, true, true);
            }

            var pointMesh = BuildPoints(levels[0]);
            particleBuffer = new ComputeBuffer(pointMesh.vertexCount, Marshal.SizeOf(typeof(VParticle_t)));

            GetComponent<MeshFilter>().sharedMesh = pointMesh;

            block = new MaterialPropertyBlock();
            renderer = GetComponent<Renderer>();
            renderer.GetPropertyBlock(block);
            bounds = mesh.bounds;

            setupKernel = new Kernel(particleUpdate, kSetupKernelKey);
            updateKernel = new Kernel(particleUpdate, kUpdateKernelKey);

            Setup(levels[0]);
        }
      
        void Update () {
            var ilevel = Mathf.FloorToInt(level * resolutions);
            Compute(updateKernel, Mathf.Clamp(ilevel, 0, resolutions - 1), Time.deltaTime);

            block.SetBuffer(kParticleBufferKey, particleBuffer);
            renderer.SetPropertyBlock(block);

            // Randomize resolution
            if(Time.frameCount % frequency == 0) level = Random.value;
        }

        void OnDestroy ()
        {
            if(levels != null)
            {
                for(int i = 0, n = levels.Length; i < n; i++)
                {
                    levels[i].Dispose();
                }
                levels = null;
            }

            if(particleBuffer != null)
            {
                particleBuffer.Release();
                particleBuffer = null;
            }
        }

        #endregion

        void Setup(GPUVoxelData data)
        {
            particleUpdate.SetBuffer(setupKernel.Index, kVoxelBufferKey, data.Buffer);
            particleUpdate.SetBuffer(setupKernel.Index, kParticleBufferKey, particleBuffer);
            particleUpdate.SetInt(kParticleCountKey, particleBuffer.count);
            particleUpdate.SetInt(kWidthKey, data.Width);
            particleUpdate.SetInt(kHeightKey, data.Height);
            particleUpdate.SetInt(kDepthKey, data.Depth);
            particleUpdate.SetVector(kSpeedKey, new Vector2(speedScaleMin, speedScaleMax));
            particleUpdate.SetFloat(kUnitLengthKey, data.UnitLength);

            particleUpdate.Dispatch(setupKernel.Index, particleBuffer.count / (int)setupKernel.ThreadX + 1, (int)setupKernel.ThreadY, (int)setupKernel.ThreadZ);
        }

        void Compute (Kernel kernel, int level, float dt)
        {
            particleUpdate.SetBuffer(kernel.Index, kVoxelBufferKey, levels[level].Buffer);
            particleUpdate.SetFloat(kUnitLengthKey, levels[level].UnitLength);
            particleUpdate.SetInt(kLevelKey, level);

            particleUpdate.SetBuffer(kernel.Index, kParticleBufferKey, particleBuffer);
            particleUpdate.SetInt(kParticleCountKey, particleBuffer.count);

            particleUpdate.SetVector(kDTKey, new Vector2(dt, 1f / dt));

            particleUpdate.Dispatch(kernel.Index, particleBuffer.count / (int)kernel.ThreadX + 1, (int)kernel.ThreadY, (int)kernel.ThreadZ);
        }

        Mesh BuildPoints(GPUVoxelData data)
        {
            var count = data.Width * data.Height * data.Depth;
            var mesh = new Mesh();
			mesh.indexFormat = IndexFormat.UInt32;
            mesh.vertices = new Vector3[count];
            var indices = new int[count];
            for (int i = 0; i < count; i++) indices[i] = i;
            mesh.SetIndices(indices, MeshTopology.Points, 0);
            mesh.RecalculateBounds();
            return mesh;
        }

    }

}


