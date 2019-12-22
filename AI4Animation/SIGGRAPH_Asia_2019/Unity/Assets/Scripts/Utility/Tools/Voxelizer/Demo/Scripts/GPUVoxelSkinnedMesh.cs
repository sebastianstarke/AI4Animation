using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using UnityEngine;
using UnityEngine.Rendering;

namespace VoxelSystem.Demo
{

    public class GPUVoxelSkinnedMesh : MonoBehaviour {

        enum MeshType {
            Volume, Surface
        };

        [SerializeField] MeshType type = MeshType.Volume;

        [SerializeField] new protected SkinnedMeshRenderer renderer;
        [SerializeField] protected ComputeShader voxelizer, particleUpdate;
        [SerializeField] protected int count = 64;

        protected Kernel setupKernel, updateKernel;
        protected ComputeBuffer particleBuffer;

        protected Renderer _renderer;
        protected MaterialPropertyBlock block;

        #region Shader property keys

        protected const string kSetupKernelKey = "Setup", kUpdateKernelKey = "Update";

        protected const string kVoxelBufferKey = "_VoxelBuffer", kVoxelCountKey = "_VoxelCount";
        protected const string kParticleBufferKey = "_ParticleBuffer", kParticleCountKey = "_ParticleCount";
        protected const string kUnitLengthKey = "_UnitLength";

        #endregion

        protected GPUVoxelData data;

        void Start () {
            var mesh = Sample();

            data = GPUVoxelizer.Voxelize(voxelizer, mesh, count, (type == MeshType.Volume));
            var pointMesh = BuildPoints(data);
            particleBuffer = new ComputeBuffer(pointMesh.vertexCount, Marshal.SizeOf(typeof(VParticle_t)));

            GetComponent<MeshFilter>().sharedMesh = pointMesh;

            block = new MaterialPropertyBlock();
            _renderer = GetComponent<Renderer>();
            _renderer.GetPropertyBlock(block);

            block.SetBuffer(kParticleBufferKey, particleBuffer);
            _renderer.SetPropertyBlock(block);

            setupKernel = new Kernel(particleUpdate, kSetupKernelKey);
            updateKernel = new Kernel(particleUpdate, kUpdateKernelKey);

            Compute(setupKernel, data, Time.deltaTime);
        }
        
        void Update () {
            if (data == null) return;

            data.Dispose();

            var mesh = Sample();
            data = GPUVoxelizer.Voxelize(voxelizer, mesh, count, (type == MeshType.Volume));

            Compute(updateKernel, data, Time.deltaTime);
        }

        Mesh Sample()
        {
            var mesh = new Mesh();
            renderer.BakeMesh(mesh);
            return mesh;
        }

        void OnDestroy ()
        {
            if(particleBuffer != null)
            {
                particleBuffer.Release();
                particleBuffer = null;
            }

            if(data != null)
            {
                data.Dispose();
                data = null;
            }
        }

        void Compute (Kernel kernel, GPUVoxelData data, float dt)
        {
            particleUpdate.SetBuffer(kernel.Index, kVoxelBufferKey, data.Buffer);
            particleUpdate.SetInt(kVoxelCountKey, data.Buffer.count);
            particleUpdate.SetFloat(kUnitLengthKey, data.UnitLength);

            particleUpdate.SetBuffer(kernel.Index, kParticleBufferKey, particleBuffer);
            particleUpdate.SetInt(kParticleCountKey, particleBuffer.count);

            particleUpdate.Dispatch(kernel.Index, particleBuffer.count / (int)kernel.ThreadX + 1, (int)kernel.ThreadY, (int)kernel.ThreadZ);
        }

        Mesh BuildPoints(GPUVoxelData data)
        {
            var count = data.Width * data.Height * data.Depth;
            var mesh = new Mesh();
			mesh.indexFormat = (count > 65535) ? IndexFormat.UInt32 : IndexFormat.UInt16;
            mesh.vertices = new Vector3[count];
            var indices = new int[count];
            for (int i = 0; i < count; i++) indices[i] = i;
            mesh.SetIndices(indices, MeshTopology.Points, 0);
            mesh.RecalculateBounds();
            return mesh;
        }

    }

}


