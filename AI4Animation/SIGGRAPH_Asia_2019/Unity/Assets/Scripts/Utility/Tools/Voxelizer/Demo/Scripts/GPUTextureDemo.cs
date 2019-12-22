using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace VoxelSystem.Demo
{

    public class GPUTextureDemo : MonoBehaviour {

        enum MeshType {
            Volume, Surface
        };

        [SerializeField] MeshType type = MeshType.Volume;
		[SerializeField] protected Mesh mesh;
		[SerializeField] protected Texture2D texture;
		[SerializeField] protected ComputeShader voxelizer;
		[SerializeField] protected int count = 32;

        [SerializeField] protected RenderTexture voxelTex;

		void Start () {
			var data = GPUVoxelizer.Voxelize(voxelizer, mesh, count, (type == MeshType.Volume));
            voxelTex = GPUVoxelizer.BuildTexture3D(voxelizer, data, texture, RenderTextureFormat.ARGBFloat, FilterMode.Bilinear);
			data.Dispose();

            var mat = GetComponent<MeshRenderer>().sharedMaterial;
            mat.SetTexture("_MainTex", voxelTex);
		}

        void OnDestroy()
        {
            if(voxelTex != null)
            {
                voxelTex.Release();
                voxelTex = null;
            }
        }

    }

}


