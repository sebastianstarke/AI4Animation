using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace VoxelSystem.Demo {

    [RequireComponent (typeof(MeshFilter))]
	public class GPUDemo : MonoBehaviour {

        enum MeshType {
            Volume, Surface
        };

        [SerializeField] MeshType type = MeshType.Volume;
		[SerializeField] protected Mesh mesh;
		[SerializeField] protected ComputeShader voxelizer;
		[SerializeField] protected int resolution = 32;
        [SerializeField] protected bool useUV = false;

		void Start () {
			var data = GPUVoxelizer.Voxelize(voxelizer, mesh, resolution, (type == MeshType.Volume));
			GetComponent<MeshFilter>().sharedMesh = VoxelMesh.Build(data.GetData(), data.UnitLength, useUV);
			data.Dispose();
		}

	}

}
