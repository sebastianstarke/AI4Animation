using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine;

public class OVRMeshJobs
{
	public struct TransformToUnitySpaceJob : IJobParallelFor
	{
		public NativeArray<Vector3> Vertices;
		public NativeArray<Vector3> Normals;
		public NativeArray<Vector2> UV;
		public NativeArray<BoneWeight> BoneWeights;
		
		public NativeArray<OVRPlugin.Vector3f> MeshVerticesPosition;
		public NativeArray<OVRPlugin.Vector3f> MeshNormals;
		public NativeArray<OVRPlugin.Vector2f> MeshUV;
		
		public NativeArray<OVRPlugin.Vector4f> MeshBoneWeights;
		public NativeArray<OVRPlugin.Vector4s> MeshBoneIndices;

		public void Execute(int index)
		{
			Vertices[index] = MeshVerticesPosition[index].FromFlippedXVector3f();
			Normals[index] = MeshNormals[index].FromFlippedXVector3f();
			
			UV[index] = new Vector2
			{
				x = MeshUV[index].x,
				y = -MeshUV[index].y
			};
			
			var currentBlendWeight = MeshBoneWeights[index];
			var currentBlendIndices = MeshBoneIndices[index];

			BoneWeights[index] = new BoneWeight
			{
				boneIndex0 = currentBlendIndices.x,
				weight0 = currentBlendWeight.x,
				
				boneIndex1 = currentBlendIndices.y,
				weight1 = currentBlendWeight.y,
				
				boneIndex2 = currentBlendIndices.z,
				weight2 = currentBlendWeight.z,
				
				boneIndex3 = currentBlendIndices.w,
				weight3 = currentBlendWeight.w,
			};
		}
	}
	
	public struct TransformTrianglesJob : IJobParallelFor
	{
		public NativeArray<uint> Triangles;
		
		[ReadOnly]
		public NativeArray<short> MeshIndices;
		public int NumIndices;

		public void Execute(int index)
		{
			Triangles[index] = (uint) MeshIndices[NumIndices - index - 1];
		}
	}

	public unsafe struct NativeArrayHelper<T> : IDisposable where T : struct
	{
		public NativeArray<T> UnityNativeArray;
		private GCHandle _handle;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
		private readonly AtomicSafetyHandle _atomicSafetyHandle;
#endif

		public NativeArrayHelper(T[] ovrArray, int length)
		{
			_handle = GCHandle.Alloc(ovrArray, GCHandleType.Pinned);
			var ptr = _handle.AddrOfPinnedObject();
			UnityNativeArray = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<T>(
				(void*)ptr, length, Allocator.None);

#if ENABLE_UNITY_COLLECTIONS_CHECKS
			_atomicSafetyHandle = AtomicSafetyHandle.Create();
			NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref UnityNativeArray, _atomicSafetyHandle);
#endif
		}

		public void Dispose()
		{
#if ENABLE_UNITY_COLLECTIONS_CHECKS
			AtomicSafetyHandle.Release(_atomicSafetyHandle);
#endif
			_handle.Free();
		}
	}
}
