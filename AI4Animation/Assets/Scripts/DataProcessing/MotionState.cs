#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MotionState {
	public int Index;
	public float Timestamp;
	public bool Mirrored;
	public Matrix4x4 Root;
	public Vector3 RootMotion;
	public Matrix4x4[] BoneTransformations;
	public Vector3[] BoneVelocities;
	public Trajectory Trajectory;
	public HeightMap HeightMap;
	public DepthMap DepthMap;

	public List<Matrix4x4[]> PastBoneTransformations;
	public List<Vector3[]> PastBoneVelocities;
	public List<Matrix4x4[]> FutureBoneTransformations;
	public List<Vector3[]> FutureBoneVelocities;

	public MotionState(MotionData.Frame frame, bool mirrored) {
		Index = frame.Index;
		Timestamp = frame.Timestamp;
		Mirrored = mirrored;
		Root = frame.GetRootTransformation(mirrored);
		RootMotion = frame.GetRootMotion(mirrored);
		BoneTransformations = frame.GetBoneTransformations(mirrored);
		BoneVelocities = frame.GetBoneVelocities(mirrored);
		Trajectory = frame.GetTrajectory(mirrored);
		HeightMap = frame.GetHeightMap(mirrored);
		DepthMap = frame.GetDepthMap(mirrored);

		PastBoneTransformations = new List<Matrix4x4[]>(6);
		PastBoneVelocities = new List<Vector3[]>(6);
		for(int i=0; i<6; i++) {
			MotionData.Frame previous = frame.Data.GetFrame(Mathf.Clamp(frame.Timestamp - 1f + (float)i/6f, 0f, frame.Data.GetTotalTime()));
			PastBoneTransformations.Add(previous.GetBoneTransformations(mirrored));
			PastBoneVelocities.Add(previous.GetBoneVelocities(mirrored));
		}

		FutureBoneTransformations = new List<Matrix4x4[]>(5);
		FutureBoneVelocities = new List<Vector3[]>(5);
		for(int i=1; i<=5; i++) {
			MotionData.Frame future = frame.Data.GetFrame(Mathf.Clamp(frame.Timestamp + (float)i/5f, 0f, frame.Data.GetTotalTime()));
			FutureBoneTransformations.Add(future.GetBoneTransformations(mirrored));
			FutureBoneVelocities.Add(future.GetBoneVelocities(mirrored));
		}
	}
}
#endif