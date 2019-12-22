#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[System.Serializable]
public class Frame {
	public MotionData Data;
	public int Index;
	public float Timestamp;
	public Matrix4x4[] World;

	public Frame(MotionData data, int index, float timestamp) {
		Data = data;
		Index = index;
		Timestamp = timestamp;
		World = new Matrix4x4[Data.Source.Bones.Length];
	}

	public Frame GetFirstFrame() {
		return Data.Frames[0];
	}

	public Frame GetLastFrame() {
		return Data.Frames[Data.Frames.Length-1];
	}

	public float[] GetBoneMasses(bool mirrored) {
		float[] masses = new float[World.Length];
		for(int i=0; i<masses.Length; i++) {
			masses[i] = GetBoneMass(i, mirrored);
		}
		return masses;
	}

	public float[] GetBoneMasses(string[] bones, bool mirrored) {
		float[] masses = new float[bones.Length];
		for(int i=0; i<masses.Length; i++) {
			masses[i] = GetBoneMass(bones[i], mirrored);
		}
		return masses;
	}

	public float GetBoneMass(string bone, bool mirrored) {
		return GetBoneMass(Data.Source.FindBone(bone).Index, mirrored);
	}

	public float GetBoneMass(int index, bool mirrored) {
		return mirrored ? Data.Source.Bones[Data.Symmetry[index]].Mass : Data.Source.Bones[index].Mass;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	//WORLD
	////////////////////////////////////////////////////////////////////////////////////////////////////
	public Matrix4x4[] GetBoneTransformations(bool mirrored) {
		Matrix4x4[] transformations = new Matrix4x4[World.Length];
		for(int i=0; i<World.Length; i++) {
			transformations[i] = GetBoneTransformation(i, mirrored);
		}
		return transformations;
	}

	public Matrix4x4[] GetBoneTransformations(string[] bones, bool mirrored) {
		Matrix4x4[] transformations = new Matrix4x4[bones.Length];
		for(int i=0; i<transformations.Length; i++) {
			transformations[i] = GetBoneTransformation(bones[i], mirrored);
		}
		return transformations;
	}

	public Matrix4x4 GetBoneTransformation(string bone, bool mirrored) {
		return GetBoneTransformation(Data.Source.FindBone(bone).Index, mirrored);
	}

	public Matrix4x4 GetBoneTransformation(int index, bool mirrored) {
		Matrix4x4 m = mirrored ? World[Data.Symmetry[index]].GetMirror(Data.MirrorAxis) : World[index];
		Vector3 o = mirrored ? Data.Offset.GetMirror(Data.MirrorAxis) : Data.Offset;
		m[0,3] = Data.Scale * (m[0,3] + o.x);
		m[1,3] = Data.Scale * (m[1,3] + o.y);
		m[2,3] = Data.Scale * (m[2,3] + o.z);
		return m;
		/*
		return 
			Matrix4x4.TRS(Data.Offset, Quaternion.identity, Data.Scaling * Vector3.one) * 
			(mirrored ? World[Data.Symmetry[index]].GetMirror(Data.MirrorAxis.GetAxis()) : World[index]) * 
			Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(Data.Source.Bones[index].Alignment), Vector3.one);
			*/
	}

	public Vector3[] GetBoneVelocities(bool mirrored, float delta) {
		Vector3[] velocities = new Vector3[Data.Source.Bones.Length];
		for(int i=0; i<World.Length; i++) {
			velocities[i] = GetBoneVelocity(i, mirrored, delta);
		}
		return velocities;
	}

	public Vector3[] GetBoneVelocities(string[] bones, bool mirrored, float delta) {
		Vector3[] velocities = new Vector3[bones.Length];
		for(int i=0; i<velocities.Length; i++) {
			velocities[i] = GetBoneVelocity(bones[i], mirrored, delta);
		}
		return velocities;
	}

	public Vector3 GetBoneVelocity(string bone, bool mirrored, float delta) {
		return GetBoneVelocity(Data.Source.FindBone(bone).Index, mirrored, delta);
	}

	public Vector3 GetBoneVelocity(int index, bool mirrored, float delta) {
		if(delta == 0f) {
			return Vector3.zero;
		}
		if(Timestamp - delta < 0f) {
			return (Data.GetFrame(Timestamp + delta).GetBoneTransformation(index, mirrored).GetPosition() - GetBoneTransformation(index, mirrored).GetPosition()) / delta;
		} else {
			return (GetBoneTransformation(index, mirrored).GetPosition() - Data.GetFrame(Timestamp - delta).GetBoneTransformation(index, mirrored).GetPosition()) / delta;
		}
	}

	public Vector3[] GetBoneAccelerations(bool mirrored, float delta) {
		Vector3[] accelerations = new Vector3[Data.Source.Bones.Length];
		for(int i=0; i<World.Length; i++) {
			accelerations[i] = GetBoneAcceleration(i, mirrored, delta);
		}
		return accelerations;
	}

	public Vector3[] GetBoneAccelerations(string[] bones, bool mirrored, float delta) {
		Vector3[] accelerations = new Vector3[bones.Length];
		for(int i=0; i<accelerations.Length; i++) {
			accelerations[i] = GetBoneAcceleration(bones[i], mirrored, delta);
		}
		return accelerations;
	}

	public Vector3 GetBoneAcceleration(string bone, bool mirrored, float delta) {
		return GetBoneAcceleration(Data.Source.FindBone(bone).Index, mirrored, delta);
	}

	public Vector3 GetBoneAcceleration(int index, bool mirrored, float delta) {
		if(delta == 0f) {
			return Vector3.zero;
		}
		if(Timestamp - delta < 0f) {
			return (Data.GetFrame(Timestamp + delta).GetBoneVelocity(index, mirrored, delta) - GetBoneVelocity(index, mirrored, delta)) / delta;
		} else {
			return (GetBoneVelocity(index, mirrored, delta) - Data.GetFrame(Timestamp - delta).GetBoneVelocity(index, mirrored, delta)) / delta;
		}
	}

	public float[] GetAngularBoneVelocities(bool mirrored, float delta) {
		float[] values = new float[Data.Source.Bones.Length];
		for(int i=0; i<World.Length; i++) {
			values[i] = GetAngularBoneVelocity(i, mirrored, delta);
		}
		return values;
	}

	public float[] GetAngularBoneVelocities(string[] bones, bool mirrored, float delta) {
		float[] values = new float[bones.Length];
		for(int i=0; i<values.Length; i++) {
			values[i] = GetAngularBoneVelocity(bones[i], mirrored, delta);
		}
		return values;
	}

	public float GetAngularBoneVelocity(string bone, bool mirrored, float delta) {
		return GetAngularBoneVelocity(Data.Source.FindBone(bone).Index, mirrored, delta);
	}

	public float GetAngularBoneVelocity(int index, bool mirrored, float delta) {
		if(delta == 0f) {
			return 0f;
		}
		if(Timestamp - delta < 0f) {
			return Quaternion.Angle(GetBoneTransformation(index, mirrored).GetRotation(), Data.GetFrame(Timestamp + delta).GetBoneTransformation(index, mirrored).GetRotation()) / delta;
		} else {
			return Quaternion.Angle(Data.GetFrame(Timestamp - delta).GetBoneTransformation(index, mirrored).GetRotation(), GetBoneTransformation(index, mirrored).GetRotation()) / delta;
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	//LOCAL
	////////////////////////////////////////////////////////////////////////////////////////////////////
	public Matrix4x4[] GetLocalBoneTransformations(bool mirrored) {
		Matrix4x4[] transformations = new Matrix4x4[World.Length];
		for(int i=0; i<World.Length; i++) {
			transformations[i] = GetLocalBoneTransformation(i, mirrored);
		}
		return transformations;
	}

	public Matrix4x4[] GetLocalBoneTransformations(string[] bones, bool mirrored) {
		Matrix4x4[] transformations = new Matrix4x4[bones.Length];
		for(int i=0; i<transformations.Length; i++) {
			transformations[i] = GetLocalBoneTransformation(bones[i], mirrored);
		}
		return transformations;
	}

	public Matrix4x4 GetLocalBoneTransformation(string bone, bool mirrored) {
		return GetLocalBoneTransformation(Data.Source.FindBone(bone).Index, mirrored);
	}

	public Matrix4x4 GetLocalBoneTransformation(int index, bool mirrored) {
		if(index == 0) {
			return GetBoneTransformation(index, mirrored);
		} else {
			return GetBoneTransformation(index, mirrored).GetRelativeTransformationTo(GetBoneTransformation(Data.Source.Bones[index].Parent, mirrored));
		}
	}

	public Vector3[] GetLocalBoneVelocities(bool mirrored, float delta) {
		Vector3[] velocities = new Vector3[Data.Source.Bones.Length];
		for(int i=0; i<World.Length; i++) {
			velocities[i] = GetLocalBoneVelocity(i, mirrored, delta);
		}
		return velocities;
	}

	public Vector3[] GetLocalBoneVelocities(string[] bones, bool mirrored, float delta) {
		Vector3[] velocities = new Vector3[bones.Length];
		for(int i=0; i<velocities.Length; i++) {
			velocities[i] = GetLocalBoneVelocity(bones[i], mirrored, delta);
		}
		return velocities;
	}

	public Vector3 GetLocalBoneVelocity(string bone, bool mirrored, float delta) {
		return GetLocalBoneVelocity(Data.Source.FindBone(bone).Index, mirrored, delta);
	}

	public Vector3 GetLocalBoneVelocity(int index, bool mirrored, float delta) {
		if(delta == 0f) {
			return Vector3.zero;
		}
		if(Timestamp - delta < 0f) {
			return (Data.GetFrame(Timestamp + delta).GetLocalBoneTransformation(index, mirrored).GetPosition() - GetLocalBoneTransformation(index, mirrored).GetPosition()) / delta;
		} else {
			return (GetLocalBoneTransformation(index, mirrored).GetPosition() - Data.GetFrame(Timestamp - delta).GetLocalBoneTransformation(index, mirrored).GetPosition()) / delta;
		}
	}

	public Vector3[] GetLocalBoneAccelerations(bool mirrored, float delta) {
		Vector3[] accelerations = new Vector3[Data.Source.Bones.Length];
		for(int i=0; i<World.Length; i++) {
			accelerations[i] = GetLocalBoneAcceleration(i, mirrored, delta);
		}
		return accelerations;
	}

	public Vector3[] GetLocalBoneAccelerations(string[] bones, bool mirrored, float delta) {
		Vector3[] accelerations = new Vector3[bones.Length];
		for(int i=0; i<accelerations.Length; i++) {
			accelerations[i] = GetLocalBoneAcceleration(bones[i], mirrored, delta);
		}
		return accelerations;
	}

	public Vector3 GetLocalBoneAcceleration(string bone, bool mirrored, float delta) {
		return GetLocalBoneAcceleration(Data.Source.FindBone(bone).Index, mirrored, delta);
	}

	public Vector3 GetLocalBoneAcceleration(int index, bool mirrored, float delta) {
		if(delta == 0f) {
			return Vector3.zero;
		}
		if(Timestamp - delta < 0f) {
			return (Data.GetFrame(Timestamp + delta).GetLocalBoneVelocity(index, mirrored, delta) - GetLocalBoneVelocity(index, mirrored, delta)) / delta;
		} else {
			return (GetLocalBoneVelocity(index, mirrored, delta) - Data.GetFrame(Timestamp - delta).GetLocalBoneVelocity(index, mirrored, delta)) / delta;
		}
	}

}
#endif