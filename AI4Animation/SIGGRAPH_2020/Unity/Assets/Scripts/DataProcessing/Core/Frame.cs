#if UNITY_EDITOR
using UnityEngine;

[System.Serializable]
public class Frame {
	public MotionData Data;
	public int Index;
	public float Timestamp;
	public Matrix4x4[] Transformations;

	[SerializeField] private Matrix4x4[] World;

	public Frame(MotionData data, int index, float timestamp, Matrix4x4[] matrices) {
		Data = data;
		Index = index;
		Timestamp = timestamp;
		World = (Matrix4x4[])matrices.Clone();
		Transformations = (Matrix4x4[])matrices.Clone();
	}

	public bool Repair() {
		if(Transformations == null || Transformations.Length != World.Length) {
			ResetTransformations();
			return true;
		}
		return false;
	}

	public Frame GetFirstFrame() {
		return Data.Frames[0];
	}

	public Frame GetLastFrame() {
		return Data.Frames[Data.Frames.Length-1];
	}

	public Matrix4x4[] GetSourceTransformations(bool mirrored) {
		Matrix4x4[] transformations = new Matrix4x4[Transformations.Length];
		for(int i=0; i<Transformations.Length; i++) {
			transformations[i] = GetSourceTransformation(i, mirrored);
		}
		return transformations;
	}

	public Matrix4x4[] GetSourceTransformations(string[] bones, bool mirrored) {
		Matrix4x4[] transformations = new Matrix4x4[bones.Length];
		for(int i=0; i<transformations.Length; i++) {
			transformations[i] = GetSourceTransformation(bones[i], mirrored);
		}
		return transformations;
	}

	public Matrix4x4[] GetSourceTransformations(int[] bones, bool mirrored) {
		Matrix4x4[] transformations = new Matrix4x4[bones.Length];
		for(int i=0; i<transformations.Length; i++) {
			transformations[i] = GetSourceTransformation(bones[i], mirrored);
		}
		return transformations;
	}

	public Matrix4x4 GetSourceTransformation(string bone, bool mirrored) {
		return GetSourceTransformation(Data.Source.FindBone(bone).Index, mirrored);
	}

	public Matrix4x4 GetSourceTransformation(int index, bool mirrored) {
		Matrix4x4 m = mirrored ? World[Data.Symmetry[index]].GetMirror(Data.MirrorAxis) : World[index];
		Vector3 o = mirrored ? Data.Offset.GetMirror(Data.MirrorAxis) : Data.Offset;
		m[0,3] += o.x;
		m[1,3] += o.y;
		m[2,3] += o.z;
		return m;
	}

	public void ResetTransformations() {
		Transformations = (Matrix4x4[])World.Clone();
	}

	public Matrix4x4[] GetBoneTransformations(bool mirrored) {
		Matrix4x4[] transformations = new Matrix4x4[Transformations.Length];
		for(int i=0; i<Transformations.Length; i++) {
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

	public Matrix4x4[] GetBoneTransformations(int[] bones, bool mirrored) {
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
		// Matrix4x4 m = mirrored ? Transformations[Data.Symmetry[index]].GetMirror(Data.MirrorAxis) : Transformations[index];
		// Vector3 o = mirrored ? Data.Offset.GetMirror(Data.MirrorAxis) : Data.Offset;
		// m[0,3] = Data.Scale * m[0,3] + o.x;
		// m[1,3] = Data.Scale * m[1,3] + o.y;
		// m[2,3] = Data.Scale * m[2,3] + o.z;
		// return m;
		Matrix4x4 scale = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, Data.Scale * Vector3.one);
		Matrix4x4 transformation = mirrored ? Transformations[Data.Symmetry[index]].GetMirror(Data.MirrorAxis) : Transformations[index];
		Matrix4x4 alignment =  mirrored ? Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(Data.Source.Bones[Data.Symmetry[index]].Alignment), Vector3.one).GetMirror(Data.MirrorAxis) : Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(Data.Source.Bones[index].Alignment), Vector3.one);
		return scale * transformation * alignment;
	}

	public Vector3[] GetBoneVelocities(bool mirrored) {
		Vector3[] velocities = new Vector3[Data.Source.Bones.Length];
		for(int i=0; i<Transformations.Length; i++) {
			velocities[i] = GetBoneVelocity(i, mirrored);
		}
		return velocities;
	}

	public Vector3[] GetBoneVelocities(string[] bones, bool mirrored) {
		Vector3[] velocities = new Vector3[bones.Length];
		for(int i=0; i<velocities.Length; i++) {
			velocities[i] = GetBoneVelocity(bones[i], mirrored);
		}
		return velocities;
	}

	public Vector3[] GetBoneVelocities(int[] bones, bool mirrored) {
		Vector3[] velocities = new Vector3[bones.Length];
		for(int i=0; i<velocities.Length; i++) {
			velocities[i] = GetBoneVelocity(bones[i], mirrored);
		}
		return velocities;
	}

	public Vector3 GetBoneVelocity(string bone, bool mirrored) {
		return GetBoneVelocity(Data.Source.FindBone(bone).Index, mirrored);
	}

	public Vector3 GetBoneVelocity(int index, bool mirrored) {
		if(Timestamp - Data.GetDeltaTime() < 0f) {
			return (Data.GetFrame(Timestamp + Data.GetDeltaTime()).GetBoneTransformation(index, mirrored).GetPosition() - GetBoneTransformation(index, mirrored).GetPosition()) / Data.GetDeltaTime();
		} else {
			return (GetBoneTransformation(index, mirrored).GetPosition() - Data.GetFrame(Timestamp - Data.GetDeltaTime()).GetBoneTransformation(index, mirrored).GetPosition()) / Data.GetDeltaTime();
		}
	}
}
#endif