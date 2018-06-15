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
	public Matrix4x4[] Local;
	public Matrix4x4[] World;

	public Frame(MotionData data, int index, float timestamp) {
		Data = data;
		Index = index;
		Timestamp = timestamp;
		Local = new Matrix4x4[Data.Source.Bones.Length];
		World = new Matrix4x4[Data.Source.Bones.Length];
	}

	public Frame GetPreviousFrame() {
		return Data.Frames[Mathf.Clamp(Index-2, 0, Data.Frames.Length-1)];
	}

	public Frame GetNextFrame() {
		return Data.Frames[Mathf.Clamp(Index, 0, Data.Frames.Length-1)];
	}

	public Frame GetFirstFrame() {
		return Data.Frames[0];
	}

	public Frame GetLastFrame() {
		return Data.Frames[Data.Frames.Length-1];
	}

	public Matrix4x4[] GetBoneTransformations(bool mirrored) {
		Matrix4x4[] transformations = new Matrix4x4[World.Length];
		for(int i=0; i<World.Length; i++) {
			transformations[i] = GetBoneTransformation(i, mirrored);
		}
		return transformations;
	}

	public Matrix4x4 GetBoneTransformation(int index, bool mirrored, int smoothing = 0) {
		if(smoothing  == 0) {
			return Matrix4x4.TRS(Vector3.zero, Quaternion.identity, Data.Scaling * Vector3.one) * (mirrored ? World[Data.Symmetry[index]].GetMirror(Data.GetAxis(Data.MirrorAxis)) : World[index]);
		} else {
			Frame[] frames = Data.GetFrames(Mathf.Clamp(Index - smoothing, 1, Data.GetTotalFrames()), Mathf.Clamp(Index + smoothing, 1, Data.GetTotalFrames()));
			Vector3 P = Vector3.zero;
			Vector3 Z = Vector3.zero;
			Vector3 Y = Vector3.zero;
			float sum = 0f;
			for(int i=0; i<frames.Length; i++) {
				float weight = 2f * (float)(i+1) / (float)(frames.Length+1);
				if(weight > 1f) {
					weight = 2f - weight;
				}
				Matrix4x4 matrix = mirrored ? frames[i].World[Data.Symmetry[index]].GetMirror(Data.GetAxis(Data.MirrorAxis)) : frames[i].World[index];
				P += weight * matrix.GetPosition();
				Z += weight * matrix.GetForward();
				Y += weight * matrix.GetUp();
				sum += weight;
			}
			P /= sum;
			Z /= sum;
			Y /= sum;
			return Matrix4x4.TRS(Vector3.zero, Quaternion.identity, Data.Scaling * Vector3.one) * Matrix4x4.TRS(P, Quaternion.LookRotation(Z, Y), Vector3.one);
		}
	}

	public Vector3[] GetBoneVelocities(bool mirrored) {
		Vector3[] velocities = new Vector3[World.Length];
		for(int i=0; i<World.Length; i++) {
			velocities[i] = GetBoneVelocity(i, mirrored);
		}
		return velocities;
	}

	public Vector3 GetBoneVelocity(int index, bool mirrored) {
		if(Index == 1) {
			return GetNextFrame().GetBoneVelocity(index, mirrored);
		} else {
			return (GetBoneTransformation(index, mirrored).GetPosition() - GetPreviousFrame().GetBoneTransformation(index, mirrored).GetPosition()) * Data.Framerate;
		}
	}

	public Matrix4x4 GetRootTransformation(bool mirrored) {
		return Matrix4x4.TRS(GetRootPosition(mirrored), GetRootRotation(mirrored), Vector3.one);
	}

	private Vector3 GetRootPosition(bool mirrored) {
		return Utility.ProjectGround(GetBoneTransformation(0, mirrored, Data.RootSmoothing).GetPosition(), Data.Ground);
	}

	private Quaternion GetRootRotation(bool mirrored) {
		
		//Vector3 v1 = GetBoneTransformation(Data.Source.FindBone("RightHip").Index, mirrored, Data.RootSmoothing).GetPosition() - GetBoneTransformation(Data.Source.FindBone("LeftHip").Index, mirrored, Data.RootSmoothing).GetPosition();
		//Vector3 v2 = GetBoneTransformation(Data.Source.FindBone("RightShoulder").Index, mirrored, Data.RootSmoothing).GetPosition() - GetBoneTransformation(Data.Source.FindBone("LeftShoulder").Index, mirrored, Data.RootSmoothing).GetPosition();
		//v1.y = 0f;
		//v2.y = 0f;
		//Vector3 v = (v1+v2).normalized;
		//Vector3 forward = -Vector3.Cross(v, Vector3.up);
		//forward.y = 0f;
		

		/*
		Vector3 neck = GetBoneTransformation(Data.Source.FindBone("Neck").Index, mirrored, Data.RootSmoothing).GetPosition();
		Vector3 hips = GetBoneTransformation(Data.Source.FindBone("Hips").Index, mirrored, Data.RootSmoothing).GetPosition();
		//int leftShoulder = Data.Source.FindBone("LeftShoulder").Index;
		//int rightShoulder = Data.Source.FindBone("RightShoulder").Index;
		//int leftUpLeg = Data.Source.FindBone("LeftUpLeg").Index;
		////int rightUpLeg = Data.Source.FindBone("RightUpLeg").Index;
		Vector3 forward = Vector3.zero;
		forward += neck - hips;
		//forward += GetBoneTransformation(leftShoulder, mirrored, Data.RootSmoothing).GetPosition() - GetBoneTransformation(leftUpLeg, mirrored, Data.RootSmoothing).GetPosition();
		//forward += GetBoneTransformation(rightShoulder, mirrored, Data.RootSmoothing).GetPosition() - GetBoneTransformation(rightUpLeg, mirrored, Data.RootSmoothing).GetPosition();
		*/

		Vector3 forward = GetBoneTransformation(Data.Source.FindBoneContains("Hip").Index, mirrored, Data.RootSmoothing).GetForward();

		forward.y = 0f;
		return Quaternion.LookRotation(forward.normalized, Vector3.up);
	}

	public Vector3 GetRootVelocity(bool mirrored) {
		if(Index == 1) {
			return GetNextFrame().GetRootVelocity(mirrored);
		} else {
			Vector3 velocity = (GetBoneTransformation(0, mirrored, Data.RootSmoothing).GetPosition() - GetPreviousFrame().GetBoneTransformation(0, mirrored, Data.RootSmoothing).GetPosition()) * Data.Framerate;
			velocity.y = 0f;
			return velocity;
		}
	}

	public Vector3 GetRootMotion(bool mirrored) {
		if(Index == 1) {
			return GetNextFrame().GetRootMotion(mirrored);
		} else {
			Matrix4x4 reference = GetPreviousFrame().GetRootTransformation(mirrored);
			Matrix4x4 current = GetRootTransformation(mirrored);
			Matrix4x4 delta = current.GetRelativeTransformationTo(reference);
			Vector3 translationalMotion = delta.GetPosition() * Data.Framerate;
			float angularMotion = Vector3.SignedAngle(Vector3.forward, delta.GetForward(), Vector3.up) * Data.Framerate;
			return new Vector3(translationalMotion.x, angularMotion, translationalMotion.z);
		}
	}

	public float GetSpeed(bool mirrored) {
		float length = 0f;
		Vector3[] positions = new Vector3[6];
		positions[0] = GetRootPosition(mirrored);
		positions[0].y = 0f;
		for(int i=1; i<=5; i++) {
			Frame future = Data.GetFrame(Mathf.Clamp(Timestamp + (float)i/5f, 0f, Data.GetTotalTime()));
			positions[i] = future.GetRootPosition(mirrored);
			positions[i].y = 0f;
		}
		for(int i=1; i<=5; i++) {
			length += Vector3.Distance(positions[i-1], positions[i]);
		}
		return length;
	}

	public Trajectory GetTrajectory(bool mirrored) {
		StyleModule styleModule = Data.GetModule(DataModule.TYPE.Style) == null ? null : (StyleModule)Data.GetModule(DataModule.TYPE.Style);
		
		Trajectory trajectory = new Trajectory(12, styleModule == null ? 0 : styleModule.Functions.Length);

		//Current
		trajectory.Points[6].SetTransformation(GetRootTransformation(mirrored));
		trajectory.Points[6].SetVelocity(GetRootVelocity(mirrored));
		trajectory.Points[6].SetSpeed(GetSpeed(mirrored));
		trajectory.Points[6].Styles = styleModule == null ? new float[0] : styleModule.GetStyle(this);

		//Past
		for(int i=0; i<6; i++) {
			float delta = -1f + (float)i/6f;
			if(Timestamp + delta < 0f) {
				float pivot = - Timestamp - delta;
				float clamped = Mathf.Clamp(pivot, 0f, Data.GetTotalTime());
				Frame reference = Data.GetFrame(clamped);
				float ratio = 1f;
				trajectory.Points[i].SetPosition(Data.GetFirstFrame().GetRootPosition(mirrored) - ratio * (reference.GetRootPosition(mirrored) - Data.GetFirstFrame().GetRootPosition(mirrored)));
				trajectory.Points[i].SetRotation(reference.GetRootRotation(mirrored));
				trajectory.Points[i].SetVelocity(reference.GetRootVelocity(mirrored));
				trajectory.Points[i].SetSpeed(reference.GetSpeed(mirrored));
				trajectory.Points[i].Styles = styleModule == null ? new float[0] : styleModule.GetStyle(reference);
			} else {
				Frame previous = Data.GetFrame(Mathf.Clamp(Timestamp + delta, 0f, Data.GetTotalTime()));
				trajectory.Points[i].SetTransformation(previous.GetRootTransformation(mirrored));
				trajectory.Points[i].SetVelocity(previous.GetRootVelocity(mirrored));
				trajectory.Points[i].SetSpeed(previous.GetSpeed(mirrored));
				trajectory.Points[i].Styles = styleModule == null ? new float[0] : styleModule.GetStyle(previous);
			}
		}

		//Future
		for(int i=1; i<=5; i++) {
			float delta = (float)i/5f;
			if(Timestamp + delta > Data.GetTotalTime()) {
				float pivot = 2f*Data.GetTotalTime() - Timestamp - delta;
				float clamped = Mathf.Clamp(pivot, 0f, Data.GetTotalTime());
				float ratio = 1f;
				Frame reference = Data.GetFrame(clamped);
				trajectory.Points[6+i].SetPosition(Data.GetLastFrame().GetRootPosition(mirrored) - ratio * (reference.GetRootPosition(mirrored) - Data.GetLastFrame().GetRootPosition(mirrored)));
				trajectory.Points[6+i].SetRotation(reference.GetRootRotation(mirrored));
				trajectory.Points[6+i].SetVelocity(reference.GetRootVelocity(mirrored));
				trajectory.Points[6+i].SetSpeed(reference.GetSpeed(mirrored));
				trajectory.Points[6+i].Styles = styleModule == null ? new float[0] : styleModule.GetStyle(reference);
			} else {
				Frame future = Data.GetFrame(Mathf.Clamp(Timestamp + delta, 0f, Data.GetTotalTime()));
				trajectory.Points[6+i].SetTransformation(future.GetRootTransformation(mirrored));
				trajectory.Points[6+i].SetVelocity(future.GetRootVelocity(mirrored));
				trajectory.Points[6+i].SetSpeed(future.GetSpeed(mirrored));
				trajectory.Points[6+i].Styles = styleModule == null ? new float[0] : styleModule.GetStyle(future);
			}
		}

		return trajectory;
	}

}
#endif