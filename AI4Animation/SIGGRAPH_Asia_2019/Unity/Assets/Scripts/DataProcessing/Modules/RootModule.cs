#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class RootModule : Module {

	public enum TOPOLOGY {Biped, Quadruped, Custom};

	public TOPOLOGY Topology = TOPOLOGY.Biped;
	public int RightShoulder, LeftShoulder, RightHip, LeftHip, Neck, Hips;
	public LayerMask Ground = -1;
    public Axis ForwardAxis = Axis.ZPositive;

	public override ID GetID() {
		return ID.Root;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		DetectSetup();
		return this;
	}

	public override void Slice(Sequence sequence) {

	}

	public override void Callback(MotionEditor editor) {
		/*
		Actor actor = editor.GetActor();
		if(actor.Bones.Length == 0) {
			return;
		}
		if(actor.Bones[0].Transform == actor.transform) {
			return;
		}
		Transform bone = actor.Bones[0].Transform;
		Transform parent = bone.parent;
		bone.SetParent(null);
		Matrix4x4 root = GetRootTransformation(editor.GetCurrentFrame(), editor.Mirror);
		actor.transform.position = root.GetPosition();
		actor.transform.rotation = root.GetRotation();
		bone.SetParent(parent);
		bone.transform.localScale = Vector3.one;
		*/
	}

	protected override void DerivedDraw(MotionEditor editor) {
		
	}

	protected override void DerivedInspector(MotionEditor editor) {
		Topology = (TOPOLOGY)EditorGUILayout.EnumPopup("Topology", Topology);
		RightShoulder = EditorGUILayout.Popup("Right Shoulder", RightShoulder, Data.Source.GetBoneNames());
		LeftShoulder = EditorGUILayout.Popup("Left Shoulder", LeftShoulder, Data.Source.GetBoneNames());
		RightHip = EditorGUILayout.Popup("Right Hip", RightHip, Data.Source.GetBoneNames());
		LeftHip = EditorGUILayout.Popup("Left Hip", LeftHip, Data.Source.GetBoneNames());
		Neck = EditorGUILayout.Popup("Neck", Neck, Data.Source.GetBoneNames());
		Hips = EditorGUILayout.Popup("Hips", Hips, Data.Source.GetBoneNames());
		ForwardAxis = (Axis)EditorGUILayout.EnumPopup("Forward Axis", ForwardAxis);
		Ground = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Ground Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Ground), InternalEditorUtility.layers));
	}

	public void DetectSetup() {
		MotionData.Hierarchy.Bone rs = Data.Source.FindBoneContains("RightShoulder");
		RightShoulder = rs == null ? 0 : rs.Index;
		MotionData.Hierarchy.Bone ls = Data.Source.FindBoneContains("LeftShoulder");
		LeftShoulder = ls == null ? 0 : ls.Index;
		MotionData.Hierarchy.Bone rh = Data.Source.FindBoneContains("RightHip");
		RightHip = rh == null ? 0 : rh.Index;
		MotionData.Hierarchy.Bone lh = Data.Source.FindBoneContains("LeftHip");
		LeftHip = lh == null ? 0 : lh.Index;
		MotionData.Hierarchy.Bone n = Data.Source.FindBoneContains("Neck");
		Neck = n == null ? 0 : n.Index;
		MotionData.Hierarchy.Bone h = Data.Source.FindBoneContains("Hips");
		Hips = h == null ? 0 : h.Index;
		Ground = LayerMask.GetMask("Ground");
	}

	public Matrix4x4 GetRootTransformation(Frame frame, bool mirrored) {
		return Matrix4x4.TRS(GetRootPosition(frame, mirrored), GetRootRotation(frame, mirrored), Vector3.one);
	}

	public Vector3 GetRootPosition(Frame frame, bool mirrored) {
		return Utility.ProjectGround(frame.GetBoneTransformation(0, mirrored).GetPosition(), Ground);
	}

	public Quaternion GetRootRotation(Frame frame, bool mirrored) {
		if(Topology == TOPOLOGY.Biped) {
			Vector3 v1 = Vector3.ProjectOnPlane(frame.GetBoneTransformation(RightHip, mirrored).GetPosition() - frame.GetBoneTransformation(LeftHip, mirrored).GetPosition(), Vector3.up).normalized;
			Vector3 v2 = Vector3.ProjectOnPlane(frame.GetBoneTransformation(RightShoulder, mirrored).GetPosition() - frame.GetBoneTransformation(LeftShoulder, mirrored).GetPosition(), Vector3.up).normalized;
			Vector3 v = (v1+v2).normalized;
			Vector3 forward = Vector3.ProjectOnPlane(-Vector3.Cross(v, Vector3.up), Vector3.up).normalized;
			return forward == Vector3.zero ? Quaternion.identity : Quaternion.LookRotation(forward, Vector3.up);
		}
		if(Topology == TOPOLOGY.Quadruped) {
			Vector3 neck = frame.GetBoneTransformation(Neck, mirrored).GetPosition();
			Vector3 hips = frame.GetBoneTransformation(Hips, mirrored).GetPosition();
			Vector3 forward = Vector3.ProjectOnPlane(neck - hips, Vector3.up).normalized;;
			return forward == Vector3.zero ? Quaternion.identity : Quaternion.LookRotation(forward.normalized, Vector3.up);
		}
		if(Topology == TOPOLOGY.Custom) {
			return Quaternion.LookRotation(
				Vector3.ProjectOnPlane(Quaternion.FromToRotation(Vector3.forward, ForwardAxis.GetAxis()) * frame.GetBoneTransformation(0, mirrored).GetForward(), Vector3.up).normalized, 
				Vector3.up
			);
		}
		return Quaternion.identity;
	}

	public Vector3 GetRootVelocity(Frame frame, bool mirrored, float delta) {
		return Vector3.ProjectOnPlane(frame.GetBoneVelocity(0, mirrored, delta), Vector3.up);
	}

	// public float GetRootSpeed(Frame frame, bool mirrored, float window, int step) {
	// 	float length = 0f;
	// 	int count = 0;
	// 	Vector3 prev = Vector3.zero;
	// 	while(true) {
	// 		float delta = step * count/Data.Framerate;
	// 		if(delta > window) {
	// 			break;
	// 		}
	// 		Vector3 pos = GetEstimatedRootPosition(frame, delta, mirrored);
	// 		pos.y = 0f;
	// 		if(count > 0) {
	// 			length += Vector3.Distance(prev, pos);
	// 		}
	// 		prev = pos;
	// 		count += 1;
	// 	}
	// 	return length / window;
	// }

	public Matrix4x4 GetEstimatedRootTransformation(Frame reference, float offset, bool mirrored) {
		return Matrix4x4.TRS(GetEstimatedRootPosition(reference, offset, mirrored), GetEstimatedRootRotation(reference, offset, mirrored), Vector3.one);
	}

	public Vector3 GetEstimatedRootPosition(Frame reference, float offset, bool mirrored) {
		float t = reference.Timestamp + offset;
		if(t < 0f || t > Data.GetTotalTime()) {
			float boundary = Mathf.Clamp(t, 0f, Data.GetTotalTime());
			float pivot = 2f*boundary - t;
			float clamped = Mathf.Clamp(pivot, 0f, Data.GetTotalTime());
			return 2f*GetRootPosition(Data.GetFrame(boundary), mirrored) - GetRootPosition(Data.GetFrame(clamped), mirrored);
		} else {
			return GetRootPosition(Data.GetFrame(t), mirrored);
		}
	}

	public Quaternion GetEstimatedRootRotation(Frame reference, float offset, bool mirrored) {
		float t = reference.Timestamp + offset;
		if(t < 0f || t > Data.GetTotalTime()) {
			float boundary = Mathf.Clamp(t, 0f, Data.GetTotalTime());
			float pivot = 2f*boundary - t;
			float clamped = Mathf.Clamp(pivot, 0f, Data.GetTotalTime());
			return GetRootRotation(Data.GetFrame(clamped), mirrored);
		} else {
			return GetRootRotation(Data.GetFrame(t), mirrored);
		}
	}

	public Vector3 GetEstimatedRootVelocity(Frame reference, float offset, bool mirrored, float delta) {
		return (GetEstimatedRootPosition(reference, offset + delta, mirrored) - GetEstimatedRootPosition(reference, offset, mirrored)) / delta;
	}

	/*
	public float GetTargetSpeed(Frame frame, bool mirrored, float window) {
		List<Vector3> positions = new List<Vector3>();
		int count = 0;
		while(true) {
			float delta = count/Data.Framerate;
			if(frame.Timestamp + delta > window) {
				break;
			}
			count += 1;
			Vector3 p = GetRootTransformation(frame, mirrored, delta).GetPosition();
			p.y = 0f;
			positions.Add(p);
		}
		if(positions.Count == 0) {
			Debug.Log("Oups! Something went wrong in computing target speed.");
			return 0f;
		} else {
			float speed = 0f;
			for(int i=1; i<positions.Count; i++) {
				speed += Vector3.Distance(positions[i-1], positions[i]);
			}
			return speed/window;
		};
	}
	*/

}
#endif
