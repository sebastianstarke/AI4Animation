using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class Actor : MonoBehaviour {

	public bool InspectSkeleton = false;
	
	public bool DrawRoot = false;
	public bool DrawSkeleton = true;
	public bool DrawVelocities = false;
	public bool DrawTransforms = false;
	
	public float BoneSize = 0.025f;
	public Color BoneColor = UltiDraw.Black;
	public Color JointColor = UltiDraw.Mustard;

	public Bone[] Bones = new Bone[0];

	void Reset() {
		ExtractSkeleton();
	}

	public Transform GetRoot() {
		return transform;
	}

	public Transform FindTransform(string name) {
		Transform element = null;
		Action<Transform> recursion = null;
		recursion = new Action<Transform>((transform) => {
			if(transform.name == name) {
				element = transform;
				return;
			}
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i));
			}
		});
		recursion(GetRoot());
		return element;
	}

	public Bone FindBone(Transform transform) {
		return Array.Find(Bones, x => x.Transform == transform);
	}

	public Bone FindBone(string name) {
		return Array.Find(Bones, x => x.GetName() == name);
	}

	public Bone FindBoneContains(string name) {
		return Array.Find(Bones, x => x.GetName().Contains(name));
	}

	public void ExtractSkeleton() {
		ArrayExtensions.Clear(ref Bones);
		Action<Transform, Bone> recursion = null;
		recursion = new Action<Transform, Bone>((transform, parent) => {
			Bone bone = new Bone(this, transform, Bones.Length);
			ArrayExtensions.Add(ref Bones, bone);
			if(parent != null) {
				bone.Parent = parent.Index;
				ArrayExtensions.Add(ref parent.Childs, bone.Index);
			}
			parent = bone;
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i), parent);
			}
		});
		recursion(GetRoot(), null);
	}

	public void ExtractSkeleton(Transform[] bones) {
		ArrayExtensions.Clear(ref Bones);
		Action<Transform, Bone> recursion = null;
		recursion = new Action<Transform, Bone>((transform, parent) => {
			if(System.Array.Find(bones, x => x == transform)) {
				Bone bone = new Bone(this, transform, Bones.Length);
				ArrayExtensions.Add(ref Bones, bone);
				if(parent != null) {
					bone.Parent = parent.Index;
					ArrayExtensions.Add(ref parent.Childs, bone.Index);
				}
				parent = bone;
			}
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i), parent);
			}
		});
		recursion(GetRoot(), null);
	}

	/*
	public Matrix4x4[] GetLocalPosture() {
		Matrix4x4[] posture = new Matrix4x4[Bones.Length];
		for(int i=0; i<posture.Length; i++) {
			posture[i] = i == 0 ? Bones[i].Transform.GetWorldMatrix() : Bones[i].Transform.GetLocalMatrix();
		}
		return posture;
	}

	public Matrix4x4[] GetWorldPosture() {
		Matrix4x4[] posture = new Matrix4x4[Bones.Length];
		for(int i=0; i<posture.Length; i++) {
			posture[i] = Bones[i].Transform.GetWorldMatrix();
		}
		return posture;
	}
	*/

	public Matrix4x4[] GetPosture() {
		Matrix4x4[] posture = new Matrix4x4[Bones.Length];
		for(int i=0; i<posture.Length; i++) {
			posture[i] = Bones[i].Transform.GetWorldMatrix();
		}
		return posture;
	}

	public Vector3[] GetVelocities() {
		Vector3[] velocities = new Vector3[Bones.Length];
		for(int i=0; i<velocities.Length; i++) {
			velocities[i] = Bones[i].Velocity;
		}
		return velocities;
	}

	public void Draw() {
		Draw(BoneColor, JointColor, 1f);
	}

	public void Draw(Color boneColor, Color jointColor, float alpha) {
		UltiDraw.Begin();
		if(DrawRoot) {
			UltiDraw.DrawTranslateGizmo(GetRoot().position, GetRoot().rotation, 0.1f);
			UltiDraw.DrawSphere(GetRoot().position, GetRoot().rotation, 0.025f, UltiDraw.Black);
			UltiDraw.DrawLine(Bones[0].Transform.position, GetRoot().position, UltiDraw.Mustard);
		}

		if(DrawSkeleton) {
			Action<Bone> recursion = null;
			recursion = new Action<Bone>((bone) => {
				if(bone.GetParent() != null) {
					//if(bone.GetLength() > 0.05f) {
						UltiDraw.DrawBone(
							bone.GetParent().Transform.position,
							Quaternion.FromToRotation(bone.GetParent().Transform.forward, bone.Transform.position - bone.GetParent().Transform.position) * bone.GetParent().Transform.rotation,
							12.5f*BoneSize*bone.GetLength(), bone.GetLength(),
							boneColor.Transparent(alpha)
						);
					//}
				}
				UltiDraw.DrawSphere(
					bone.Transform.position,
					Quaternion.identity,
					5f/8f * BoneSize,
					jointColor.Transparent(alpha)
				);
				for(int i=0; i<bone.Childs.Length; i++) {
					recursion(bone.GetChild(i));
				}
			});
			if(Bones.Length > 0) {
				recursion(Bones[0]);
			}
		}

		if(DrawVelocities) {
			UltiDraw.Begin();
			for(int i=0; i<Bones.Length; i++) {
				UltiDraw.DrawArrow(
					Bones[i].Transform.position,
					Bones[i].Transform.position + Bones[i].Velocity,
					0.75f,
					0.0075f,
					0.05f,
					UltiDraw.Purple.Transparent(0.5f)
				);
			}
			UltiDraw.End();
		}

		if(DrawTransforms) {
			Action<Bone> recursion = null;
			recursion = new Action<Bone>((bone) => {
				UltiDraw.DrawTranslateGizmo(bone.Transform.position, bone.Transform.rotation, 0.05f);
				for(int i=0; i<bone.Childs.Length; i++) {
					recursion(bone.GetChild(i));
				}
			});
			if(Bones.Length > 0) {
				recursion(Bones[0]);
			}
		}
		UltiDraw.End();
	}

	public void DrawSimple(Color color) {
		UltiDraw.Begin();
		if(DrawSkeleton) {
			Action<Bone> recursion = null;
			recursion = new Action<Bone>((bone) => {
				if(bone.GetParent() != null) {
					UltiDraw.DrawLine(bone.GetParent().Transform.position, bone.Transform.position, color);
				}
				UltiDraw.DrawCircle(bone.Transform.position, 0.02f, Color.Lerp(color, UltiDraw.Black, 0.25f));
				for(int i=0; i<bone.Childs.Length; i++) {
					recursion(bone.GetChild(i));
				}
			});
			if(Bones.Length > 0) {
				recursion(Bones[0]);
			}
		}
		UltiDraw.End();
	}

	public void DrawSimple(Color color, Matrix4x4[] transformations) {
		UltiDraw.Begin();
		if(DrawSkeleton) {
			Action<Bone> recursion = null;
			recursion = new Action<Bone>((bone) => {
				if(bone.GetParent() != null) {
					UltiDraw.DrawLine(transformations[bone.GetParent().Index].GetPosition(), transformations[bone.Index].GetPosition(), color);
				}
				UltiDraw.DrawCircle(transformations[bone.Index].GetPosition(), 0.02f, Color.Lerp(color, UltiDraw.Black, 0.25f));
				for(int i=0; i<bone.Childs.Length; i++) {
					recursion(bone.GetChild(i));
				}
			});
			if(Bones.Length > 0) {
				recursion(Bones[0]);
			}
		}
		UltiDraw.End();
	}

	void OnRenderObject() {
		Draw();
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			OnRenderObject();
		}
	}

	[Serializable]
	public class Bone {
		public Actor Actor;
		public Transform Transform;
		public Vector3 Velocity;
		public int Index;
		public int Parent;
		public int[] Childs;

		public Bone(Actor actor, Transform transform, int index) {
			Actor = actor;
			Transform = transform;
			Velocity = Vector3.zero;
			Index = index;
			Parent = -1;
			Childs = new int[0];
		}

		public string GetName() {
			return Transform.name;
		}

		public Bone GetParent() {
			return Parent == -1 ? null : Actor.Bones[Parent];
		}

		public Bone GetChild(int index) {
			return index >= Childs.Length ? null : Actor.Bones[Childs[index]];
		}

		public float GetLength() {
			if(GetParent() == null) {
				return 0f;
			} else {
				return Vector3.Distance(GetParent().Transform.position, Transform.position);
			}
		}
	}

	#if UNITY_EDITOR
	[CustomEditor(typeof(Actor))]
	public class Actor_Editor : Editor {

		public Actor Target;

		void Awake() {
			Target = (Actor)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);
	
			EditorGUILayout.LabelField("Name: " + Target.name);
			EditorGUILayout.ObjectField("Root", Target.GetRoot(), typeof(Transform), true);
			Target.DrawRoot = EditorGUILayout.Toggle("Draw Root", Target.DrawRoot);
			Target.DrawSkeleton = EditorGUILayout.Toggle("Draw Skeleton", Target.DrawSkeleton);
			Target.DrawVelocities = EditorGUILayout.Toggle("Draw Velocities", Target.DrawVelocities);
			Target.DrawTransforms = EditorGUILayout.Toggle("Draw Transforms", Target.DrawTransforms);

			Utility.SetGUIColor(Color.white);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				Utility.SetGUIColor(Color.grey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					if(Utility.GUIButton("Skeleton", UltiDraw.DarkGrey, UltiDraw.White)) {
						Target.InspectSkeleton = !Target.InspectSkeleton;
					}
					if(Target.InspectSkeleton) {
						EditorGUILayout.LabelField("Skeleton Bones: " + Target.Bones.Length);
						Target.BoneSize = EditorGUILayout.FloatField("Bone Size", Target.BoneSize);
						Target.JointColor = EditorGUILayout.ColorField("Joint Color", Target.JointColor);
						Target.BoneColor = EditorGUILayout.ColorField("Bone Color", Target.BoneColor);
						InspectSkeleton(Target.GetRoot(), 0);
					}
				}
			}

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

		private void InspectSkeleton(Transform transform, int indent) {
			Bone bone = Target.FindBone(transform.name);
			Utility.SetGUIColor(bone == null ? UltiDraw.LightGrey : UltiDraw.Mustard);
			using(new EditorGUILayout.HorizontalScope ("Box")) {
				Utility.ResetGUIColor();
				EditorGUILayout.BeginHorizontal();
				for(int i=0; i<indent; i++) {
					EditorGUILayout.LabelField("|", GUILayout.Width(20f));
				}
				EditorGUILayout.LabelField("-", GUILayout.Width(20f));
				EditorGUILayout.LabelField(transform.name + " " + (bone == null ? string.Empty : "(" + bone.Index.ToString() + ")"), GUILayout.Width(100f), GUILayout.Height(20f));
				GUILayout.FlexibleSpace();

				if(Utility.GUIButton("Bone", bone == null ? UltiDraw.White : UltiDraw.DarkGrey, bone == null ? UltiDraw.DarkGrey : UltiDraw.White)) {
					Transform[] bones = new Transform[Target.Bones.Length];
					for(int i=0; i<bones.Length; i++) {
						bones[i] = Target.Bones[i].Transform;
					}
					if(bone == null) {
						ArrayExtensions.Add(ref bones, transform);
						Target.ExtractSkeleton(bones);
					} else {
						ArrayExtensions.Remove(ref bones, transform);
						Target.ExtractSkeleton(bones);
					}
				}
				EditorGUILayout.EndHorizontal();
			}
			for(int i=0; i<transform.childCount; i++) {
				InspectSkeleton(transform.GetChild(i), indent+1);
			}
		}

	}
	#endif

}
