using System;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

[ExecuteInEditMode]
public class Actor : MonoBehaviour {

	public bool InspectSkeleton = false;
	
	public bool DrawRoot = false;
	public bool DrawSkeleton = true;
	public bool DrawSketch = false;
	public bool DrawTransforms = false;
	public bool DrawVelocities = false;
	public bool DrawHistory = false;

	public int MaxHistory = 0;
	public int Sampling = 0;

	public float BoneSize = 0.025f;
	public Color BoneColor = UltiDraw.Cyan;
	public Color JointColor = UltiDraw.Mustard;

	public Bone[] Bones = new Bone[0];

	private List<State> History = new List<State>();

	private string[] BoneNames = null;

	void Reset() {
		ExtractSkeleton();
	}

	void LateUpdate() {
		SaveState();
	}

	public void CopySetup(Actor reference) {
		ExtractSkeleton(reference.GetBoneNames());
	}

	public void SaveState() {
		if(MaxHistory > 0) {
			State state = new State();
			state.Transformations = GetBoneTransformations();
			state.Velocities = GetBoneVelocities();
			History.Add(state);
		}
		while(History.Count > MaxHistory) {
			History.RemoveAt(0);
		}
	}

	public Transform GetRoot() {
		return transform;
	}

	public Transform[] FindTransforms(params string[] names) {
		Transform[] transforms = new Transform[names.Length];
		for(int i=0; i<transforms.Length; i++) {
			transforms[i] = FindTransform(names[i]);
		}
		return transforms;
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

	public Bone[] FindBones(params Transform[] transforms) {
		Bone[] bones = new Bone[transforms.Length];
		for(int i=0; i<bones.Length; i++) {
			bones[i] = FindBone(transforms[i]);
		}
		return bones;
	}

	public Bone[] FindBones(params string[] names) {
		Bone[] bones = new Bone[names.Length];
		for(int i=0; i<bones.Length; i++) {
			bones[i] = FindBone(names[i]);
		}
		return bones;
	}

	public Bone FindBone(Transform transform) {
		return FindBone(transform.name);
	}

	public Bone FindBone(string name) {
		return Array.Find(Bones, x => x.GetName() == name);
	}

	public Bone FindBoneContains(string name) {
		return Array.Find(Bones, x => x.GetName().Contains(name));
	}

	public string[] GetBoneNames() {
		if(BoneNames == null || BoneNames.Length != Bones.Length) {
			BoneNames = new string[Bones.Length];
			for(int i=0; i<BoneNames.Length; i++) {
				BoneNames[i] = Bones[i].GetName();
			}
		}
		return BoneNames;
	}

	public int[] GetBoneIndices(params string[] names) {
		int[] indices = new int[names.Length];
		for(int i=0; i<indices.Length; i++) {
			indices[i] = FindBone(names[i]).Index;
		}
		return indices;
	}

	public Transform[] GetBoneTransforms(params string[] names) {
		Transform[] transforms = new Transform[names.Length];
		for(int i=0; i<names.Length; i++) {
			transforms[i] = FindTransform(names[i]);
		}
		return transforms;
	}

	public void ExtractSkeleton() {
		ArrayExtensions.Clear(ref Bones);
		Action<Transform, Bone> recursion = null;
		recursion = new Action<Transform, Bone>((transform, parent) => {
			Bone bone = new Bone(this, transform, Bones.Length);
			ArrayExtensions.Append(ref Bones, bone);
			if(parent != null) {
				bone.Parent = parent.Index;
				ArrayExtensions.Append(ref parent.Childs, bone.Index);
			}
			parent = bone;
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i), parent);
			}
		});
		recursion(GetRoot(), null);
		BoneNames = new string[0];
	}

	public void ExtractSkeleton(Transform[] bones) {
		ArrayExtensions.Clear(ref Bones);
		Action<Transform, Bone> recursion = null;
		recursion = new Action<Transform, Bone>((transform, parent) => {
			if(System.Array.Find(bones, x => x == transform)) {
				Bone bone = new Bone(this, transform, Bones.Length);
				ArrayExtensions.Append(ref Bones, bone);
				if(parent != null) {
					bone.Parent = parent.Index;
					ArrayExtensions.Append(ref parent.Childs, bone.Index);
				}
				parent = bone;
			}
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i), parent);
			}
		});
		recursion(GetRoot(), null);
		BoneNames = new string[0];
	}

	public void ExtractSkeleton(string[] bones) {
		ExtractSkeleton(FindTransforms(bones));
	}

	public void WriteTransforms(Matrix4x4[] values, string[] names) {
		Action<Transform> recursion = null;
		recursion = new Action<Transform>((t) => {
			int index = ArrayExtensions.FindIndex(ref names, t.name);
			if(index > -1) {
				t.position = values[index].GetPosition();
				t.rotation = values[index].GetRotation();
			}
			for(int i=0; i<t.childCount; i++) {
				recursion(t.GetChild(i));
			}
		});
		recursion(transform);
	}

	public void SetBoneTransformations(Matrix4x4[] values) {
		if(values.Length != Bones.Length) {
			return;
		}
		for(int i=0; i<Bones.Length; i++) {
			Bones[i].Transform.position = values[i].GetPosition();
			Bones[i].Transform.rotation = values[i].GetRotation();
		}
	}

	public void SetBoneTransformations(Matrix4x4[] values, params string[] bones) {
		for(int i=0; i<bones.Length; i++) {
			SetBoneTransformation(values[i], bones[i]);
		}
	}

	public void SetBoneTransformation(Matrix4x4 value, string bone) {
		Bone b = FindBone(bone);
		if(b != null) {
			b.Transform.position = value.GetPosition();
			b.Transform.rotation = value.GetRotation();
		}
	}

	public Matrix4x4[] GetBoneTransformations() {
		Matrix4x4[] transformations = new Matrix4x4[Bones.Length];
		for(int i=0; i<transformations.Length; i++) {
			transformations[i] = Bones[i].Transform.GetWorldMatrix();
		}
		return transformations;
	}

	public Matrix4x4[] GetBoneTransformations(params string[] bones) {
		Matrix4x4[] transformations = new Matrix4x4[bones.Length];
		for(int i=0; i<transformations.Length; i++) {
			transformations[i] = GetBoneTransformation(bones[i]);
		}
		return transformations;
	}

	public Matrix4x4 GetBoneTransformation(string bone) {
		return FindBone(bone).Transform.GetWorldMatrix();
	}

	public void SetBoneVelocities(Vector3[] values) {
		if(values.Length != Bones.Length) {
			return;
		}
		for(int i=0; i<Bones.Length; i++) {
			Bones[i].Velocity = values[i];
		}
	}

	public void SetBoneVelocities(Vector3[] values, params string[] bones) {
		for(int i=0; i<bones.Length; i++) {
			SetBoneVelocity(values[i], bones[i]);
		}
	}

	public void SetBoneVelocity(Vector3 value, string bone) {
		Bone b = FindBone(bone);
		if(b != null) {
			b.Velocity = value;
		}
	}

	public Vector3[] GetBoneVelocities() {
		Vector3[] velocities = new Vector3[Bones.Length];
		for(int i=0; i<velocities.Length; i++) {
			velocities[i] = Bones[i].Velocity;
		}
		return velocities;
	}

	public Vector3[] GetBoneVelocities(params string[] bones) {
		Vector3[] velocities = new Vector3[bones.Length];
		for(int i=0; i<velocities.Length; i++) {
			velocities[i] = GetBoneVelocity(bones[i]);
		}
		return velocities;
	}

	public Vector3 GetBoneVelocity(string bone) {
		return FindBone(bone).Velocity;
	}

	public Vector3[] GetBonePositions() {
		Vector3[] positions = new Vector3[Bones.Length];
		for(int i=0; i<positions.Length; i++) {
			positions[i] = Bones[i].Transform.position;
		}
		return positions;
	}

	public Vector3[] GetBonePositions(params string[] bones) {
		Vector3[] positions = new Vector3[bones.Length];
		for(int i=0; i<positions.Length; i++) {
			positions[i] = GetBonePosition(bones[i]);
		}
		return positions;
	}

	public Vector3 GetBonePosition(string bone) {
		return FindBone(bone).Transform.position;
	}

	public Quaternion[] GetBoneRotations() {
		Quaternion[] rotation = new Quaternion[Bones.Length];
		for(int i=0; i<rotation.Length; i++) {
			rotation[i] = Bones[i].Transform.rotation;
		}
		return rotation;
	}

	public Quaternion[] GetBoneRotations(params string[] bones) {
		Quaternion[] rotation = new Quaternion[bones.Length];
		for(int i=0; i<rotation.Length; i++) {
			rotation[i] = GetBoneRotation(bones[i]);
		}
		return rotation;
	}

	public Quaternion GetBoneRotation(string bone) {
		return FindBone(bone).Transform.rotation;
	}

	public Bone[] GetRootBones() {
		List<Bone> bones = new List<Bone>();
		for(int i=0; i<Bones.Length; i++) {
			if(Bones[i].GetParent() == null) {
				bones.Add(Bones[i]);
			}
		}
		return bones.ToArray();
	}

	public void Draw() {
		UltiDraw.Begin();
		if(DrawRoot) {
			UltiDraw.DrawCube(GetRoot().position, GetRoot().rotation, 0.1f, UltiDraw.Black);
			UltiDraw.DrawTranslateGizmo(GetRoot().position, GetRoot().rotation, 0.1f);
		}

		if(DrawSkeleton) {
			Action<Bone> recursion = null;
			recursion = new Action<Bone>((bone) => {
				if(bone.GetParent() != null) {
					UltiDraw.DrawBone(
						bone.GetParent().Transform.position,
						Quaternion.FromToRotation(bone.GetParent().Transform.forward, bone.Transform.position - bone.GetParent().Transform.position) * bone.GetParent().Transform.rotation,
						12.5f*BoneSize*bone.GetLength(), bone.GetLength(),
						bone.Color == UltiDraw.None ? BoneColor : bone.Color
					);
				}
				UltiDraw.DrawSphere(
					bone.Transform.position,
					Quaternion.identity,
					5f/8f * BoneSize,
					JointColor
				);
				for(int i=0; i<bone.Childs.Length; i++) {
					recursion(bone.GetChild(i));
				}
			});
			foreach(Bone bone in GetRootBones()) {
				recursion(bone);
			}
		}

		if(DrawVelocities) {
			for(int i=0; i<Bones.Length; i++) {
				UltiDraw.DrawArrow(
					Bones[i].Transform.position,
					Bones[i].Transform.position + Bones[i].Velocity,
					0.75f,
					0.0075f,
					0.05f,
					UltiDraw.DarkGreen.Opacity(0.5f)
				);
			}
		}

		if(DrawTransforms) {
			Action<Bone> recursion = null;
			recursion = new Action<Bone>((bone) => {
				UltiDraw.DrawTranslateGizmo(bone.Transform.position, bone.Transform.rotation, 0.05f);
				for(int i=0; i<bone.Childs.Length; i++) {
					recursion(bone.GetChild(i));
				}
			});
			foreach(Bone bone in GetRootBones()) {
				recursion(bone);
			}
		}
		UltiDraw.End();

		if(DrawSketch) {
			Sketch(GetBoneTransformations(), BoneColor);
		}

		if(DrawHistory) {
			if(DrawSkeleton) {
				int step = Mathf.Max(Sampling, 1);
				for(int i=0; i<History.Count; i+=step) {
					Sketch(History[i].Transformations, BoneColor.Darken(1f - (float)i/(float)History.Count));
				}
			}
			if(DrawVelocities) {
				float max = 0f;
				float[][] functions = new float[History.Count][];
				for(int i=0; i<History.Count; i++) {
					functions[i] = new float[Bones.Length];
					for(int j=0; j<Bones.Length; j++) {
						functions[i][j] = History[i].Velocities[j].magnitude;
						max = Mathf.Max(max, functions[i][j]);
					}
				}
				UltiDraw.Begin();
				UltiDraw.PlotFunctions(new Vector2(0.5f, 0.05f), new Vector2(0.9f, 0.1f), functions, UltiDraw.Dimension.Y, yMin: 0f, yMax: max, thickness: 0.0025f);
				UltiDraw.End();
			}
		}
	}

	public void Draw(Matrix4x4[] transformations, Color color) {
		UltiDraw.Begin();
		if(transformations.Length != Bones.Length) {
			Debug.Log("Number of given transformations does not match number of bones.");
		} else {
			Action<Bone> recursion = null;
			recursion = new Action<Bone>((bone) => {
				Matrix4x4 current = transformations[bone.Index];
				if(bone.GetParent() != null) {
					Matrix4x4 parent = transformations[bone.GetParent().Index];
					UltiDraw.DrawBone(
						parent.GetPosition(),
						Quaternion.FromToRotation(parent.GetForward(), current.GetPosition() - parent.GetPosition()) * parent.GetRotation(),
						12.5f*BoneSize*bone.GetLength(), bone.GetLength(),
						color
					);
				}
				UltiDraw.DrawSphere(
					current.GetPosition(),
					Quaternion.identity,
					5f/8f * BoneSize,
					color
				);
				for(int i=0; i<bone.Childs.Length; i++) {
					recursion(bone.GetChild(i));
				}
			});
			foreach(Bone bone in GetRootBones()) {
				recursion(bone);
			}
		}
		UltiDraw.End();
	}

	public void Sketch(Matrix4x4[] transformations, Color color) {
		UltiDraw.Begin();
		if(transformations.Length != Bones.Length) {
			Debug.Log("Number of given transformations does not match number of bones.");
		} else {
			Action<Bone> recursion = null;
			recursion = new Action<Bone>((bone) => {
				Matrix4x4 current = transformations[bone.Index];
				if(bone.GetParent() != null) {
					Matrix4x4 parent = transformations[bone.GetParent().Index];
					UltiDraw.DrawLine(parent.GetPosition(), current.GetPosition(), color);
				}
				UltiDraw.DrawCube(current.GetPosition(), current.GetRotation(), 0.02f, color);
				for(int i=0; i<bone.Childs.Length; i++) {
					recursion(bone.GetChild(i));
				}
			});
			foreach(Bone bone in GetRootBones()) {
				recursion(bone);
			}
		}
		UltiDraw.End();
	}

	void OnRenderObject() {
		Draw();
	}

	public class State {
		public Matrix4x4[] Transformations;
		public Vector3[] Velocities;
	}

	[Serializable]
	public class Bone {
		public Color Color;
		public Actor Actor;
		public Transform Transform;
		public Vector3 Velocity;
		public int Index;
		public int Parent;
		public int[] Childs;

		public Bone(Actor actor, Transform transform, int index) {
			Color = UltiDraw.None;
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
			return GetParent() == null ? 0f : Vector3.Distance(GetParent().Transform.position, Transform.position);
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
	
			Target.DrawRoot = EditorGUILayout.Toggle("Draw Root", Target.DrawRoot);
			Target.DrawSkeleton = EditorGUILayout.Toggle("Draw Skeleton", Target.DrawSkeleton);
			Target.DrawTransforms = EditorGUILayout.Toggle("Draw Transforms", Target.DrawTransforms);
			Target.DrawVelocities = EditorGUILayout.Toggle("Draw Velocities", Target.DrawVelocities);
			Target.DrawHistory = EditorGUILayout.Toggle("Draw History", Target.DrawHistory);

			Target.MaxHistory = EditorGUILayout.IntField("Max History", Target.MaxHistory);
			Target.Sampling = EditorGUILayout.IntField("Sampling", Target.Sampling);

			Utility.SetGUIColor(Color.grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				if(Utility.GUIButton("Skeleton", UltiDraw.DarkGrey, UltiDraw.White)) {
					Target.InspectSkeleton = !Target.InspectSkeleton;
				}
				if(Target.InspectSkeleton) {
					Actor reference = (Actor)EditorGUILayout.ObjectField("Reference", null, typeof(Actor), true);
					if(reference != null) {
						Target.CopySetup(reference);
					}
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Bones: " + Target.Bones.Length);
					if(Utility.GUIButton("Clear", UltiDraw.DarkGrey, UltiDraw.White)) {
						Target.ExtractSkeleton(new Transform[0]);
					}
					EditorGUILayout.EndHorizontal();
					Target.BoneSize = EditorGUILayout.FloatField("Bone Size", Target.BoneSize);
					Target.JointColor = EditorGUILayout.ColorField("Joint Color", Target.JointColor);
					Target.BoneColor = EditorGUILayout.ColorField("Bone Color", Target.BoneColor);
					InspectSkeleton(Target.GetRoot(), 0);
				}
			}

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

		private void InspectSkeleton(Transform transform, int indent) {
			float indentSpace = 10f;
			Bone bone = Target.FindBone(transform.name);
			Utility.SetGUIColor(bone == null ? UltiDraw.LightGrey : UltiDraw.Mustard);
			using(new EditorGUILayout.HorizontalScope ("Box")) {
				Utility.ResetGUIColor();
				EditorGUILayout.BeginHorizontal();
				for(int i=0; i<indent; i++) {
					EditorGUILayout.LabelField("|", GUILayout.Width(indentSpace));
				}
				EditorGUILayout.LabelField("-", GUILayout.Width(indentSpace));
				EditorGUILayout.LabelField(transform.name, GUILayout.Width(400f - indent*indentSpace));
				GUILayout.FlexibleSpace();
				if(bone != null) {
					EditorGUILayout.LabelField("Index: " + bone.Index.ToString(), GUILayout.Width(60f));
					EditorGUILayout.LabelField("Length: " + bone.GetLength().ToString("F3"), GUILayout.Width(90f));
					bone.Color = EditorGUILayout.ColorField(bone.Color);
				}
				if(Utility.GUIButton("Bone", bone == null ? UltiDraw.White : UltiDraw.DarkGrey, bone == null ? UltiDraw.DarkGrey : UltiDraw.White)) {
					Transform[] bones = new Transform[Target.Bones.Length];
					for(int i=0; i<bones.Length; i++) {
						bones[i] = Target.Bones[i].Transform;
					}
					if(bone == null) {
						ArrayExtensions.Append(ref bones, transform);
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
