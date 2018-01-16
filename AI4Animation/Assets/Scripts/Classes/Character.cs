using UnityEngine;
using System;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif

[Serializable]
public class Character {

	public enum DRAWTYPE {Diffuse, Transparent}

	public bool Inspect = false;

	public Segment[] Hierarchy = new Segment[0];

	public float BoneSize = 0.025f;
	public Color BoneColor = Utility.Cyan;
	public Color JointColor = Utility.Mustard;
	public DRAWTYPE DrawType = DRAWTYPE.Transparent;
	public bool DrawSkeleton = true;
	public bool DrawTransforms = false;

	private Mesh JointMesh;
	private Mesh BoneMesh;
	private Material DiffuseMaterial;
	private Material TransparentMaterial;

	public Character() {

	}

	public void SetWorldTransformations(Matrix4x4[] transformations) {
		for(int i=0; i<Hierarchy.Length; i++) {
			Hierarchy[i].SetTransformation(transformations[i]);
		}
	}

	public Matrix4x4[] GetWorldTransformations() {
		Matrix4x4[] transformations = new Matrix4x4[Hierarchy.Length];
		for(int i=0; i<Hierarchy.Length; i++) {
			transformations[i] = Hierarchy[i].GetTransformation();
		}
		return transformations;
	}

	public void SetLocalTransformations(Matrix4x4[] transformations) {
		for(int i=0; i<Hierarchy.Length; i++) {
			Segment parent = Hierarchy[i].GetParent(Hierarchy);
			if(parent == null) {
				Hierarchy[i].SetTransformation(transformations[i]);
			} else {
				Hierarchy[i].SetTransformation(parent.GetTransformation() * transformations[i]);
			}
		}
	}

	public Matrix4x4[] GetLocalTransformations() {
		Matrix4x4[] transformations = new Matrix4x4[Hierarchy.Length];
		for(int i=0; i<Hierarchy.Length; i++) {
			Segment parent = Hierarchy[i].GetParent(Hierarchy);
			if(parent == null) {
				transformations[i] = Hierarchy[i].GetTransformation();
			} else {
				transformations[i] = parent.GetTransformation().inverse * Hierarchy[i].GetTransformation();
			}
		}
		return transformations;
	}

	public Matrix4x4[] WorldToLocal(Matrix4x4[] world) {
		Matrix4x4[] local = new Matrix4x4[Hierarchy.Length];
		for(int i=0; i<Hierarchy.Length; i++) {
			Segment parent = Hierarchy[i].GetParent(Hierarchy);
			if(parent == null) {
				local[i] = world[i];
			} else {
				local[i] = world[parent.GetIndex()].inverse * world[i];
			}
		}
		return local;
	}

	public Matrix4x4[] LocalToWorld(Matrix4x4[] local) {
		Matrix4x4[] world = new Matrix4x4[Hierarchy.Length];
		for(int i=0; i<Hierarchy.Length; i++) {
			Segment parent = Hierarchy[i].GetParent(Hierarchy);
			if(parent == null) {
				world[i] = local[i];
			} else {
				world[i] = world[parent.GetIndex()] * local[i];
			}
		}
		return world;
	}

	public string[] GetBoneNames() {
		string[] names = new string[Hierarchy.Length];
		for(int i=0; i<Hierarchy.Length; i++) {
			names[i] = Hierarchy[i].GetName();
		}
		return names;
	}

	public float[] GetBoneLengths() {
		float[] lengths = new float[Hierarchy.Length];
		int index = 0;
		Action<Segment> recursion = null;
		recursion = new Action<Segment>((segment) => {
			Segment parent = segment.GetParent(Hierarchy);
			if(parent == null) {
				lengths[index] = 0f;
			} else {
				lengths[index] = Vector3.Distance(segment.GetTransformation().GetPosition(), parent.GetTransformation().GetPosition());
			}
			index += 1;
			for(int i=0; i<segment.GetChildCount(); i++) {
				recursion(segment.GetChild(Hierarchy, i));
			}
		});
		recursion(GetRoot());
		return lengths;
	}

	public Segment GetRoot() {
		if(Hierarchy.Length == 0) {
			return null;
		}
		return Hierarchy[0];
	}

	public Segment FindSegment(string name) {
		return Array.Find(Hierarchy, x => x.GetName() == name);
	}

	public void WriteTransformations(Transform root) {
		if(Hierarchy.Length == 0) {
			return;
		}
		int index = 0;
		Action<Transform> recursion = null;
		recursion = new Action<Transform>((transform) => {
			if(index == Hierarchy.Length) {
				return;
			}
			if(transform.name == Hierarchy[index].GetName()) {
				transform.position = Hierarchy[index].GetTransformation().GetPosition();
				transform.rotation = Hierarchy[index].GetTransformation().GetRotation();
				index += 1;
			}
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i));
			}
		});
		recursion(root);
	}

	public void FetchTransformations(Transform root) {
		if(Hierarchy.Length == 0) {
			return;
		}
		int index = 0;
		Action<Transform> recursion = null;
		recursion = new Action<Transform>((transform) => {
			if(index == Hierarchy.Length) {
				return;
			}
			if(transform.name == Hierarchy[index].GetName()) {
				Hierarchy[index].SetTransformation(Matrix4x4.TRS(transform.position, transform.rotation, Vector3.one));
				index += 1;
			}
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i));
			}
		});
		recursion(root);
	}

	public bool RebuildRequired(Transform root) {
		if(Hierarchy.Length == 0) {
			return false;
		}
		List<Transform> list = new List<Transform>();
		Action<Transform> recursion = null;
		recursion = new Action<Transform>((transform) => {
			list.Add(transform);
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i));
			}
		});
		recursion(root);
		Transform[] transforms = list.ToArray();
		for(int i=0; i<Hierarchy.Length; i++) {
			if(!Array.Exists(transforms, x => x.name == Hierarchy[i].GetName())) {
				return true;
			}
		}
		return false;
	}

	public void BuildHierarchy(Transform root) {
		Utility.Clear(ref Hierarchy);
		Action<Transform, Segment> recursion = null;
		recursion = new Action<Transform, Segment>((transform, parent) => {
			Segment segment = AddSegment(transform.name, parent);
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i), segment);
			}
		});
		recursion(root, null);
	}

	public void BuildHierarchy(Transform root, string[] names) {
		Utility.Clear(ref Hierarchy);
		Action<Transform, Segment> recursion = null;
		recursion = new Action<Transform, Segment>((transform, parent) => {
			Segment segment = Array.Exists(names, x => x == transform.name) ? AddSegment(transform.name, parent) : parent;
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i), segment);
			}
		});
		recursion(root, null);
	}

	public void BuildHierarchy(string[] names, string[] parents) {
		Utility.Clear(ref Hierarchy);
		for(int i=0; i<names.Length; i++) {
			if(FindSegment(names[i]) != null) {
				Debug.Log("Failed building hierarchy because there were multiple segments with name " + names[i] + ".");
				Utility.Clear(ref Hierarchy);
				return;
			}
			AddSegment(names[i], FindSegment(parents[i]));
		}
	}

	private Segment AddSegment(string name, Segment parent) {
		Segment segment = new Segment(name, parent, Hierarchy.Length);
		Utility.Add(ref Hierarchy, segment);
		return segment;
	}

	[Serializable]
	public class Segment {
		[SerializeField] private string Name = "Empty";
		[SerializeField] private int Index = -1;
		[SerializeField] private int Parent = -1;
		[SerializeField] private int[] Childs = new int[0];
		[SerializeField] private Matrix4x4 Transformation;

		public Segment(string name, Segment parent, int index) {
			Name = name;
			Index = index;
			if(parent != null) {
				Parent = parent.Index;
				Utility.Add(ref parent.Childs, index);
			}
			Childs = new int[0];
			Transformation = Matrix4x4.identity;
		}

		public string GetName() {
			return Name;
		}

		public int GetIndex() {
			return Index;
		}

		public Segment GetParent(Segment[] segments) {
			if(Parent == -1) {
				return null;
			} else {
				return segments[Parent];
			}
		}

		public Segment GetChild(Segment[] segments, int index) {
			return segments[Childs[index]];
		}
		
		public int GetChildCount() {
			return Childs.Length;
		}

		public void SetTransformation(Matrix4x4 transformation) {
			Transformation = transformation;
		}

		public Matrix4x4 GetTransformation() {
			return Transformation;
		}
	}

	public void Draw() {
		Draw(DrawType, BoneColor, JointColor, 1f);
	}

	public void Draw(DRAWTYPE drawType, Color boneColor, Color jointColor, float alpha) {
		UnityGL.Start();

		if(DrawSkeleton) {
			Material jointMaterial = new Material(GetMaterial(drawType));
			Material boneMaterial = new Material(GetMaterial(drawType));
			jointMaterial.color = jointColor.Transparent(alpha);
			boneMaterial.color = boneColor.Transparent(alpha);
			Action<Segment, Segment> recursion = null;
			recursion = new Action<Segment, Segment>((segment, parent) => {
				if(segment == null) {
					return;
				}
				if(parent != null) {
					UnityGL.DrawMesh(
						GetJointMesh(),
						parent.GetTransformation().GetPosition(),
						parent.GetTransformation().GetRotation(),
						5f*BoneSize*Vector3.one,
						jointMaterial
					);
					float distance = Vector3.Distance(parent.GetTransformation().GetPosition(), segment.GetTransformation().GetPosition());
					if(distance > 0f) {
						UnityGL.DrawMesh(
							GetBoneMesh(),
							parent.GetTransformation().GetPosition(),
							Quaternion.FromToRotation(parent.GetTransformation().GetForward(), segment.GetTransformation().GetPosition() - parent.GetTransformation().GetPosition()) * parent.GetTransformation().GetRotation(),
							new Vector3(4f*BoneSize, 4f*BoneSize, distance),
							boneMaterial
						);
					}
				}
				parent = segment;
				for(int i=0; i<segment.GetChildCount(); i++) {
					recursion(segment.GetChild(Hierarchy, i), parent);
				}
			});
			recursion(GetRoot(), null);
			Utility.Destroy(jointMaterial);
			Utility.Destroy(boneMaterial);
		}

		if(DrawTransforms) {
			Action<Segment> recursion = null;
			recursion = new Action<Segment>((segment) => {
				UnityGL.DrawArrow(segment.GetTransformation().GetPosition(), segment.GetTransformation().GetPosition() + 0.05f * (segment.GetTransformation().GetRotation() * Vector3.forward), 0.75f, 0.005f, 0.025f, Color.blue);
				UnityGL.DrawArrow(segment.GetTransformation().GetPosition(), segment.GetTransformation().GetPosition() + 0.05f * (segment.GetTransformation().GetRotation() * Vector3.up), 0.75f, 0.005f, 0.025f, Color.green);
				UnityGL.DrawArrow(segment.GetTransformation().GetPosition(), segment.GetTransformation().GetPosition() + 0.05f * (segment.GetTransformation().GetRotation() * Vector3.right), 0.75f, 0.005f, 0.025f, Color.red);
				for(int i=0; i<segment.GetChildCount(); i++) {
					recursion(segment.GetChild(Hierarchy, i));
				}
			});
			recursion(GetRoot());
		}

		UnityGL.Finish();
	}

	
	public void DrawSimple() {
		UnityGL.Start();
		DrawSimple(GetRoot());
		UnityGL.Finish();
	}

	private void DrawSimple(Segment segment) {
		if(segment == null) {
			return;
		}
		for(int i=0; i<segment.GetChildCount(); i++) {
			Segment child = segment.GetChild(Hierarchy, i);
			UnityGL.DrawLine(segment.GetTransformation().GetPosition(), child.GetTransformation().GetPosition(), Color.grey);
		}
		UnityGL.DrawCircle(segment.GetTransformation().GetPosition(), 0.01f, Color.black);
		for(int i=0; i<segment.GetChildCount(); i++) {
			DrawSimple(segment.GetChild(Hierarchy, i));
		}
	}

	private Mesh GetJointMesh() {
		if(JointMesh == null) {
			JointMesh = (Mesh)Resources.Load("Meshes/Joint", typeof(Mesh));
		}
		return JointMesh;
	}

	private Mesh GetBoneMesh() {
		if(BoneMesh == null) {
			BoneMesh = (Mesh)Resources.Load("Meshes/Bone", typeof(Mesh));
		}
		return BoneMesh;
	}

	private Material GetMaterial(DRAWTYPE drawType) {
		switch(drawType) {
			case DRAWTYPE.Diffuse:
			if(DiffuseMaterial == null) {
				DiffuseMaterial = (Material)Resources.Load("Materials/UnityGLDiffuse", typeof(Material));
			}
			return DiffuseMaterial;
			case DRAWTYPE.Transparent:
			if(TransparentMaterial == null) {
				TransparentMaterial = (Material)Resources.Load("Materials/UnityGLTransparent", typeof(Material));
			}
			return TransparentMaterial;
		}
		Debug.Log("Material could not be found.");
		return null;
	}

	#if UNITY_EDITOR
	public void Inspector(Transform root = null) {
		Utility.SetGUIColor(Color.grey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			if(Utility.GUIButton("Character", Utility.DarkGrey, Utility.White)) {
				Inspect = !Inspect;
			}

			if(Inspect) {
				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.LabelField("Skeleton Bones: " + Hierarchy.Length);
					DrawType = (DRAWTYPE)EditorGUILayout.EnumPopup("Draw Type", DrawType);
					BoneSize = EditorGUILayout.FloatField("Bone Size", BoneSize);
					JointColor = EditorGUILayout.ColorField("Joint Color", JointColor);
					BoneColor = EditorGUILayout.ColorField("Bone Color", BoneColor);
					DrawSkeleton = EditorGUILayout.Toggle("Draw Skeleton", DrawSkeleton);
					DrawTransforms = EditorGUILayout.Toggle("Draw Transforms", DrawTransforms);
					if(Utility.GUIButton("Clear", Utility.DarkRed, Utility.White)) {
						Utility.Clear(ref Hierarchy);
					}
					if(root == null) {
						if(Hierarchy.Length == 0) {
							EditorGUILayout.HelpBox("No skeleton available.", MessageType.Warning);
						} else {
							InspectHierarchy(GetRoot(), 0);
						}
					} else {
						InspectHierarchy(root, root, 0);
					}
				}
			}
		}
	}

	private void InspectHierarchy(Transform root, Transform transform, int indent) {
		Segment segment = FindSegment(transform.name);
		Utility.SetGUIColor(segment == null ? Utility.White : Utility.LightGrey);
		using(new EditorGUILayout.HorizontalScope ("Box")) {
			Utility.ResetGUIColor();
			EditorGUILayout.BeginHorizontal();
			for(int i=0; i<indent; i++) {
				EditorGUILayout.LabelField("|", GUILayout.Width(20f));
			}
			EditorGUILayout.LabelField("-", GUILayout.Width(20f));
			EditorGUILayout.LabelField(transform.name, GUILayout.Width(100f), GUILayout.Height(20f));
			GUILayout.FlexibleSpace();
			if(Utility.GUIButton("Bone", segment == null ? Utility.White : Utility.DarkGrey, segment == null ? Utility.DarkGrey : Utility.White)) {
				if(segment == null) {
					string[] names = GetBoneNames();
					Utility.Add(ref names, transform.name);
					BuildHierarchy(root, names);
				} else {
					string[] names = GetBoneNames();
					Utility.Remove(ref names, transform.name);
					BuildHierarchy(root, names);
				}
			}
			EditorGUILayout.EndHorizontal();
		}
		for(int i=0; i<transform.childCount; i++) {
			InspectHierarchy(root, transform.GetChild(i), indent+1);
		}
	}

	private void InspectHierarchy(Segment segment, int indent) {
		Utility.SetGUIColor(Utility.LightGrey);
		using(new EditorGUILayout.HorizontalScope ("Box")) {
			Utility.ResetGUIColor();
			EditorGUILayout.BeginHorizontal();
			for(int i=0; i<indent; i++) {
				EditorGUILayout.LabelField("|", GUILayout.Width(20f));
			}
			EditorGUILayout.LabelField("-", GUILayout.Width(20f));
			EditorGUILayout.LabelField(segment.GetName(), GUILayout.Width(100f), GUILayout.Height(20f));
			EditorGUILayout.EndHorizontal();
		}
		for(int i=0; i<segment.GetChildCount(); i++) {
			InspectHierarchy(segment.GetChild(Hierarchy, i), indent+1);
		}
	}
	#endif

}
