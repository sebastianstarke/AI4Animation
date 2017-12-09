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
	public bool[] Expanded = new bool[0];

	public Segment[] Hierarchy = new Segment[0];

	public float BoneSize = 0.025f;
	public DRAWTYPE DrawType = DRAWTYPE.Diffuse;
	public bool DrawHierarchy = false;
	public bool DrawSkeleton = true;
	public bool DrawTransforms = false;

	private Mesh JointMesh;
	private Mesh BoneMesh;
	private Material DiffuseMaterial;
	private Material TransparentMaterial;

	public Character() {
		Inspect = false;
		BoneSize = 0.025f;
	}

	public Segment GetRoot() {
		if(Hierarchy.Length == 0) {
			return null;
		}
		return Hierarchy[0];
	}
	
	public Segment AddSegment(string name, Segment parent) {
		if(FindSegment(name) != null) {
			Debug.Log("Segment has not been added because another segment with name " + name + " already exists.");
			return null;
		}
		Segment segment = new Segment(name, Hierarchy.Length);
		Array.Resize(ref Expanded, Expanded.Length+1);
		Expanded[Expanded.Length-1] = false;
		Array.Resize(ref Hierarchy, Hierarchy.Length+1);
		Hierarchy[Hierarchy.Length-1] = segment;
		if(parent != null) {
			segment.SetParent(parent);
			parent.AddChild(segment);
		}
		return segment;
	}

	public Segment AddSegment(string name, string parent) {
		return AddSegment(name, FindSegment(parent));
	}

	public Segment FindSegment(string name) {
		return Array.Find(Hierarchy, x => x.GetName() == name);
	}

	public void WriteTransformations(Transform root) {
		Action<Transform, Segment> recursion = null;
		recursion = new Action<Transform, Segment>((transform, segment) => {
			transform.position = segment.GetTransformation().GetPosition();
			transform.rotation = segment.GetTransformation().GetRotation();
			if(transform.childCount != segment.GetChildCount()) {
				Debug.Log("Writing transformations did not finish because the hierarchy does not match.");
			} else {
				for(int i=0; i<segment.GetChildCount(); i++) {
					recursion(transform.GetChild(i), segment.GetChild(this, i));
				}
			}
		});
		recursion(root, GetRoot());
	}

	public void FetchTransformations(Transform root) {
		Action<Transform, Segment> recursion = null;
		recursion = new Action<Transform, Segment>((transform, segment) => {
			segment.SetTransformation(Matrix4x4.TRS(transform.position, transform.rotation, Vector3.one));
			if(transform.childCount != segment.GetChildCount()) {
				Debug.Log("Fetching transformations did not finish because the hierarchy does not match.");
			} else {
				for(int i=0; i<transform.childCount; i++) {
					recursion(transform.GetChild(i), segment.GetChild(this, i));
				}
			}
		});
		recursion(root, GetRoot());
	}

	public bool RebuildRequired(Transform root) {
		Func<Transform, Segment, bool> recursion = null;
		recursion = new Func<Transform, Segment, bool>((transform, segment) => {
			if(transform.name != segment.GetName() || transform.childCount != segment.GetChildCount()) {
				return true;
			}
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i), segment.GetChild(this, i));
			}
			return false;
		});
		return recursion(root, GetRoot());
	}

	public void BuildHierarchy(Transform root) {
		Array.Resize(ref Expanded, 0);
		Array.Resize(ref Hierarchy, 0);
		Action<Transform, Segment> recursion = null;
		recursion = new Action<Transform, Segment>((transform, parent) => {
			Segment segment = AddSegment(transform.name, parent);
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i), segment);
			}
		});
		recursion(root, null);
	}

	[Serializable]
	public class Segment {
		[SerializeField] private string Name = "Empty";
		[SerializeField] private int Index = -1;
		[SerializeField] private int Parent = -1;
		[SerializeField] private int[] Childs = new int[0];
		[SerializeField] private bool Bone = true;
		[SerializeField] private Matrix4x4 Transformation;

		public Segment(string name, int index) {
			Name = name;
			Index = index;
			Parent = -1;
			Childs = new int[0];
			Transformation = Matrix4x4.identity;
		}

		public string GetName() {
			return Name;
		}

		public void SetIndex(int index) {
			Index = index;
		}

		public int GetIndex() {
			return Index;
		}

		public void SetParent(Segment parent) {
			if(parent == null) {
				Parent = -1;
			} else {
				Parent = parent.Index;
			}
		}

		public Segment GetParent(Character character) {
			if(Parent == -1) {
				return null;
			} else {
				return character.Hierarchy[Parent];
			}
		}

		public void AddChild(Segment child) {
			if(child != null) {
				Array.Resize(ref Childs, Childs.Length+1);
				Childs[Childs.Length-1] = child.Index;
			}
		}

		public Segment GetChild(Character character, int index) {
			return character.Hierarchy[Childs[index]];
		}
		
		public Segment[] GetChilds(Character character) {
			Segment[] childs = new Segment[Childs.Length];
			for(int i=0; i<childs.Length; i++) {
				childs[i] = character.Hierarchy[Childs[i]];
			}
			return childs;
		}

		public int GetChildCount() {
			return Childs.Length;
		}

		public void SetBone(bool value) {
			Bone = value;
		}

		public bool IsBone() {
			return Bone;
		}

		public void SetTransformation(Matrix4x4 transformation) {
			Transformation = transformation;
		}

		public Matrix4x4 GetTransformation() {
			return Transformation;
		}
	}

	public void Draw() {
		UnityGL.Start();

		if(DrawSkeleton) {
			Action<Segment, Segment> recursion = null;
			recursion = new Action<Segment, Segment>((segment, parent) => {
				if(segment.IsBone()) {
					if(parent != null) {
						UnityGL.DrawMesh(
							GetJointMesh(),
							parent.GetTransformation().GetPosition(),
							parent.GetTransformation().GetRotation(),
							5f*BoneSize*Vector3.one,
							GetMaterial()
						);
						UnityGL.DrawSphere(parent.GetTransformation().GetPosition(), 0.5f*BoneSize, Utility.Mustard);
						float distance = Vector3.Distance(parent.GetTransformation().GetPosition(), segment.GetTransformation().GetPosition());
						if(distance > 0f) {
							UnityGL.DrawMesh(
								GetBoneMesh(),
								parent.GetTransformation().GetPosition(),
								Quaternion.FromToRotation(parent.GetTransformation().GetForward(), segment.GetTransformation().GetPosition() - parent.GetTransformation().GetPosition()) * parent.GetTransformation().GetRotation(),
								new Vector3(4f*BoneSize, 4f*BoneSize, distance),
								GetMaterial()
							);
						}
					}
					parent = segment;
				}
				for(int i=0; i<segment.GetChildCount(); i++) {
					recursion(segment.GetChild(this, i), parent);
				}
			});
			recursion(GetRoot(), null);
		}

		if(DrawHierarchy) {
			Action<Segment> recursion = null;
			recursion = new Action<Segment>((segment) => {
				for(int i=0; i<segment.GetChildCount(); i++) {
					UnityGL.DrawLine(segment.GetTransformation().GetPosition(), segment.GetChild(this, i).GetTransformation().GetPosition(), 0.25f*BoneSize, 0.25f*BoneSize, Color.cyan, new Color(0f, 0.5f, 0.5f, 1f));
					recursion(segment.GetChild(this, i));
				}
			});
			recursion(GetRoot());
		}

		if(DrawTransforms) {
			Action<Segment> recursion = null;
			recursion = new Action<Segment>((segment) => {
				if(segment.IsBone()) {
					UnityGL.DrawArrow(segment.GetTransformation().GetPosition(), segment.GetTransformation().GetPosition() + 0.05f * (segment.GetTransformation().GetRotation() * Vector3.forward), 0.75f, 0.005f, 0.025f, Color.blue);
					UnityGL.DrawArrow(segment.GetTransformation().GetPosition(), segment.GetTransformation().GetPosition() + 0.05f * (segment.GetTransformation().GetRotation() * Vector3.up), 0.75f, 0.005f, 0.025f, Color.green);
					UnityGL.DrawArrow(segment.GetTransformation().GetPosition(), segment.GetTransformation().GetPosition() + 0.05f * (segment.GetTransformation().GetRotation() * Vector3.right), 0.75f, 0.005f, 0.025f, Color.red);
				}
				for(int i=0; i<segment.GetChildCount(); i++) {
					recursion(segment.GetChild(this, i));
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
			Segment child = segment.GetChild(this, i);
			UnityGL.DrawLine(segment.GetTransformation().GetPosition(), child.GetTransformation().GetPosition(), Color.grey);
		}
		UnityGL.DrawCircle(segment.GetTransformation().GetPosition(), 0.01f, Color.black);
		for(int i=0; i<segment.GetChildCount(); i++) {
			DrawSimple(segment.GetChild(this, i));
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

	private Material GetMaterial() {
		switch(DrawType) {
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
	public void Inspector() {
		Utility.SetGUIColor(Color.grey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			if(Utility.GUIButton("Character", Utility.DarkGrey, Utility.White)) {
				Inspect = !Inspect;
			}

			if(Inspect) {
				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.LabelField("Hierarchy Nodes: " + Hierarchy.Length);
					BoneSize = EditorGUILayout.FloatField("Bone Size", BoneSize);
					DrawType = (DRAWTYPE)EditorGUILayout.EnumPopup("Draw Type", DrawType);
					DrawHierarchy = EditorGUILayout.Toggle("Draw Hierarchy", DrawHierarchy);
					DrawSkeleton = EditorGUILayout.Toggle("Draw Skeleton", DrawSkeleton);
					DrawTransforms = EditorGUILayout.Toggle("Draw Transforms", DrawTransforms);
					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Expand All", Color.grey, Color.white)) {
						for(int i=0; i<Hierarchy.Length; i++) {
							Expanded[i] = true;
						}
					}
					if(Utility.GUIButton("Collapse All", Color.grey, Color.white)) {
						for(int i=0; i<Hierarchy.Length; i++) {
							Expanded[i] = false;
						}
					}
					EditorGUILayout.EndHorizontal();
					if(Hierarchy.Length > 0) {
						InspectHierarchy(GetRoot(), 0);
					} else {
						EditorGUILayout.LabelField("No hierarchy available.");
					}
				}
			}
		}
	}

	private void InspectHierarchy(Segment segment, int indent) {
		Utility.SetGUIColor(segment.IsBone() ? Utility.LightGrey : Utility.White);
		using(new EditorGUILayout.HorizontalScope ("Box")) {
			Utility.ResetGUIColor();

			EditorGUILayout.BeginHorizontal();

			EditorGUILayout.LabelField("", GUILayout.Width(indent*20f));
			if(segment.GetChildCount() > 0) {
				if(Utility.GUIButton(Expanded[segment.GetIndex()] ? "-" : "+", Color.grey, Color.white, 20f, 20f)) {
					Expanded[segment.GetIndex()] = !Expanded[segment.GetIndex()];
				}
			} else {
				Expanded[segment.GetIndex()] = false;
			}

			EditorGUILayout.LabelField(segment.GetName(), GUILayout.Width(100f), GUILayout.Height(20f));
			GUILayout.FlexibleSpace();
			if(Utility.GUIButton("Bone", segment.IsBone() ? Utility.DarkGrey : Utility.White, segment.IsBone() ? Utility.White : Utility.DarkGrey)) {
				segment.SetBone(!segment.IsBone());
			}

			EditorGUILayout.EndHorizontal();
		}

		if(Expanded[segment.GetIndex()]) {
			for(int i=0; i<segment.GetChildCount(); i++) {
				Segment child = segment.GetChild(this, i);
				if(child != null) {
					InspectHierarchy(segment.GetChild(this, i), indent+1);
				}
			}
		}
	}
	#endif

}
