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
	public bool[] Connections = new bool[0];

	public Segment[] Hierarchy = new Segment[0];
	public Bone[] Skeleton = new Bone[0];

	public float BoneSize = 0.025f;
	public DRAWTYPE DrawType = DRAWTYPE.Diffuse;
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

	public Segment GetHierarchyRoot() {
		if(Hierarchy.Length == 0) {
			//Debug.Log("Hierarchy has not been assigned a root segment.");
			return null;
		}
		return Hierarchy[0];
	}

	public Bone GetSkeletonRoot() {
		if(Skeleton.Length == 0) {
			//Debug.Log("Skeleton has not been assigned a root bone.");
			return null;
		}
		return Skeleton[0];
	}

	public Segment FindSegment(string name) {
		return Array.Find(Hierarchy, x => x.GetName() == name);
	}

	public Bone FindBone(string name) {
		return Array.Find(Skeleton, x => x.GetName() == name);
	}

	public void FetchForwardKinematics(Transform root) {
		List<Matrix4x4> transformations = new List<Matrix4x4>();
		Action<Transform> recursion = null;
		recursion = new Action<Transform>((transform) => {
			transformations.Add(Matrix4x4.TRS(transform.position, transform.rotation, Vector3.one));
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i));
			}
		});
		recursion(root);
		if(Hierarchy.Length != transformations.Count) {
			Debug.Log("Forward kinematics returned because the number of given transformations does not match the number of segments.");
			return;
		}
		for(int i=0; i<Hierarchy.Length; i++) {
			Hierarchy[i].SetTransformation(transformations[i]);
		}
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
		return recursion(root, GetHierarchyRoot());
	}

	public void BuildHierarchy(Transform root) {
		Array.Resize(ref Expanded, 0);
		Array.Resize(ref Connections, 0);

		Array.Resize(ref Hierarchy, 0);
		Array.Resize(ref Skeleton, 0);
		Action<Transform, Segment> recursion = null;
		recursion = new Action<Transform, Segment>((transform, parent) => {
			Segment segment = AddSegment(transform.name, parent);
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i), segment);
			}
		});
		recursion(root, null);
	}

	public void BuildSkeleton() {
		Array.Resize(ref Skeleton, 0);
		Action<Segment, Bone> recursion = null;
		recursion = new Action<Segment, Bone>((segment, parent) => {
			Bone bone = Connections[segment.GetIndex()] ? AddBone(segment, parent) : parent;
			for(int i=0; i<segment.GetChildCount(); i++) {
				recursion(segment.GetChild(this, i), bone);
			}
		});
		recursion(GetHierarchyRoot(), null);
	}
	
	public Segment AddSegment(string name, Segment parent) {
		if(FindSegment(name) != null) {
			Debug.Log("Segment has not been added because another segment with name " + name + " already exists.");
			return null;
		}
		Segment segment = new Segment(name, Hierarchy.Length);
		Array.Resize(ref Expanded, Expanded.Length+1);
		Expanded[Expanded.Length-1] = false;
		Array.Resize(ref Connections, Connections.Length+1);
		Connections[Connections.Length-1] = false;
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

	public Bone AddBone(Segment segment, Bone parent) {
		if(segment == null) {
			Debug.Log("Bone has not been added because given segment is null.");
			return null;
		}
		if(FindBone(segment.GetName()) != null) {
			Debug.Log("Bone has not been added because another bone with name " + segment.GetName() + " already exists.");
			return null;
		}
		Bone bone = new Bone(segment, Skeleton.Length);
		Array.Resize(ref Skeleton, Skeleton.Length+1);
		Skeleton[Skeleton.Length-1] = bone;
		if(parent != null) {
			bone.SetParent(parent);
			parent.AddChild(bone);
		}
		Connections[segment.GetIndex()] = true;
		return bone;
	}

	public void AddBone(Segment segment) {
		if(segment == null) {
			Debug.Log("Bone has not been added because given segment is null.");
			return;
		}
		if(Connections[segment.GetIndex()]) {
			Debug.Log("Bone has not been added because segment with name " + segment.GetName() + " was already specified as bone.");
			return;
		}
		Connections[segment.GetIndex()] = true;
		BuildSkeleton();
	}

	public void RemoveBone(Segment segment) {
		if(segment == null) {
			Debug.Log("Bone has not been removed because given segment is null.");
			return;
		}
		if(!Connections[segment.GetIndex()]) {
			Debug.Log("Bone has not been removed because segment with name " + segment.GetName() + " was not specified as bone.");
			return;
		}
		Connections[segment.GetIndex()] = false;
		BuildSkeleton();
	}

	[Serializable]
	public class Bone {
		[SerializeField] private Segment Segment;
		[SerializeField] private int Index = -1;
		[SerializeField] private int Parent = -1;
		[SerializeField] private int[] Childs = new int[0];

		public Bone(Segment segment, int index) {
			Segment = segment;
			Index = index;
			Parent = -1;
			Childs = new int[0];
		}

		public string GetName() {
			return Segment.GetName();
		}

		public void SetIndex(int index) {
			Index = index;
		}

		public int GetIndex() {
			return Index;
		}

		public void SetParent(Bone parent) {
			if(parent == null) {
				Parent = -1;
			} else {
				Parent = parent.Index;
			}
		}

		public Bone GetParent(Character character) {
			if(Parent == -1) {
				return null;
			} else {
				return character.Skeleton[Parent];
			}
		}

		public void AddChild(Bone child) {
			if(child != null) {
				Array.Resize(ref Childs, Childs.Length+1);
				Childs[Childs.Length-1] = child.Index;
			}
		}

		public Bone GetChild(Character character, int index) {
			return character.Skeleton[Childs[index]];
		}
		
		public Bone[] GetChilds(Character character) {
			Bone[] childs = new Bone[Childs.Length];
			for(int i=0; i<childs.Length; i++) {
				childs[i] = character.Skeleton[Childs[i]];
			}
			return childs;
		}

		public int GetChildCount() {
			return Childs.Length;
		}

		public void SetTransformation(Matrix4x4 transformation) {
			Segment.SetTransformation(transformation);
		}

		public Matrix4x4 GetTransformation() {
			return Segment.GetTransformation();
		}
	}

	[Serializable]
	public class Segment {
		[SerializeField] private string Name = "Empty";
		[SerializeField] private int Index = -1;
		[SerializeField] private int Parent = -1;
		[SerializeField] private int[] Childs = new int[0];
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

		public void SetTransformation(Matrix4x4 transformation) {
			Transformation = transformation;
		}

		public Matrix4x4 GetTransformation() {
			return Transformation;
		}
	}

	public void Draw() {
		UnityGL.Start();
		Draw(GetSkeletonRoot());
		UnityGL.Finish();
	}

	private void Draw(Bone bone) {
		if(bone == null) {
			return;
		}
		if(DrawSkeleton) {
			UnityGL.DrawMesh(
				GetJointMesh(),
				bone.GetTransformation().GetPosition(),
				bone.GetTransformation().GetRotation(),
				5f*BoneSize*Vector3.one,
				GetMaterial()
			);
			UnityGL.DrawSphere(bone.GetTransformation().GetPosition(), 0.5f*BoneSize, Utility.Mustard);
			for(int i=0; i<bone.GetChildCount(); i++) {
				Bone child = bone.GetChild(this, i);
				float distance = Vector3.Distance(bone.GetTransformation().GetPosition(), child.GetTransformation().GetPosition());
				if(distance > 0f) {
					UnityGL.DrawMesh(
						GetBoneMesh(),
						bone.GetTransformation().GetPosition(),
						Quaternion.FromToRotation(bone.GetTransformation().GetForward(), child.GetTransformation().GetPosition() - bone.GetTransformation().GetPosition()) * bone.GetTransformation().GetRotation(),
						new Vector3(4f*BoneSize, 4f*BoneSize, distance),
						GetMaterial()
					);
				}
				//UnityGL.DrawLine(segment.GetTransformation().GetPosition(), child.GetTransformation().GetPosition(), BoneSize, 0f, Color.cyan, new Color(0f, 0.5f, 0.5f, 1f));
			}	
		}
		if(DrawTransforms) {
			UnityGL.DrawArrow(bone.GetTransformation().GetPosition(), bone.GetTransformation().GetPosition() + 0.05f * (bone.GetTransformation().GetRotation() * Vector3.forward), 0.75f, 0.005f, 0.025f, Color.blue);
			UnityGL.DrawArrow(bone.GetTransformation().GetPosition(), bone.GetTransformation().GetPosition() + 0.05f * (bone.GetTransformation().GetRotation() * Vector3.up), 0.75f, 0.005f, 0.025f, Color.green);
			UnityGL.DrawArrow(bone.GetTransformation().GetPosition(), bone.GetTransformation().GetPosition() + 0.05f * (bone.GetTransformation().GetRotation() * Vector3.right), 0.75f, 0.005f, 0.025f, Color.red);
		}		
		for(int i=0; i<bone.GetChildCount(); i++) {
			Draw(bone.GetChild(this, i));
		}
	}

	public void DrawSimple() {
		UnityGL.Start();
		DrawSimple(GetSkeletonRoot());
		UnityGL.Finish();
	}

	private void DrawSimple(Bone bone) {
		if(bone == null) {
			return;
		}
		for(int i=0; i<bone.GetChildCount(); i++) {
			Bone child = bone.GetChild(this, i);
			UnityGL.DrawLine(bone.GetTransformation().GetPosition(), child.GetTransformation().GetPosition(), Color.grey);
		}
		UnityGL.DrawCircle(bone.GetTransformation().GetPosition(), 0.01f, Color.black);
		for(int i=0; i<bone.GetChildCount(); i++) {
			DrawSimple(bone.GetChild(this, i));
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
					EditorGUILayout.LabelField("Hierarchy: " + Hierarchy.Length);
					EditorGUILayout.LabelField("Skeleton: " + Skeleton.Length);
					BoneSize = EditorGUILayout.FloatField("Bone Size", BoneSize);
					DrawType = (DRAWTYPE)EditorGUILayout.EnumPopup("Draw Type", DrawType);
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
						InspectHierarchy(GetHierarchyRoot(), 0);
					} else {
						EditorGUILayout.LabelField("No hierarchy available.");
					}
				}
			}
		}
	}

	private void InspectHierarchy(Segment segment, int indent) {
		bool isBone = Connections[segment.GetIndex()];

		Utility.SetGUIColor(isBone ? Utility.LightGrey : Utility.White);
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
			if(Utility.GUIButton("Bone", isBone ? Utility.DarkGrey : Utility.White, isBone ? Utility.White : Utility.DarkGrey)) {
				if(isBone) {
					RemoveBone(segment);
				} else {
					AddBone(segment);
				}
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
