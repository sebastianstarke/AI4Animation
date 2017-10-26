using UnityEngine;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class Character {

	public Transform Owner = null;

	public bool Inspect = false;

	public Bone[] Bones = new Bone[0];

	public float BoneSize = 0.025f;

	public Character(Transform owner) {
		Owner = owner;
		Inspect = false;
		BuildHierarchy();
		BoneSize = 0.025f;
	}

	public void ForwardKinematics() {
		
	}

	public Bone FindBone(string name) {
		return System.Array.Find(Bones, x => x.Transform.name == name);
	}

	public Bone FindBone(Transform transform) {
		return System.Array.Find(Bones, x => x.Transform == transform);
	}

	public bool RebuildRequired() {
		for(int i=0; i<Bones.Length; i++) {
			if(Bones[i].Transform == null) {
				return true;
			}
			if(Bones[i].Transform.childCount != Bones[i].Childs.Length) {
				return true;
			} else {
				for(int j=0; j<Bones[i].Transform.childCount; j++) {
					if(Bones[i].Transform.GetChild(j) != Bones[i].GetChild(this, j).Transform) {
						return true;
					}
				}
			}
		}
		return false;
	}

	public void BuildHierarchy() {
		List<Bone> bones = new List<Bone>();
		BuildHierarchy(Owner, ref bones);
		Bones = bones.ToArray();
		for(int i=0; i<Bones.Length; i++) {
			Bones[i].SetParent(this, FindBone(Bones[i].Transform.parent));
			for(int j=0; j<Bones[i].Transform.childCount; j++) {
				Bones[i].AddChild(this, FindBone(Bones[i].Transform.GetChild(j)));
			}
		}
	}

	private void BuildHierarchy(Transform transform, ref List<Bone> bones) {
		Bone bone = FindBone(transform);
		if(bone == null) {
			bone = new Bone(transform, bones.Count);
		} else {
			bone.Index = bones.Count;
			bone.RemoveParent();
			bone.RemoveChilds();
		}
		bones.Add(bone);
		for(int i=0; i<transform.childCount; i++) {
			BuildHierarchy(transform.GetChild(i), ref bones);
		}
	}

	[System.Serializable]
	public class Bone {
		public bool Expanded = false;
		public bool Draw = true;
		
		public Transform Transform = null;	
		public int Index = -1;
		public int Parent = -1;
		public int[] Childs = new int[0];
		
		public Vector3 Velocity = Vector3.zero;

		public Bone(Transform t, int index) {
			Draw = true;
			Transform = t;
			Index = index;
			Parent = -1;
			Childs = new int[0];
			
			Velocity = Vector3.zero;
		}

		public void SetParent(Character character, Bone parent) {
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
				return character.Bones[Parent];
			}
		}

		public void RemoveParent() {
			Parent = -1;
		}

		public void AddChild(Character character, Bone child) {
			if(child != null) {
				System.Array.Resize(ref Childs, Childs.Length+1);
				Childs[Childs.Length-1] = child.Index;
			}
		}

		public Bone GetChild(Character character, int index) {
			return character.Bones[Childs[index]];
		}
		
		public Bone[] GetChilds(Character character) {
			Bone[] childs = new Bone[Childs.Length];
			for(int i=0; i<childs.Length; i++) {
				childs[i] = character.Bones[Childs[i]];
			}
			return childs;
		}

		public void RemoveChilds() {
			System.Array.Resize(ref Childs, 0);
		}

		public void SetPosition(Vector3 position) {
			Transform.OverridePosition(position);
		}

		public void SetRotation(Quaternion rotation) {
			Transform.OverrideRotation(rotation);
		}

		public Vector3 GetPosition() {
			return Transform.position;
		}

		public Quaternion GetRotation() {
			return Transform.rotation;
		}

		public void SetVelocity(Vector3 velocity) {
			Velocity = velocity;
		}

		public Vector3 GetVelocity() {
			return Velocity;
		}
	}

	public void Draw() {
		Draw(FindBone(Owner));
	}

	private void Draw(Bone bone) {
		if(bone.Transform != null && bone.Draw) {
			for(int i=0; i<bone.Childs.Length; i++) {
				Bone child = bone.GetChild(this, i);
				if(child.Transform != null && child.Draw) {
					UnityGL.DrawLine(bone.Transform.position, child.Transform.position, BoneSize, 0f, Color.cyan);
				}
			}
			UnityGL.DrawMesh(
				Utility.GetPrimitiveMesh(PrimitiveType.Sphere),
				bone.Transform.position,
				bone.Transform.rotation,
				BoneSize*Vector3.one,
				(Material)Resources.Load("Materials/Black", typeof(Material))
			);
		}
		for(int i=0; i<bone.Childs.Length; i++) {
			Draw(bone.GetChild(this, i));
		}
	}

	#if UNITY_EDITOR
	public void Inspector() {
		Utility.SetGUIColor(Color.grey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			if(GUILayout.Button("Character")) {
				Inspect = !Inspect;
			}

			if(Inspect) {
				using(new EditorGUILayout.VerticalScope ("Box")) {
					BoneSize = EditorGUILayout.FloatField("Bone Size", BoneSize);
					if(Utility.GUIButton("Build Hierarchy", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
						BuildHierarchy();
					}
					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Expand All", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
						for(int i=0; i<Bones.Length; i++) {
							Bones[i].Expanded = true;
						}
					}
					if(Utility.GUIButton("Collapse All", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
						for(int i=0; i<Bones.Length; i++) {
							Bones[i].Expanded = false;
						}
					}
					EditorGUILayout.EndHorizontal();
					if(RebuildRequired()) {
						EditorGUILayout.HelpBox("Rebuild required because hierarchy was changed externally.", MessageType.Error);
					}
					InspectHierarchy(FindBone(Owner), 0);
				}
			}
		}
	}

	private void InspectHierarchy(Bone bone, int indent) {
		EditorGUILayout.BeginHorizontal();
		EditorGUILayout.LabelField("", GUILayout.Width(indent*20f));
		if(bone.Transform == null) {
			EditorGUILayout.LabelField("Bone information missing.");
		} else {
			if(Utility.GUIButton(bone.Expanded ? "-" : "+", Color.grey, Color.white, TextAnchor.MiddleLeft, 20f, 20f)) {
				bone.Expanded = !bone.Expanded;
			}
			EditorGUILayout.LabelField(bone.Transform.name, GUILayout.Width(100f), GUILayout.Height(20f));
			GUILayout.FlexibleSpace();
			if(Utility.GUIButton("Draw", bone.Draw ? Color.grey : Color.white, bone.Draw ? Color.white : Color.grey, TextAnchor.MiddleLeft, 40f, 20f)) {
				bone.Draw = !bone.Draw;
			}
		}
		EditorGUILayout.EndHorizontal();
		if(bone.Expanded) {
			for(int i=0; i<bone.Childs.Length; i++) {
				Bone child = bone.GetChild(this, i);
				if(child != null) {
					InspectHierarchy(bone.GetChild(this, i), indent+1);
				}
			}
		}
	}
	#endif

}
