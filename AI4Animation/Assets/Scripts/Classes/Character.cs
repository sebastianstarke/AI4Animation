using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class Character {

	public bool Inspect = false;

	public Bone[] Bones = new Bone[0];

	public float BoneSize = 0.025f;

	private Material Material;

	public Character() {
		Inspect = false;
		BoneSize = 0.025f;
	}

	public void Clear() {
		System.Array.Resize(ref Bones, 0);
	}

	public Bone GetRoot() {
		if(Bones.Length == 0) {
			Debug.Log("Character has not been assigned a root bone.");
			return null;
		}
		return Bones[0];
	}

	public void FetchForwardKinematics(Transform root) {
		int index = 0;
		FetchForwardKinematics(root, ref index);
		if(index != Bones.Length) {
			Debug.Log("Forward kinematics did not finish properly because the number of transforms and bones did not match.");
		}
	}

	private void FetchForwardKinematics(Transform transform, ref int index) {
		if(index < Bones.Length) {
			Bones[index].SetPosition(transform.position);
			Bones[index].SetRotation(transform.rotation);
			index += 1;
			for(int i=0; i<transform.childCount; i++) {
				FetchForwardKinematics(transform.GetChild(i), ref index);
			}
		}
	}

	public void FetchForwardKinematics(Transformation[] transformations) {
		if(Bones.Length != transformations.Length) {
			Debug.Log("Forward kinematics returned because the number of given transformations does not match the number of bones.");
			return;
		}
		for(int i=0; i<Bones.Length; i++) {
			Bones[i].SetPosition(transformations[i].Position);
			Bones[i].SetRotation(transformations[i].Rotation);
		}
	}

	public Bone FindBone(string name) {
		return System.Array.Find(Bones, x => x.GetName() == name);
	}

	public bool RebuildRequired(Transform root) {
		bool error = false;
		RebuildRequired(root, GetRoot(), ref error);
		return error;
	}

	private void RebuildRequired(Transform transform, Bone bone, ref bool error) {
		if(error) {
			return;
		}
		if(transform.name != bone.GetName() || transform.childCount != bone.GetChildCount()) {
			error = true;
		}
		for(int i=0; i<transform.childCount; i++) {
			RebuildRequired(transform.GetChild(i), bone.GetChild(this, i), ref error);
		}
	}

	public void BuildHierarchy(Transform root) {
		System.Array.Resize(ref Bones, 0);
		BuildHierarchy(root, null);
	}

	private void BuildHierarchy(Transform transform, Bone parent) {
		Bone bone = AddBone(transform.name, parent);
		for(int i=0; i<transform.childCount; i++) {
			BuildHierarchy(transform.GetChild(i), bone);
		}
	}

	public Bone AddBone(string name, Bone parent) {
		if(FindBone(name) != null) {
			Debug.Log("Bone has not been added because another bone with name " + name + " already exists.");
			return null;
		}
		Bone bone = new Bone(name, Bones.Length);
		System.Array.Resize(ref Bones, Bones.Length+1);
		Bones[Bones.Length-1] = bone;
		if(parent != null) {
			bone.SetParent(this, parent);
			parent.AddChild(this, bone);
		}
		return bone;
	}

	public void Print() {
		Print(GetRoot());
	}

	private void Print(Bone bone) {
		string output = string.Empty;
		output += "Name: " + bone.GetName() + " ";
		output += "Parent: " + bone.GetParent(this) + " ";
		output += "Childs: ";
		for(int i=0; i<bone.GetChildCount(); i++) {
			output += bone.GetChild(this, i).GetName() + " ";
		}
		UnityEngine.Debug.Log(output);
		for(int i=0; i<bone.GetChildCount(); i++) {
			Print(bone.GetChild(this, i));
		}
	}

	[System.Serializable]
	public class Bone {
		public bool Expanded = false;
		public bool Draw = true;
		public bool Transform = false;

		[SerializeField] private string Name = "Empty";
		[SerializeField] private int Index = -1;
		[SerializeField] private int Parent = -1;
		[SerializeField] private int[] Childs = new int[0];
		[SerializeField] private Vector3 Position = Vector3.zero;	
		[SerializeField] private Quaternion Rotation = Quaternion.identity;

		public Bone(string name, int index) {
			Draw = true;
			Name = name;
			Index = index;
			Parent = -1;
			Childs = new int[0];
			Position = Vector3.zero;
			Rotation = Quaternion.identity;
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

		public int GetChildCount() {
			return Childs.Length;
		}

		public void RemoveChilds() {
			System.Array.Resize(ref Childs, 0);
		}

		public void SetPosition(Vector3 position) {
			Position = position;
		}

		public void SetRotation(Quaternion rotation) {
			Rotation = rotation;
		}

		public Vector3 GetPosition() {
			return Position;
		}

		public Quaternion GetRotation() {
			return Rotation;
		}
	}

	public void Draw() {
		UnityGL.Start();
		Draw(GetRoot());
		UnityGL.Finish();
	}

	private void Draw(Bone bone) {
		if(bone.Draw) {
			for(int i=0; i<bone.GetChildCount(); i++) {
				Bone child = bone.GetChild(this, i);
				if(child.Draw) {
					UnityGL.DrawLine(bone.GetPosition(), child.GetPosition(), BoneSize, 0f, Color.cyan, new Color(0f, 0.5f, 0.5f, 1f));
				}
			}
			UnityGL.DrawSphere(bone.GetPosition(), 0.5f*BoneSize, Color.black);
			//if(bone.Transform) {
				UnityGL.DrawArrow(bone.GetPosition(), bone.GetPosition() + 0.05f * (bone.GetRotation() * Vector3.forward), 0.75f, 0.005f, 0.025f, Color.blue);
				UnityGL.DrawArrow(bone.GetPosition(), bone.GetPosition() + 0.05f * (bone.GetRotation() * Vector3.up), 0.75f, 0.005f, 0.025f, Color.green);
				UnityGL.DrawArrow(bone.GetPosition(), bone.GetPosition() + 0.05f * (bone.GetRotation() * Vector3.right), 0.75f, 0.005f, 0.025f, Color.red);
			//}
			/*
			UnityGL.DrawMesh(
				Utility.GetPrimitiveMesh(PrimitiveType.Cube),
				bone.GetPosition(),
				bone.GetRotation(),
				BoneSize*Vector3.one,
				GetMaterial()
			);
			*/
		}
		for(int i=0; i<bone.GetChildCount(); i++) {
			Draw(bone.GetChild(this, i));
		}
	}

	public void DrawSimple() {
		UnityGL.Start();
		DrawSimple(GetRoot());
		UnityGL.Finish();
	}

	private void DrawSimple(Bone bone) {
		if(bone.Draw) {
			for(int i=0; i<bone.GetChildCount(); i++) {
				Bone child = bone.GetChild(this, i);
				if(child.Draw) {
					UnityGL.DrawLine(bone.GetPosition(), child.GetPosition(), Color.grey);
				}
			}
			UnityGL.DrawCircle(bone.GetPosition(), 0.01f, Color.black);
		}
		for(int i=0; i<bone.GetChildCount(); i++) {
			DrawSimple(bone.GetChild(this, i));
		}
	}

	private Material GetMaterial() {
		if(Material == null) {
			Material = (Material)Resources.Load("Materials/Black", typeof(Material));
		}
		return Material;
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
					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Expand All", Color.grey, Color.white)) {
						for(int i=0; i<Bones.Length; i++) {
							Bones[i].Expanded = true;
						}
					}
					if(Utility.GUIButton("Collapse All", Color.grey, Color.white)) {
						for(int i=0; i<Bones.Length; i++) {
							Bones[i].Expanded = false;
						}
					}
					EditorGUILayout.EndHorizontal();
					if(Bones.Length > 0) {
						InspectHierarchy(GetRoot(), 0);
					} else {
						EditorGUILayout.LabelField("No bones available.");
					}
				}
			}
		}
	}

	private void InspectHierarchy(Bone bone, int indent) {
		EditorGUILayout.BeginHorizontal();
		EditorGUILayout.LabelField("", GUILayout.Width(indent*20f));
		if(bone.GetChildCount() > 0) {
			if(Utility.GUIButton(bone.Expanded ? "-" : "+", Color.grey, Color.white, 20f, 20f)) {
				bone.Expanded = !bone.Expanded;
			}
		} else {
			bone.Expanded = false;
		}
		EditorGUILayout.LabelField(bone.GetName(), GUILayout.Width(100f), GUILayout.Height(20f));
		GUILayout.FlexibleSpace();
		if(Utility.GUIButton("Draw", bone.Draw ? Color.grey : Color.white, bone.Draw ? Color.white : Color.grey, 40f, 20f)) {
			bone.Draw = !bone.Draw;
		}
		if(Utility.GUIButton("Transform", bone.Transform ? Color.grey : Color.white, bone.Transform ? Color.white : Color.gray, 40f, 20f)) {
			bone.Transform = !bone.Transform;
		}
		EditorGUILayout.EndHorizontal();
		if(bone.Expanded) {
			for(int i=0; i<bone.GetChildCount(); i++) {
				Bone child = bone.GetChild(this, i);
				if(child != null) {
					InspectHierarchy(bone.GetChild(this, i), indent+1);
				}
			}
		}
	}
	#endif

}
