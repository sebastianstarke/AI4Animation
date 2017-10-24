using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class Character {

	public bool Inspect = false;

	public Transform Owner = null;

	public float Phase = 0f;

	public Bone[] Bones = new Bone[0];

	public Joint[] Joints = new Joint[0];

	private const float JointRadius = 0.05f;
	private const float BoneStartWidth = 1f/30f;
	private const float BoneEndWidth = 0.01f;

	public Character(Transform owner) {
		Owner = owner;
		DetectHierarchy();
	}

	public void DetectHierarchy() {
		for(int i=0; i<Bones.Length; i++) {
			if(Bones[i].Transform == null) {
				for(int j=i; j<Bones.Length-1; j++) {
					Bones[j] = Bones[j+1];
				}
				System.Array.Resize(ref Bones, Bones.Length-1);
				i--;
			}
		}
		DetectHierarchy(Owner, 0);
	}

	private void DetectHierarchy(Transform transform, int depth) {
		Bone bone = FindBone(transform);
		if(bone == null) {
			System.Array.Resize(ref Bones, Bones.Length+1);
			Bones[Bones.Length-1] = new Bone(transform);
		}
		for(int i=0; i<transform.childCount; i++) {
			DetectHierarchy(transform.GetChild(i), depth+1);
		}
	}

	public Bone FindBone(Transform transform) {
		return System.Array.Find(Bones, x => x.Transform == transform);
	}

	public Joint AddJoint() {
		System.Array.Resize(ref Joints, Joints.Length+1);
		Joints[Joints.Length-1] = new Joint();
		return Joints[Joints.Length-1];
	}

	public void RemoveJoint() {
		if(Joints.Length == 0) {
			return;
		}
		System.Array.Resize(ref Joints, Joints.Length-1);
	}

	public Joint FindJoint(Transform t) {
		return System.Array.Find(Joints, x => x.Transform == t);
	}

	public int FindIndex(Transform t) {
		return System.Array.FindIndex(Joints, x => x.Transform == t);
	}

	[System.Serializable]
	public class Bone {
		public bool Expanded = false;
		public bool Inspect = false;
		
		public Transform Transform = null;

		public Bone(Transform t) {
			Transform = t;
		}
	}

	[System.Serializable]
	public class Joint {
		public Transform Transform;
		public Transform Parent;
		public Transform[] Childs;

		public LineRenderer Visual;

		private Vector3 Velocity;

		public Joint() {

		}

		public void CreateVisual() {
			if(Visual != null) {
				return;
			}
			Visual = GameObject.CreatePrimitive(PrimitiveType.Sphere).AddComponent<LineRenderer>();
			Visual.name = "Visual";
			Visual.transform.SetParent(Transform);
			Visual.transform.localPosition = Vector3.zero;
			Visual.transform.localRotation = Quaternion.identity;
			Visual.transform.localScale = Vector3.zero;
			Visual.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.TwoSided;
			Visual.GetComponent<MeshRenderer>().material = Resources.Load("Materials/Joint", typeof(Material)) as Material;

			Visual.positionCount = 2;
			Visual.startWidth = 0f;
			Visual.endWidth = 0f;
			Visual.SetPosition(0, Visual.transform.position);
			Visual.SetPosition(1, Visual.transform.position);
			Visual.material = Resources.Load("Materials/Line", typeof(Material)) as Material;
			
			Utility.Destroy(Visual.GetComponent<Collider>());
		}

		public void RemoveVisual() {
			if(Visual == null) {
				return;
			}
			Utility.Destroy(Visual.gameObject);
		}

		public void SetTransform(Transform t, Character character) {
			if(t == Transform) {
				return;
			}
			if(t == null) {
				//Unset
				Transform = null;
				if(Parent != null) {
					character.FindJoint(Parent).UpdateChilds(character);
				}
				for(int i=0; i<Childs.Length; i++) {
					character.FindJoint(Childs[i]).UpdateParent(character);
				}
				Parent = null;
				Childs = new Transform[0];
				
				RemoveVisual();
			} else {
				//Set
				Transform = t;
				UpdateParent(character);
				UpdateChilds(character);
				if(Parent != null) {
					character.FindJoint(Parent).UpdateChilds(character);
				}
				for(int i=0; i<Childs.Length; i++) {
					character.FindJoint(Childs[i]).UpdateParent(character);
				}

				CreateVisual();
			}
		}

		private void UpdateParent(Character character) {
			Parent = null;
			if(Transform != Transform.root) {
				FindParent(Transform.parent, character);
			}
		}

		private void FindParent(Transform t, Character character) {
			Joint parentJoint = character.FindJoint(t);
			if(parentJoint != null) {
				Parent = t;
				return;
			}
			if(t != t.root) {
				FindParent(t.parent, character);
			}
		}

		private void UpdateChilds(Character character) {
			Childs = new Transform[0];
			FindChilds(Transform, character);
		}

		private void FindChilds(Transform t, Character character) {
			for(int i=0; i<t.childCount; i++) {
				Joint childJoint = character.FindJoint(t.GetChild(i));
				if(childJoint != null) {
					System.Array.Resize(ref Childs, Childs.Length+1);
					Childs[Childs.Length-1] = t.GetChild(i);
				} else {
					FindChilds(t.GetChild(i), character);
				}
			}
		}

		public void SetPosition(Vector3 position) {
			Transform[] childs = new Transform[Transform.childCount];
			for(int i=0; i<childs.Length; i++) {
				childs[i] = Transform.GetChild(i);
			}
			Transform.DetachChildren();
			Transform.position = position;
			for(int i=0; i<childs.Length; i++) {
				childs[i].SetParent(Transform);
			}

			Visual.transform.position = position;
			Visual.SetPosition(1, position);
			if(Parent == null) {
				Visual.SetPosition(0, position);
			}
			for(int i=0; i<Childs.Length; i++) {
				Childs[i].Find("Visual").GetComponent<LineRenderer>().SetPosition(0, position);
			}
		}

		public void SetPosition(Vector3 position, Transformation relativeTo) {
			SetPosition(relativeTo.Position + relativeTo.Rotation * position);
		}

		public Vector3 GetPosition() {
			return Transform.position;
		}

		public Vector3 GetPosition(Transformation relativeTo) {
			return Quaternion.Inverse(relativeTo.Rotation) * (GetPosition() - relativeTo.Position);
		}

		public void SetVelocity(Vector3 velocity) {
			Velocity = velocity;
		}

		public void SetVelocity(Vector3 velocity, Transformation relativeTo) {
			SetVelocity(relativeTo.Rotation * velocity);
		}

		public Vector3 GetVelocity() {
			return Velocity;
		}

		public Vector3 GetVelocity(Transformation relativeTo) {
			return Quaternion.Inverse(relativeTo.Rotation) * GetVelocity();
		}
	}

	public void Draw() {
		for(int i=0; i<Joints.Length; i++) {
			//Joints[i].RemoveVisual();
			//Joints[i].CreateVisual();
			if(Joints[i].Transform != null) {
				Joints[i].Visual.transform.localScale = JointRadius * Vector3.one;
				Joints[i].Visual.startWidth = BoneStartWidth;
				Joints[i].Visual.endWidth = BoneEndWidth;

				if(!Application.isPlaying) {
					if(Joints[i].Parent != null) {
						UnityGL.DrawLine(Joints[i].Parent.position, Joints[i].Transform.position, BoneStartWidth, BoneEndWidth, Color.cyan);
					}
				}

				UnityGL.DrawArrow(
					Joints[i].Transform.position,
					Joints[i].Transform.position + 10f*Joints[i].GetVelocity(),
					0.75f,
					0.0075f,
					0.05f,
					new Color(0f, 1f, 0f, 0.5f)
				);
			}
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
					if(Utility.GUIButton("Detect Hierarchy", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
						DetectHierarchy();
					}
					InspectHierarchy(Owner, 0);

					//Obsolete
					EditorGUILayout.LabelField("Joints");
					for(int i=0; i<Joints.Length; i++) {
						using(new EditorGUILayout.VerticalScope ("Box")) {
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20));
							Joints[i].SetTransform((Transform)EditorGUILayout.ObjectField(Joints[i].Transform, typeof(Transform), true), this);
							EditorGUILayout.EndHorizontal();
						}
					}
					if(GUILayout.Button("+")) {
						AddJoint();
					}
					if(GUILayout.Button("-")) {
						RemoveJoint();
					}
					//
				}
			}
		}
	}

	private void InspectHierarchy(Transform t, int indent) {
		Bone bone = FindBone(t);
		EditorGUILayout.BeginHorizontal();
		for(int i=0; i<indent; i++) {
			EditorGUILayout.LabelField("", GUILayout.Width(20));
		}
		if(bone == null) {
			EditorGUILayout.LabelField("Bone information missing.");
			EditorGUILayout.EndHorizontal();
		} else {
			if(t.childCount == 0) {
				bone.Expanded = false;
			} else {
				if(Utility.GUIButton(bone.Expanded ? "-" : "+", Color.grey, Color.white, TextAnchor.MiddleLeft, 20f, 20f)) {
					bone.Expanded = !bone.Expanded;
				}
			}

			EditorGUILayout.BeginVertical();
			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.LabelField(bone.Transform.name);
			if(Utility.GUIButton("Edit", Color.white, Color.black, TextAnchor.MiddleLeft, 40, 20)) {
				bone.Inspect = !bone.Inspect;
			}
			EditorGUILayout.EndHorizontal();
			if(bone.Inspect) {
				InspectBone(bone);
			}
			EditorGUILayout.EndVertical();
			EditorGUILayout.EndHorizontal();

			if(bone.Expanded) {
				for(int i=0; i<t.childCount; i++) {
					InspectHierarchy(t.GetChild(i), indent+1);
				}
			}
		}
	}

	private void InspectBone(Bone bone) {
		Utility.SetGUIColor(Color.grey);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			using(new EditorGUILayout.VerticalScope ("Box")) {
				EditorGUILayout.LabelField("INSPECTING");
			}
		}
	}
	#endif

}
