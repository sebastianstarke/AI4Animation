using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class Character {

	public bool Inspect = false;

	public Transform Owner = null;

	public float Phase = 0f;

	public Joint[] Joints = new Joint[0];

	private const float JointRadius = 0.05f;
	private const float BoneStartWidth = 1f/30f;
	private const float BoneEndWidth = 0.01f;

	public Character(Transform owner) {
		Owner = owner;
	}

	public void ForwardKinematics() {
		for(int i=0; i<Joints.Length; i++) {
			if(Joints[i].Transform != null) {
				Joints[i].Transform.position = Joints[i].GetPosition();
				if(Joints[i].Parent != null) {
					Joints[i].Visual.SetPosition(0, Joints[i].Parent.position);
					Joints[i].Visual.SetPosition(1, Joints[i].Transform.position);
				} else {
					Joints[i].Visual.SetPosition(0, Joints[i].Transform.position);
					Joints[i].Visual.SetPosition(1, Joints[i].Transform.position);
				}
			}
		}
	}

	public void AddJoint() {
		System.Array.Resize(ref Joints, Joints.Length+1);
		Joints[Joints.Length-1] = new Joint();
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

	public void CreateVisuals() {
		for(int i=0; i<Joints.Length; i++) {
			Joints[i].CreateVisual();
		}
	}

	public void RemoveVisuals() {
		for(int i=0; i<Joints.Length; i++) {
			Joints[i].RemoveVisual();
		}
	}

	[System.Serializable]
	public class Joint {
		public Transform Transform;
		public Transform Parent;
		public Transform[] Childs;

		public LineRenderer Visual;

		private Vector3 Position;
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
			Position = position;
		}

		public Vector3 GetPosition() {
			return Position;
		}

		public void SetVelocity(Vector3 velocity) {
			Velocity = velocity;
		}

		public Vector3 GetVelocity() {
			return Velocity;
		}
	}

	public void Draw() {
		for(int i=0; i<Joints.Length; i++) {
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
					//if(Utility.GUIButton("Detect Hierarchy", Color.grey, Color.white, TextAnchor.MiddleCenter)) {
					//	DetectHierarchy();
					//}
					//InspectHierarchy(Owner, 0);

					//Obsolete
					//JointSmoothing = EditorGUILayout.Slider("Joint Smoothing", JointSmoothing, 0f, 1f);
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

	/*
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
	*/
	#endif

}
