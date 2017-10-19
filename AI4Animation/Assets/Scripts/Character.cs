using UnityEngine;

[System.Serializable]
public class Character {

	public bool Inspect = false;

	public float JointSmoothing = 0.5f;

	public float Phase = 0f;

	public Joint[] Joints = new Joint[0];

	public Character() {

	}

	public void ForwardKinematics() {
		for(int i=0; i<Joints.Length; i++) {
			if(Joints[i].Transform != null) {
				Joints[i].Transform.position = Joints[i].GetPosition();
				if(Joints[i].Visual != null) {
					Joints[i].Visual.SetActive(true);
					if(Joints[i].Parent != null) {
						Joints[i].Visual.GetComponent<Line>().Draw(Joints[i].Parent.position, Joints[i].Transform.position);
					}
				}
			} else {
				if(Joints[i].Visual != null) {
					Joints[i].Visual.SetActive(false);
				}
			}
		}
	}

	public void AddJoint(int index) {
		System.Array.Resize(ref Joints, Joints.Length+1);
		for(int i=Joints.Length-2; i>=index; i--) {
			Joints[i+1] = Joints[i];
		}
		Joints[index] = new Joint();
	}

	public void RemoveJoint(int index) {
		if(Joints.Length < index || Joints.Length == 0) {
			return;
		}
		for(int i=index; i<Joints.Length-1; i++) {
			Joints[i] = Joints[i+1];
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

		public GameObject Visual;

		private Vector3 Position;
		private Vector3 Velocity;

		public Joint() {
			Childs = new Transform[0];
		}

		public void CreateVisual() {
			if(Visual != null) {
				return;
			}
			Visual = GameObject.CreatePrimitive(PrimitiveType.Sphere);
			Visual.transform.SetParent(Transform);
			Visual.transform.localPosition = Vector3.zero;
			Visual.transform.localRotation = Quaternion.identity;
			Visual.transform.localScale = 0.05f * Vector3.one;
			Visual.GetComponent<MeshRenderer>().material = Resources.Load("Materials/Joint", typeof(Material)) as Material;
			
			Line line = Visual.AddComponent<Line>();
			line.SetWidth(0.005f);
			line.SetMaterial(Resources.Load("Materials/Joint", typeof(Material)) as Material);

			if(Application.isPlaying) {
				GameObject.Destroy(Visual.GetComponent<Collider>());
			} else {
				GameObject.DestroyImmediate(Visual.GetComponent<Collider>());
			}
		}

		public void RemoveVisual() {
			if(Visual == null) {
				return;
			}
			if(Application.isPlaying) {
				GameObject.Destroy(Visual);
			} else {
				GameObject.DestroyImmediate(Visual);
			}
		}

		public void SetTransform(Transform t, Character character) {
			if(t == Transform) {
				return;
			}
			Debug.Log("Setting Transform " + t);
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

		public void SetPosition(Vector3 position, Transformation relativeTo) {
			Position = relativeTo.Position + relativeTo.Rotation * position;
		}

		public Vector3 GetPosition() {
			return Position;
		}

		public Vector3 GetPosition(Transformation relativeTo) {
			return Quaternion.Inverse(relativeTo.Rotation) * (Position - relativeTo.Position);
		}

		public void SetVelocity(Vector3 velocity) {
			Velocity = velocity;
		}

		public void SetVelocity(Vector3 velocity, Transformation relativeTo) {
			Velocity = relativeTo.Rotation * velocity;
		}

		public Vector3 GetVelocity() {
			return Velocity;
		}

		public Vector3 GetVelocity(Transformation relativeTo) {
			return Quaternion.Inverse(relativeTo.Rotation) * Velocity;
		}
	}

}
