using UnityEngine;

public class Character : MonoBehaviour {

	public CharacterJoint[] Joints = new CharacterJoint[0];

	private float JointSize = 0.01f;

	public Vector3 GetRelativePosition(Vector3 position) {
		return Quaternion.Inverse(transform.rotation) * (position - transform.position);
	}

	public Vector3 GetRelativeDirection(Vector3 direction) {
		return Quaternion.Inverse(transform.rotation) * direction;
	}

	public Quaternion GetRelativeRotation(Quaternion rotation) {
		return Quaternion.Inverse(transform.rotation) * rotation;
	}

	public bool IsJoint(Transform t) {
		return System.Array.Find(Joints, x => x.Transform == t) != null;
	}

	public void AddJoint(int index) {
		System.Array.Resize(ref Joints, Joints.Length+1);
		for(int i=Joints.Length-2; i>=index; i--) {
			Joints[i+1] = Joints[i];
		}
		Joints[index] = new CharacterJoint(this);
	}

	public void RemoveJoint(int index) {
		if(Joints.Length < index) {
			return;
		}
		for(int i=index; i<Joints.Length-1; i++) {
			Joints[i] = Joints[i+1];
		}
		System.Array.Resize(ref Joints, Joints.Length-1);
	}

	void OnDrawGizmos() {
		DrawSkeleton(transform);
		DrawJoints(transform);
	}

	private void DrawSkeleton(Transform t, Transform parent = null) {
		Gizmos.color = Color.cyan;
		bool isJoint = IsJoint(t);
		if(parent != null && isJoint) {
			Gizmos.DrawLine(parent.position, t.position);
		}
		for(int i=0; i<t.childCount; i++) {
			if(isJoint) {
				DrawSkeleton(t.GetChild(i), t);
			} else {
				DrawSkeleton(t.GetChild(i), parent);
			}
		}
	}

	private void DrawJoints(Transform t) {
		Gizmos.color = Color.magenta;
		bool isJoint = IsJoint(t);
		if(isJoint) {
			Gizmos.DrawSphere(t.position, JointSize);
		}
		for(int i=0; i<t.childCount; i++) {
			DrawJoints(t.GetChild(i));
		}
	}

	[System.Serializable]
	public class CharacterJoint {
		public Character Character;
		public Transform Transform;
		public Vector3 Velocity;

		public CharacterJoint(Character c) {
			Character = c;
			Transform = null;
			Velocity = Vector3.zero;
		}

		public void SetRelativePosition(Vector3 relativePosition) {
			Transform.position = Character.transform.position + Character.transform.rotation * relativePosition;
		}
		
		public Vector3 GetRelativePosition() {
			return Quaternion.Inverse(Character.transform.rotation) * (Transform.position - Character.transform.position);
		}

		public void SetRelativeVelocity(Vector3 relativeVelocity) {
			Velocity = Character.transform.rotation * relativeVelocity;
		}

		public Vector3 GetRelativeVelocity() {
			return Quaternion.Inverse(Character.transform.rotation) * Velocity;
		}
	}

}
