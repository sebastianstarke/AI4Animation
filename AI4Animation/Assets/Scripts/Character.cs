using UnityEngine;

public class Character {

	public Transform Root;

	public Vector3 Velocity = Vector3.zero;
	
	public float Phase;
	public CharacterJoint[] Joints;

	public Character(Transform t, Transform start) {
		Root = t;
		Phase = 0.0f;
		Joints = new CharacterJoint[0];
		BuildJoints(start);
	}

	public void ForwardKinematics() {
		for(int i=0; i<Joints.Length; i++) {
			Joints[i].Apply();
		}
	}

	private void BuildJoints(Transform t) {
		System.Array.Resize(ref Joints, Joints.Length+1);
		Joints[Joints.Length-1] = new CharacterJoint(this, t);
		Joints[Joints.Length-1].Parent = FindJoint(t.parent);
		for(int i=0; i<t.childCount; i++) {
			BuildJoints(t.GetChild(i));
		}
	}

	private CharacterJoint FindJoint(Transform t) {
		return System.Array.Find(Joints, x => x.Transform == t);
	}

	public class CharacterJoint {
		public Transform Transform;
		public Character Character;
		public CharacterJoint Parent;

		public Vector3 Position;
		public Quaternion Rotation;

		public Matrix4x4 MeshTransformation;
		public Matrix4x4 RestTransformation;

		public CharacterJoint(Character character, Transform transform) {
			Character = character;
			Transform = transform;

			RestTransformation = Matrix4x4.TRS(GetRelativeToRootPosition(Transform.position), GetRelativeToRootRotation(Transform.rotation), Vector3.one);
		}

		public void SetPosition(Vector3 relativeToRoot) {
			Position = Character.Root.position + Character.Root.rotation * relativeToRoot;
		}

		public void SetRotation(Quaternion relativeToRoot) {
			Rotation = Character.Root.rotation * relativeToRoot;
		}

		public Vector3 GetRelativeToRootPosition(Vector3 position) {
			return Quaternion.Inverse(Character.Root.rotation) * (position - Character.Root.position);
		}

		public Quaternion GetRelativeToRootRotation(Quaternion rotation) {
			return Quaternion.Inverse(Character.Root.rotation) * rotation;
		}

		public void Apply() {
			MeshTransformation = Matrix4x4.TRS(Position, Rotation, Vector3.one) * RestTransformation.inverse;
			Transform.position = Utility.ExtractPosition(MeshTransformation);
			Transform.rotation = Utility.ExtractRotation(MeshTransformation);
		}
	}

}
