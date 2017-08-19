using UnityEngine;

public class Character {

	public Transform Transform;

	public Vector3 Velocity = Vector3.zero;
	
	public float Phase;
	public CharacterJoint[] Joints;

	public Vector3 PositionOffset;
	public Quaternion RotationOffset;

	public Character(Transform t, Transform root) {
		Transform = t;

		Phase = 0.0f;

		Joints = new CharacterJoint[31];
		int index = 0;
		BuildJoints(root, ref index);

		PositionOffset = Vector3.zero;
		RotationOffset = Quaternion.Euler(0f, 0f, 0f);
	}

	public void Move(Vector2 direction) {
		float acceleration = 5f;
		float damping = 2f;

		Velocity = Utility.Interpolate(Velocity, Vector3.zero, damping * Time.deltaTime);
		Velocity += acceleration * Time.deltaTime * (new Vector3(direction.x, 0f, direction.y).normalized);
		Transform.position += Velocity * Time.deltaTime;
	}

	public void Turn(float direction) {
		Transform.Rotate(0f, 100f*direction*Time.deltaTime, 0f);
	}

	private void BuildJoints(Transform t, ref int index) {
		Joints[index] = new CharacterJoint(this, t);
		index += 1;
		for(int i=0; i<t.childCount; i++) {
			BuildJoints(t.GetChild(i), ref index);
		}
	}

	public class CharacterJoint {
		public Transform Transform;
		public Character Character;

		public CharacterJoint(Character character, Transform transform) {
			Character = character;
			Transform = transform;
		}

		public void SetConfiguration(Vector3 relativePosition, Quaternion relativeRotation) {
			Transform.position = (Character.Transform.position + Character.PositionOffset) + (Character.RotationOffset * Character.Transform.rotation) * relativePosition;
			Transform.rotation = (Character.RotationOffset * Character.Transform.rotation) * relativeRotation;
		}

		public void GetConfiguration(out Vector3 relativePosition, out Quaternion relativeRotation) {
			relativePosition = Quaternion.Inverse(Character.RotationOffset * Character.Transform.rotation) * (Transform.position - (Character.Transform.position + Character.PositionOffset));
			relativeRotation = Quaternion.Inverse(Character.RotationOffset * Character.Transform.rotation) * Transform.rotation;
		}
	}

}
