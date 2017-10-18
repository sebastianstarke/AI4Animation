using UnityEngine;

[System.Serializable]
public class Character {

	public bool Inspect = false;

	public Joint[] Joints = new Joint[0];

	public Character() {

	}

	public void ForwardKinematics() {
		for(int i=0; i<Joints.Length; i++) {
			Joints[i].Transform.position = Joints[i].GetPosition();
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

	[System.Serializable]
	public class Joint {
		public Transform Transform;

		private Vector3 Position;
		private Vector3 Velocity;

		public Joint() {

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
