using UnityEngine;

[System.Serializable]
public class Trajectory {

	public bool Inspect = false;

	public int Size = 120;
	public float Width = 0.5f;

	public float Smoothing = 0.1f;

	public Vector3 TargetDirection;
	public Vector3 TargetVelocity;

	public Point[] Points;

	public void Initialise(Vector3 position, Vector3 direction) {
		TargetDirection = direction;
		TargetVelocity = Vector3.zero;
		Points = new Point[Size];
		for(int i=0; i<Size; i++) {
			Points[i] = new Point(position, direction);
		}
	}

	public void UpdateTarget(Vector3 move, float turn) {
		TargetDirection = Quaternion.AngleAxis(turn*120f*Time.deltaTime, Vector3.up) * TargetDirection;
		TargetVelocity = Vector3.Lerp(TargetVelocity, (Quaternion.LookRotation(TargetDirection, Vector3.up) * move).normalized, Smoothing);
	}

	public Point GetCurrent() {
		return Points[Size/2];
	}

	public Point GetPrevious() {
		return Points[Size/2-1];
	}

	public class Point {
		private Vector3 Position;
		private Vector3 Direction;

		public float Stand, Walk, Jog, Crouch, Jump, Bump;

		public Point(Vector3 position, Vector3 direction) {
			SetPosition(position);
			SetDirection(direction);

			Stand = 0f;
			Walk = 0f;
			Jog = 0f;
			Crouch = 0f;
			Jump = 0f;
			Bump = 0f;
		}

		public void SetPosition(Vector3 position) {
			Position = position;
			Position.y = Utility.GetHeight(Position.x, Position.z, LayerMask.GetMask("Ground"));
		}

		public void SetPosition(Vector3 position, Transformation relativeTo) {
			Position = relativeTo.Position + relativeTo.Rotation * position;
			Position.y = Utility.GetHeight(Position.x, Position.z, LayerMask.GetMask("Ground"));
		}

		public Vector3 GetPosition() {
			return Position;
		}

		public Vector3 GetPosition(Transformation relativeTo) {
			return Quaternion.Inverse(relativeTo.Rotation) * (Position - relativeTo.Position);
		}

		public void SetDirection(Vector3 direction) {
			Direction = direction;
		}

		public void SetDirection(Vector3 direction, Transformation relativeTo) {
			Direction = relativeTo.Rotation * direction;
		}

		public Vector3 GetDirection() {
			return Direction;
		}

		public Vector3 GetDirection(Transformation relativeTo) {
			return Quaternion.Inverse(relativeTo.Rotation) * Direction;
		}

		public float GetHeight() {
			return Position.y;
		}

		public float GetHeight(Transformation relativeTo) {
			return Position.y - relativeTo.Position.y;
		}

		public Quaternion GetRotation() {
			return Quaternion.LookRotation(Direction, Vector3.up);
		}

		public Quaternion GetRotation(Transformation relativeTo) {
			return Quaternion.LookRotation(Quaternion.Inverse(relativeTo.Rotation) * Direction, Vector3.up);
		}

		public Vector3 Project(float distance) {
			Vector3 ortho = Quaternion.Euler(0f, 90f, 0f) * Direction;
			Vector3 proj = Position + distance * ortho.normalized;
			proj.y = Utility.GetHeight(proj.x, proj.z, LayerMask.GetMask("Ground"));
			return proj;
		}
	}

}
