using UnityEngine;

[System.Serializable]
public class Trajectory {

	public bool Inspect = false;

	public int PastPoints = 6;
	public int FuturePoints = 5;
	public int Density = 10;

	public float Width = 0.5f;

	public float TargetSmoothing = 0.25f;
	public float GaitSmoothing = 0.25f;
	public float CorrectionSmoothing = 0.5f;

	public Vector3 TargetDirection;
	public Vector3 TargetVelocity;

	public Point[] Points;

	public void Initialise(Vector3 position, Vector3 direction) {
		if(Application.isPlaying) {
			TargetDirection = direction;
			TargetVelocity = Vector3.zero;
			Points = new Point[GetPointCount()];
			for(int i=0; i<GetPointCount(); i++) {
				Points[i] = new Point(position, direction);
			}
		}
	}

	public void UpdateTarget(Vector3 move, float turn) {
		//TargetDirection = Quaternion.AngleAxis(turn*120f*Time.deltaTime, Vector3.up) * TargetDirection;
		TargetDirection = Vector3.Lerp(TargetDirection, Quaternion.AngleAxis(turn*60f, Vector3.up) * GetRoot().GetDirection(), TargetSmoothing);
		TargetVelocity = Vector3.Lerp(TargetVelocity, (Quaternion.LookRotation(TargetDirection, Vector3.up) * move).normalized, TargetSmoothing);
	}

	public void SetDensity(int density) {
		if(density == Density) {
			return;
		}
		if(Application.isPlaying) {
			Vector3 pos = GetRoot().GetPosition();
			Vector3 dir = GetRoot().GetDirection();
			Density = density;
			Initialise(pos, dir);
		} else {
			Density = density;
		}
	}

	public int GetDensity() {
		return Density;
	}

	public int GetSampleCount() {
		return PastPoints + FuturePoints + 1;
	}

	public int GetRootSampleIndex() {
		return PastPoints;
	}

	public int GetPointCount() {
		return Density*(PastPoints + FuturePoints) + 1;
	}

	public int GetRootPointIndex() {
		return Density*PastPoints;
	}

	public Point GetRoot() {
		return Points[GetRootPointIndex()];
	}

	public Point GetPrevious() {
		return Points[GetRootPointIndex()-1];
	}

	public class Point {
		private Vector3 Position;
		private Vector3 Direction;

		public float Stand, Walk, Jog, Crouch, Jump, Bump;

		public Point(Vector3 position, Vector3 direction) {
			SetPosition(position);
			SetDirection(direction);

			Stand = 1f;
			Walk = 0f;
			Jog = 0f;
			Crouch = 0f;
			Jump = 0f;
			Bump = 0f;
		}

		public void SetPosition(Vector3 position, bool recalculateHeight = true) {
			Position = position;
			if(recalculateHeight) {
				Position.y = Utility.GetHeight(Position.x, Position.z, LayerMask.GetMask("Ground"));
			}
		}

		public void SetPosition(Vector3 position, Transformation relativeTo, bool recalculateHeight = true) {
			Position = relativeTo.Position + relativeTo.Rotation * position;
			if(recalculateHeight) {
				Position.y = Utility.GetHeight(Position.x, Position.z, LayerMask.GetMask("Ground"));
			}
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
