using UnityEngine;

public class Trajectory_BACKUP {

	public Transform Root;

	public int Length;
	public float Width;

	public Vector3 TargetPosition;
	public Vector3 TargetVelocity;
	public Vector3 TargetDirection;
	public Quaternion TargetRotation;

	public Point[] Points;

	public Trajectory_BACKUP(Transform t, int length, float width) {
		Root = t;

		Length = length;
		Width = width;

		TargetPosition = Root.position;
		TargetVelocity = Vector3.zero;
		TargetDirection = Root.forward;
		TargetRotation = Quaternion.identity;

		Points = new Point[Length];
		for(int i=0; i<Length; i++) {
			Points[i] = new Point(this);
		}
	}

	public void Predict(Vector3 direction) {
		//Update Trajectory Targets
		float acceleration = 80f;
		float damping = 20f;
		float decay = 2.5f;

		int current = Length/2;
		int last = Length-1;

		direction = new Vector3(direction.x, 0f, direction.z).normalized;
		Vector3 velocity = Utility.Interpolate(TargetVelocity, Vector3.zero, damping * Time.deltaTime) + acceleration * Time.deltaTime * direction;
		Vector3 target = TargetPosition + Time.deltaTime * TargetVelocity;
		if(!Physics.Raycast(Root.position, target-Root.position, (target-Root.position).magnitude, LayerMask.GetMask("Obstacles"))) {
			TargetPosition = target;
			TargetVelocity = velocity;
		} else {
			TargetVelocity = Vector3.zero;
		}
		
		if(direction.magnitude == 0f) {
			TargetPosition = Utility.Interpolate(TargetPosition, Root.position, decay * Time.deltaTime);
			TargetVelocity = Utility.Interpolate(TargetVelocity, Vector3.zero, decay * Time.deltaTime);
			TargetDirection = Vector3.zero;
			for(int i=current+1; i<Length; i++) {
				Points[i].Position = Utility.Interpolate(Points[i].Position, Root.position, decay * Time.deltaTime);
				Points[i].Velocity = Utility.Interpolate(Points[i].Velocity, Vector3.zero, decay * Time.deltaTime);
			}
		} else {
			TargetDirection = direction;
		}
		
		//Predict Trajectory
		//float rate = 10f * Time.deltaTime;
		float rate = 0.5f;

		Points[last].Position = TargetPosition;
		Points[last].Velocity = TargetVelocity;

		float pastDamp = 1.5f;
		float futureDamp = 1.5f;
		for(int i=Length-2; i>=0; i--) {
			float factor = (float)(i+1)/(float)Length;
			factor = 2f * factor - 1f;
			factor = 1f - Mathf.Abs(factor);
			factor = Utility.Normalise(factor, 1f/(float)Length, ((float)Length-1f)/(float)Length, 1f - 60f / Length, 1f);

			if(i < current) {
				Points[i].Position = 
					Points[i].Position + Utility.Interpolate(
						Mathf.Pow(factor, pastDamp) * (Points[i+1].Position - Points[i].Position), 
						Points[i+1].Position - Points[i].Position,
						rate
					);

				Points[i].Velocity = 
					Points[i].Velocity + Utility.Interpolate(
						Mathf.Pow(factor, pastDamp) * (Points[i+1].Velocity - Points[i].Velocity), 
						Points[i+1].Velocity - Points[i].Velocity,
						rate
					);
			} else {
				Points[i].Position = 
					Points[i].Position + Utility.Interpolate(
						Mathf.Pow(factor, futureDamp) * (Points[i+1].Position - Points[i].Position), 
						Points[i+1].Position - Points[i].Position,
						rate
					);

				Points[i].Velocity = 
					Points[i].Velocity + Utility.Interpolate(
						Mathf.Pow(factor, futureDamp) * (Points[i+1].Velocity - Points[i].Velocity), 
						Points[i+1].Velocity - Points[i].Velocity,
						rate
					);
			}
		}
	}

	public void Correct() {
		//Adjust Trajectory
		int current = Length/2;
		int last = Length-1;

		Vector3 error = (Root.position - Points[current].Position);
		for(int i=0; i<Length; i++) {
			float factor = (float)i / (float)(Length-1);
			Points[i].Position += factor * error;
			Points[i].Velocity = Points[i].Velocity.magnitude * (Points[i].Velocity + factor * error).normalized;
		}

		for(int i=0; i<Length; i++) {
			Points[i].Position.y = Utility.GetHeight(Points[i].Position.x, Points[i].Position.z, LayerMask.GetMask("Ground"));
			Vector3 start = Points[i].Position;
			Vector3 end = Points[i].Position + 0.1f * Points[i].Velocity.normalized;
			end.y = (Utility.GetHeight(end.x, end.z, LayerMask.GetMask("Ground")) - start.y) / 0.1f;
			Points[i].Velocity = Points[i].Velocity.magnitude * new Vector3(Points[i].Velocity.x, end.y, Points[i].Velocity.z).normalized;
		}

		TargetPosition = Points[last].Position;

		//Character.Phase = GetPhase();
	}

	public class Point {

		public Vector3 Position;
		public Vector3 Velocity;
		public Vector3 Direction;
		public Quaternion Rotation;
		public float Height;

		public Trajectory_BACKUP Trajectory;

		public Point(Trajectory_BACKUP trajectory) {
			Trajectory = trajectory;
			Position = Trajectory.TargetPosition;
			Velocity = Trajectory.TargetVelocity;
			Direction = Trajectory.TargetDirection;
			Rotation = Trajectory.TargetRotation;
			Height = Trajectory.Root.position.y;
		}

		public Vector3 GetRelativePosition() {
			return Quaternion.Inverse(Trajectory.Root.rotation) * (Position - Trajectory.Root.position);
		}

		public Vector3 GetRelativeDirection() {
			return Quaternion.Inverse(Trajectory.Root.rotation) * Direction;
		}

		public Vector3 ProjectLeft() {
			Vector3 ortho = Quaternion.Euler(0f, 90f, 0f) * Velocity;
			Vector3 proj = Position - Trajectory.Width * ortho.normalized;
			proj.y = Utility.GetHeight(proj.x, proj.z, LayerMask.GetMask("Ground"));
			return proj;
		}

		public Vector3 ProjectRight() {
			Vector3 ortho = Quaternion.Euler(0f, 90f, 0f) * Velocity;
			Vector3 proj = Position + Trajectory.Width * ortho.normalized;
			proj.y = Utility.GetHeight(proj.x, proj.z, LayerMask.GetMask("Ground"));
			return proj;
		}

	}

}
