using UnityEngine;

public class Trajectory : MonoBehaviour {

	public int Size = 120;
	public float Width = 0.3f;

	public Vector3 TargetDirection;
	public Vector3 TargetVelocity;

	public float TargetSmoothing = 0.1f;

	public Point[] Points;

	void Awake() {
		Points = new Point[Size];
		for(int i=0; i<Points.Length; i++) {
			Points[i] = new Point(
				transform,
				transform.position, 
				Quaternion.LookRotation(new Vector3(Camera.main.transform.forward.x, 0f, Camera.main.transform.forward.z).normalized, Vector3.up), 
				new Vector3(Camera.main.transform.forward.x, 0f, Camera.main.transform.forward.z).normalized
				);
		}
		TargetDirection = new Vector3(Camera.main.transform.forward.x, 0f, Camera.main.transform.forward.z).normalized;
		TargetVelocity = Vector3.zero;
	}

	public void Target(Vector3 control) {
		Vector3 direction = new Vector3(Camera.main.transform.forward.x, 0f, Camera.main.transform.forward.z).normalized;
		Vector3 velocity = Quaternion.LookRotation(direction, Vector3.up) * (2.0f * control) / 100f;

		TargetDirection = Vector3.Lerp(TargetDirection, direction, 1f-TargetSmoothing);
		TargetVelocity = Vector3.Lerp(TargetVelocity, velocity, 1f-TargetSmoothing);
	}

	public void Predict() {
		Vector3[] positions_blend = new Vector3[Size];
		positions_blend[Size/2] = Points[Size/2].Position;

		for(int i=Size/2+1; i<Size; i++) {
			
			float bias_pos = 0.75f;
			float bias_dir = 1.25f;
			
			float scale_pos = (1.0f - Mathf.Pow(1.0f - ((float)(i - Size/2) / (Size/2)), bias_pos));
			float scale_dir = (1.0f - Mathf.Pow(1.0f - ((float)(i - Size/2) / (Size/2)), bias_dir));

			positions_blend[i] = positions_blend[i-1] + Vector3.Lerp(
				Points[i].Position - Points[i-1].Position,
				TargetVelocity,
				scale_pos
				);
				
			Points[i].Direction = Vector3.Lerp(Points[i].Direction, TargetDirection, scale_dir);
		}
		
		for(int i=Size/2+1; i<Size; i++) {
			Points[i].Position = positions_blend[i];
		}

		for(int i=0; i<Size; i++) {
			Points[i].Rotation = Quaternion.LookRotation(Points[i].Direction, Vector3.up);
		}
			
		for(int i=0; i<Size; i++) {
			Points[i].Position.y = Utility.GetHeight(Points[i].Position.x, Points[i].Position.z, LayerMask.GetMask("Ground"));
		}
	}

	public void Correct(float updateX, float updateZ, float angle, float[] future) {
		// Update Past Trajectory
		for(int i=0; i<Size/2; i++) {
			Points[i].Position  = Points[i+1].Position;
			Points[i].Direction = Points[i+1].Direction;
			Points[i].Rotation = Points[i+1].Rotation;
		}

		// Update Current Trajectory
		//float stand_amount = powf(1.0f-trajectory->gait_stand[Size/2], 0.25f);
		
		Vector3 trajectory_update = (Points[Size/2].Rotation * new Vector3(updateX, updateZ));
		Points[Size/2].Position = Points[Size/2].Position + trajectory_update;
		Points[Size/2].Direction = Quaternion.AngleAxis(angle, Vector3.up) * Points[Size/2].Direction;
		Points[Size/2].Rotation = Quaternion.LookRotation(Points[Size/2].Direction, Vector3.up);
		
		// Update Future Trajectory
		for (int i = Size/2+1; i<Size; i++) {
			int w = (Size/2)/10;
			float m = Mathf.Repeat(((float)i - (Size/2)) / 10.0f, 1.0f);
			int k = i - (Size/2+1);
			Points[i].Position.x  = (1-m) * future[8*k+0] + m * future[8*k+1];
			Points[i].Position.z  = (1-m) * future[8*k+2] + m * future[8*k+3];
			Points[i].Direction.x  = (1-m) * future[8*k+4] + m * future[8*k+5];
			Points[i].Direction.z  = (1-m) * future[8*k+6] + m * future[8*k+7];
			Points[i].Position = (Points[Size/2].Rotation * Points[i].Position) + Points[Size/2].Position;
			Points[i].Direction = (Points[Size/2].Rotation * Points[i].Direction).normalized;
			Points[i].Rotation = Quaternion.LookRotation(Points[i].Direction, Vector3.up);
		}

		for(int i=0; i<Size; i++) {
			Points[i].Position.y = Utility.GetHeight(Points[i].Position.x, Points[i].Position.z, LayerMask.GetMask("Ground"));
		}
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			return;
		}

		//Gizmos.color = Color.cyan;
		//Gizmos.DrawLine(transform.position, transform.position + TargetDirection);

		//Gizmos.color = Color.green;
		//Gizmos.DrawLine(transform.position, transform.position + TargetVelocity);

		Gizmos.color = Color.cyan;
		for(int i=0; i<Points.Length-1; i++) {
			Gizmos.DrawLine(Points[i].Position, Points[i+1].Position);
		}

		/*
		Gizmos.color = Color.blue;
		for(int i=0; i<Trajectory.Points.Length; i++) {
			Vector3 center = Trajectory.Points[i].Position;
			Vector3 left = Trajectory.Points[i].ProjectLeft(Trajectory.Width/2f);
			Vector3 right = Trajectory.Points[i].ProjectRight(Trajectory.Width/2f);
			Gizmos.DrawLine(center, left);
			Gizmos.DrawLine(center, right);
			Gizmos.DrawSphere(left, 0.01f);
			Gizmos.DrawSphere(right, 0.01f);
		}
		Gizmos.color = Color.green;
		for(int i=0; i<Trajectory.Points.Length; i++) {
		//	Gizmos.DrawLine(Trajectory.Points[i].Position, Trajectory.Points[i].Position + Trajectory.Points[i].Velocity);
		}
		*/

		Gizmos.color = Color.black;
		for(int i=0; i<Points.Length; i++) {
			Gizmos.DrawSphere(Points[i].Position, 0.0025f);
		}
	}

	public class Point {
		public Transform Root;
		public Vector3 Position;
		public Quaternion Rotation;
		public Vector3 Direction;

		public Point(Transform root, Vector3 position, Quaternion rotation, Vector3 direction) {
			Root = root;
			Position = position;
			Rotation = rotation;
			Direction = direction;
		}

		public Vector3 GetRelativePosition() {
			return Quaternion.Inverse(Root.rotation) * (Position - Root.position);
		}

		public Vector3 GetRelativeDirection() {
			return Quaternion.Inverse(Root.rotation) * Direction;
		}

		public Vector3 ProjectLeft(float distance) {
			Vector3 ortho = Quaternion.Euler(0f, 90f, 0f) * Direction;
			Vector3 proj = Position - distance * ortho.normalized;
			proj.y = Utility.GetHeight(proj.x, proj.z, LayerMask.GetMask("Ground"));
			return proj;
		}

		public Vector3 ProjectRight(float distance) {
			Vector3 ortho = Quaternion.Euler(0f, 90f, 0f) * Direction;
			Vector3 proj = Position + distance * ortho.normalized;
			proj.y = Utility.GetHeight(proj.x, proj.z, LayerMask.GetMask("Ground"));
			return proj;
		}
	}
}
