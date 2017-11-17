using UnityEngine;

//[System.Serializable]
public class Trajectory {

	public bool Inspect = false;

	public Point[] Points = new Point[0];

	private static float Width = 0.5f;

	public Trajectory(int size, int styles) {
		Inspect = false;
		Points = new Point[size];
		for(int i=0; i<Points.Length; i++) {
			Points[i] = new Point(i, styles);
			Points[i].SetPosition(Vector3.zero);
			Points[i].SetDirection(Vector3.forward);
		}
	}

	public Trajectory(int size, int styles, Vector3 seedPosition, Vector3 seedDirection) {
		Inspect = false;
		Points = new Point[size];
		for(int i=0; i<Points.Length; i++) {
			Points[i] = new Point(i, styles);
			Points[i].SetPosition(seedPosition);
			Points[i].SetDirection(seedDirection);
			Points[i].Postprocess();
		}
	}

	//[System.Serializable]
	public class Point {
		public int Index;
		public Vector3 Position;
		public Vector3 Direction;
		public Vector3 LeftSample;
		public Vector3 RightSample;
		public float Rise;
		public float[] Styles = new float[0];

		public Point(int index, int styles) {
			Index = index;
			Position = Vector3.zero;
			Direction = Vector3.forward;
			LeftSample = Vector3.zero;
			RightSample = Vector3.zero;
			Rise = 0f;
			Styles = new float[styles];
		}

		public int GetIndex() {
			return Index;
		}

		public Vector3 GetLeftSample() {
			return LeftSample;
		}

		public Vector3 GetRightSample() {
			return RightSample;
		}

		public void SetPosition(Vector3 position) {
			Position = position;
		}

		public Vector3 GetPosition() {
			return Position;
		}

		public void SetDirection(Vector3 direction) {
			Direction = direction;
		}

		public Vector3 GetDirection() {
			return Direction;
		}

		public Quaternion GetRotation() {
			return Quaternion.LookRotation(Direction, Vector3.up);
		}

		public Transformation GetTransformation() {
			return new Transformation(GetPosition(), GetRotation());
		}

		public void Postprocess() {
			LayerMask mask = LayerMask.GetMask("Ground");
			Position.y = Utility.GetHeight(Position, mask);
			Rise = Utility.GetRise(Position, mask);
			Vector3 ortho = Quaternion.Euler(0f, 90f, 0f) * Direction;
			RightSample = Position + Trajectory.Width * ortho.normalized;
			RightSample.y = Utility.GetHeight(RightSample, mask);
			LeftSample = Position - Trajectory.Width * ortho.normalized;
			LeftSample.y = Utility.GetHeight(LeftSample, mask);
		}
	}

	public void Draw(int step=1) {
		UnityGL.Start();
		//Connections
		for(int i=0; i<Points.Length-step; i+=step) {
			UnityGL.DrawLine(Points[i].GetPosition(), Points[i+step].GetPosition(), 0.01f, Utility.Black);
		}

		//Projections
		for(int i=0; i<Points.Length; i+=step) {
			Vector3 right = Points[i].GetRightSample();
			Vector3 left = Points[i].GetLeftSample();
			UnityGL.DrawCircle(right, 0.01f, Utility.Yellow);
			UnityGL.DrawCircle(left, 0.01f, Utility.Yellow);
		}

		//Directions
		Color transparentDirection = new Color(Utility.Orange.r, Utility.Orange.g, Utility.Orange.b, 0.75f);
		for(int i=0; i<Points.Length; i+=step) {
			UnityGL.DrawLine(Points[i].GetPosition(), Points[i].GetPosition() + 0.25f * Points[i].GetDirection(), 0.025f, 0f, transparentDirection);
		}

		//Rises
		Color transparentRise = new Color(Utility.Blue.r, Utility.Blue.g, Utility.Blue.b, 0.75f);
		for(int i=0; i<Points.Length; i+=step) {
			UnityGL.DrawLine(Points[i].GetPosition(), Points[i].GetPosition() + 1f * Points[i].Rise * Vector3.up, 0.025f, 0f, transparentRise);
		}

		//Positions
		for(int i=0; i<Points.Length; i+=step) {
			UnityGL.DrawCircle(Points[i].GetPosition(), 0.025f, Utility.Black);
		}
		UnityGL.Finish();
	}

}
