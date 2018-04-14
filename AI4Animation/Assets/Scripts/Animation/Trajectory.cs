using UnityEngine;

[System.Serializable]
public class Trajectory {

	public bool Inspect = false;

	public Point[] Points = new Point[0];

	private static float Width = 0.5f;

	public Trajectory(int size, int styles) {
		Inspect = false;
		Points = new Point[size];
		for(int i=0; i<Points.Length; i++) {
			Points[i] = new Point(i, styles);
			Points[i].SetTransformation(Matrix4x4.identity);
		}
	}

	public Trajectory(int size, int styles, Vector3 seedPosition, Vector3 seedDirection) {
		Inspect = false;
		Points = new Point[size];
		for(int i=0; i<Points.Length; i++) {
			Points[i] = new Point(i, styles);
			Points[i].SetTransformation(Matrix4x4.TRS(seedPosition, Quaternion.LookRotation(seedDirection, Vector3.up), Vector3.one));
		}
	}

	public Trajectory(int size, int styles, Vector3[] positions, Vector3[] directions) {
		Inspect = false;
		Points = new Point[size];
		for(int i=0; i<Points.Length; i++) {
			Points[i] = new Point(i, styles);
			Points[i].SetTransformation(Matrix4x4.TRS(positions[i], Quaternion.LookRotation(directions[i], Vector3.up), Vector3.one));
		}
	}

	public Point GetFirst() {
		return Points[0];
	}

	public Point GetLast() {
		return Points[Points.Length-1];
	}

	public float GetLength() {
		float length = 0f;
		for(int i=1; i<Points.Length; i++) {
			length += Vector3.Distance(Points[i-1].GetPosition(), Points[i].GetPosition());
		}
		return length;
	}

	public void Postprocess() {
		for(int i=0; i<Points.Length; i++) {
			Points[i].Postprocess();
		}
	}

	[System.Serializable]
	public class Point {
		[SerializeField] private int Index;
		[SerializeField] private Matrix4x4 Transformation;
		[SerializeField] private Vector3 Velocity;
		[SerializeField] private Vector3 LeftSample;
		[SerializeField] private Vector3 RightSample;
		[SerializeField] private float Slope;
		public float[] Styles = new float[0];

		public Point(int index, int styles) {
			Index = index;
			Transformation = Matrix4x4.identity;
			Velocity = Vector3.zero;
			LeftSample = Vector3.zero;
			RightSample = Vector3.zero;
			Slope = 0f;
			Styles = new float[styles];
		}

		public void SetIndex(int index) {
			Index = index;
		}

		public int GetIndex() {
			return Index;
		}

		public void SetTransformation(Matrix4x4 matrix) {
			Transformation = matrix;
		}

		public Matrix4x4 GetTransformation() {
			return Transformation;
		}

		public void SetPosition(Vector3 position) {
			Matrix4x4Extensions.SetPosition(ref Transformation, position);
		}

		public Vector3 GetPosition() {
			return Transformation.GetPosition();
		}

		public void SetRotation(Quaternion rotation) {
			Matrix4x4Extensions.SetRotation(ref Transformation, rotation);
		}

		public Quaternion GetRotation() {
			return Transformation.GetRotation();
		}

		public void SetDirection(Vector3 direction) {
			SetRotation(Quaternion.LookRotation(direction == Vector3.zero ? Vector3.forward : direction, Vector3.up));
		}

		public Vector3 GetDirection() {
			return Transformation.GetForward();
		}

		public void SetVelocity(Vector3 velocity) {
			Velocity = velocity;
		}

		public Vector3 GetVelocity() {
			return Velocity;
		}

		public void SetLeftsample(Vector3 position) {
			LeftSample = position;
		}

		public Vector3 GetLeftSample() {
			return LeftSample;
		}

		public void SetRightSample(Vector3 position) {
			RightSample = position;
		}

		public Vector3 GetRightSample() {
			return RightSample;
		}

		public void SetSlope(float slope) {
			Slope = slope;
		}

		public float GetSlope() {
			return Slope;
		}

		public void Postprocess() {
			LayerMask mask = LayerMask.GetMask("Ground");
			Vector3 position = Transformation.GetPosition();
			Vector3 direction = Transformation.GetForward();

			position.y = Utility.GetHeight(Transformation.GetPosition(), mask);
			SetPosition(position);

			Slope = Utility.GetSlope(position, mask);

			Vector3 ortho = Quaternion.Euler(0f, 90f, 0f) * direction;
			RightSample = position + Trajectory.Width * ortho.normalized;
			RightSample.y = Utility.GetHeight(RightSample, mask);
			LeftSample = position - Trajectory.Width * ortho.normalized;
			LeftSample.y = Utility.GetHeight(LeftSample, mask);
		}
	}

	public void Draw(int step=1) {
		UltiDraw.Begin();
		//Connections
		for(int i=0; i<Points.Length-step; i+=step) {
			UltiDraw.DrawLine(Points[i].GetPosition(), Points[i+step].GetPosition(), 0.01f, UltiDraw.Black);
		}

		//Projections
		//for(int i=0; i<Points.Length; i+=step) {
		//	Vector3 right = Points[i].GetRightSample();
		//	Vector3 left = Points[i].GetLeftSample();
		//	UltiDraw.DrawCircle(right, 0.01f, UltiDraw.Yellow);
		//	UltiDraw.DrawCircle(left, 0.01f, UltiDraw.Yellow);
		//}

		//Directions
		for(int i=0; i<Points.Length; i+=step) {
			Vector3 start = Points[i].GetPosition();
			Vector3 end = Points[i].GetPosition() + 0.25f * Points[i].GetDirection();
			end = Utility.ProjectGround(end, LayerMask.GetMask("Ground"));
			UltiDraw.DrawLine(start, end, 0.025f, 0f, UltiDraw.Orange.Transparent(0.75f));
		}

		//Velocities
		for(int i=0; i<Points.Length; i+=step) {
			Vector3 start = Points[i].GetPosition();
			Vector3 end = Points[i].GetPosition() + Points[i].GetVelocity();
			end = Utility.ProjectGround(end, LayerMask.GetMask("Ground"));
			UltiDraw.DrawLine(start, end, 0.025f, 0f, UltiDraw.DarkGreen.Transparent(0.5f));
		}

		//Slopes
		//for(int i=0; i<Points.Length; i+=step) {
		//	UltiDraw.DrawLine(Points[i].GetPosition(), Points[i].GetPosition() + 1f * Points[i].GetSlope() * Vector3.up, 0.025f, 0f, UltiDraw.Blue.Transparent(0.75f));
		//}

		//Positions
		for(int i=0; i<Points.Length; i+=step) {
			UltiDraw.DrawCircle(Points[i].GetPosition(), 0.025f, UltiDraw.Black);
		}
		UltiDraw.End();
	}

}
