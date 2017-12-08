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

	public void Postprocess() {
		for(int i=0; i<Points.Length; i++) {
			Points[i].Postprocess();
		}
	}

	[System.Serializable]
	public class Point {
		[SerializeField] private int Index;
		[SerializeField] private Matrix4x4 Transformation;
		[SerializeField] private Vector3 LeftSample;
		[SerializeField] private Vector3 RightSample;
		[SerializeField] private float Rise;
		public float[] Styles = new float[0];

		public Point(int index, int styles) {
			Index = index;
			Transformation = Matrix4x4.identity;
			LeftSample = Vector3.zero;
			RightSample = Vector3.zero;
			Rise = 0f;
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
			Transformations.SetPosition(ref Transformation, position);
		}

		public Vector3 GetPosition() {
			return Transformation.GetPosition();
		}

		public void SetRotation(Quaternion rotation) {
			Transformations.SetRotation(ref Transformation, rotation);
		}

		public Quaternion GetRotation() {
			return Transformation.GetRotation();
		}

		public void SetDirection(Vector3 direction) {
			SetRotation(Quaternion.LookRotation(direction, Vector3.up));
		}

		public Vector3 GetDirection() {
			return Transformation.GetForward();
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

		public void SetRise(float rise) {
			Rise = rise;
		}

		public float GetRise() {
			return Rise;
		}

		public void Postprocess() {
			LayerMask mask = LayerMask.GetMask("Ground");
			Vector3 position = Transformation.GetPosition();
			Vector3 direction = Transformation.GetForward();

			position.y = Utility.GetHeight(Transformation.GetPosition(), mask);
			SetPosition(position);

			Rise = Utility.GetRise(position, mask);

			Vector3 ortho = Quaternion.Euler(0f, 90f, 0f) * direction;
			RightSample = position + Trajectory.Width * ortho.normalized;
			RightSample.y = Utility.GetHeight(RightSample, mask);
			LeftSample = position - Trajectory.Width * ortho.normalized;
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
			UnityGL.DrawLine(Points[i].GetPosition(), Points[i].GetPosition() + 1f * Points[i].GetRise() * Vector3.up, 0.025f, 0f, transparentRise);
		}

		//Positions
		for(int i=0; i<Points.Length; i+=step) {
			UnityGL.DrawCircle(Points[i].GetPosition(), 0.025f, Utility.Black);
		}
		UnityGL.Finish();
	}

	/*
	public void Draw(int start, int end, int step=1) {
		UnityGL.Start();
		//Connections
		for(int i=start; i<=end-step; i+=step) {
			UnityGL.DrawLine(Points[i].GetPosition(), Points[i+step].GetPosition(), 0.01f, Utility.Black);
		}

		//Projections
		for(int i=start; i<=end; i+=step) {
			Vector3 right = Points[i].GetRightSample();
			Vector3 left = Points[i].GetLeftSample();
			UnityGL.DrawCircle(right, 0.01f, Utility.Yellow);
			UnityGL.DrawCircle(left, 0.01f, Utility.Yellow);
		}

		//Directions
		Color transparentDirection = new Color(Utility.Orange.r, Utility.Orange.g, Utility.Orange.b, 0.75f);
		for(int i=start; i<=end; i+=step) {
			UnityGL.DrawLine(Points[i].GetPosition(), Points[i].GetPosition() + 0.25f * Points[i].GetDirection(), 0.025f, 0f, transparentDirection);
		}

		//Rises
		Color transparentRise = new Color(Utility.Blue.r, Utility.Blue.g, Utility.Blue.b, 0.75f);
		for(int i=start; i<=end; i+=step) {
			UnityGL.DrawLine(Points[i].GetPosition(), Points[i].GetPosition() + 1f * Points[i].Rise * Vector3.up, 0.025f, 0f, transparentRise);
		}

		//Positions
		for(int i=start; i<=end; i+=step) {
			UnityGL.DrawCircle(Points[i].GetPosition(), 0.025f, Utility.Black);
		}
		UnityGL.Finish();
	}
	*/

}
