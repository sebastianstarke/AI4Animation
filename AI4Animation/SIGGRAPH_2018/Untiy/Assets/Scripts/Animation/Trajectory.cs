using UnityEngine;
using System.Collections.Generic;

public class Trajectory {

	public bool Inspect = false;
	public Point[] Points = new Point[0];
	public string[] Styles = new string[0];

	private static float Width = 0.5f;

	public Trajectory(int size, string[] styles) {
		Inspect = false;
		Points = new Point[size];
		Styles = styles;
		for(int i=0; i<Points.Length; i++) {
			Points[i] = new Point(i, styles.Length);
			Points[i].SetTransformation(Matrix4x4.identity);
		}
	}

	public Trajectory(int size, string[] styles, Vector3 seedPosition, Vector3 seedDirection) {
		Inspect = false;
		Points = new Point[size];
		Styles = styles;
		for(int i=0; i<Points.Length; i++) {
			Points[i] = new Point(i, styles.Length);
			Points[i].SetTransformation(Matrix4x4.TRS(seedPosition, Quaternion.LookRotation(seedDirection, Vector3.up), Vector3.one));
		}
	}

	public Trajectory(int size, string[] styles, Vector3[] positions, Vector3[] directions) {
		Inspect = false;
		Points = new Point[size];
		Styles = styles;
		for(int i=0; i<Points.Length; i++) {
			Points[i] = new Point(i, styles.Length);
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

	public float GetLength(int start, int end, int step) {
		float length = 0f;
		for(int i=0; i<end-step; i+=step) {
			length += Vector3.Distance(Points[i+step].GetPosition(), Points[i].GetPosition());
		}
		return length;
	}
	
	public float GetCurvature(int start, int end, int step) {
		float curvature = 0f;
		for(int i=step; i<end-step; i+=step) {
			curvature += Vector3.SignedAngle(Points[i].GetPosition() - Points[i-step].GetPosition(), Points[i+step].GetPosition() - Points[i].GetPosition(), Vector3.up);
		}
		curvature = Mathf.Abs(curvature);
		curvature = Mathf.Clamp(curvature / 180f, 0f, 1f);
		return curvature;
	}

	public void Postprocess() {
		for(int i=0; i<Points.Length; i++) {
			Points[i].Postprocess();
		}
	}

	public class Point {
		private int Index;
		private Matrix4x4 Transformation;
		private Vector3 Velocity;
		private float Speed;
		private Vector3 LeftSample;
		private Vector3 RightSample;
		private float Slope;
		public float Phase;
		public float[] Signals = new float[0];
		public float[] Styles = new float[0];
		public float[] StyleUpdate = new float[0];

		public Point(int index, int styles) {
			Index = index;
			Transformation = Matrix4x4.identity;
			Velocity = Vector3.zero;
			LeftSample = Vector3.zero;
			RightSample = Vector3.zero;
			Slope = 0f;
			Signals = new float[styles];
			Styles = new float[styles];
			StyleUpdate = new float[styles];
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

		public void SetSpeed(float speed) {
			Speed = speed;
		}

		public float GetSpeed() {
			return Speed;
		}

		public void SetPhase(float value) {
			Phase = value;
		}

		public float GetPhase() {
			return Phase;
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

		Color[] colors = UltiDraw.GetRainbowColors(Styles.Length);

		//Connections
		for(int i=0; i<Points.Length-step; i+=step) {
			UltiDraw.DrawLine(Points[i].GetPosition(), Points[i+step].GetPosition(), 0.01f, UltiDraw.Black);
		}

		//Velocities
		for(int i=0; i<Points.Length; i+=step) {
			//Vector3 start = Points[i].GetPosition();
			//Vector3 end = Points[i].GetPosition() + Points[i].GetVelocity();
			//end = Utility.ProjectGround(end, LayerMask.GetMask("Ground"));
			//UltiDraw.DrawLine(start, end, 0.025f, 0f, UltiDraw.DarkGreen.Transparent(0.5f));
			
			/*
			float r = 0f;
			float g = 0f;
			float b = 0f;
			for(int j=0; j<Points[i].Styles.Length; j++) {
				r += Points[i].Styles[j] * colors[j].r;
				g += Points[i].Styles[j] * colors[j].g;
				b += Points[i].Styles[j] * colors[j].b;
			}
			UltiDraw.DrawLine(Points[i].GetPosition(), Points[i].GetPosition() + Points[i].GetVelocity(), 0.025f, 0f, new Color(r, g, b, 0.5f));
			*/

			//UltiDraw.DrawLine(Points[i].GetPosition(), Points[i].GetPosition() + Points[i].GetVelocity(), 0.025f, 0f, UltiDraw.DarkGreen.Transparent(0.5f));
		}

		//Directions
		for(int i=0; i<Points.Length; i+=step) {
			//Vector3 start = Points[i].GetPosition();
			//Vector3 end = Points[i].GetPosition() + 0.25f * Points[i].GetDirection();
			//end = Utility.ProjectGround(end, LayerMask.GetMask("Ground"));
			//UltiDraw.DrawLine(start, end, 0.025f, 0f, UltiDraw.Orange.Transparent(0.75f));
			UltiDraw.DrawLine(Points[i].GetPosition(), Points[i].GetPosition() + 0.25f*Points[i].GetDirection(), 0.025f, 0f, UltiDraw.Orange.Transparent(0.75f));
		}

		//Styles
		if(Styles.Length > 0) {
			for(int i=0; i<Points.Length; i+=step) {
				float r = 0f;
				float g = 0f;
				float b = 0f;
				for(int j=0; j<Points[i].Styles.Length; j++) {
					r += Points[i].Styles[j] * colors[j].r;
					g += Points[i].Styles[j] * colors[j].g;
					b += Points[i].Styles[j] * colors[j].b;
				}
				Color color = new Color(r,g,b,1f);
				UltiDraw.DrawCube(Points[i].GetPosition(), Points[i].GetRotation(), 0.05f, color);
			}
		}

		//Signals
		/*
		if(Styles.Length > 0) {
			for(int i=0; i<Points.Length; i+=step) {
				Color color = UltiDraw.Black;
				for(int j=0; j<Points[i].Signals.Length; j++) {
					if(Points[i].Signals[j]) {
						color = colors[j];
						break;
					}
				}
				UltiDraw.DrawCone(Points[i].GetPosition(), Quaternion.identity, 0.1f, 0.1f, color);
			}
		}
		*/

		/*
		//Speed
		for(int i=0; i<Points.Length; i+=step) {
			float r = 0f;
			float g = 0f;
			float b = 0f;
			for(int j=0; j<Points[i].Styles.Length; j++) {
				r += Points[i].Styles[j] * colors[j].r;
				g += Points[i].Styles[j] * colors[j].g;
				b += Points[i].Styles[j] * colors[j].b;
			}
			UltiDraw.DrawArrow(Points[i].GetPosition(), Points[i].GetPosition() + Points[i].GetSpeed() * Points[i].GetTransformation().GetForward(), 0.8f, 0.02f, 0.04f, new Color(r, g, b, 0.5f));
		}
		*/

		//Projections
		//for(int i=0; i<Points.Length; i+=step) {
		//	Vector3 right = Points[i].GetRightSample();
		//	Vector3 left = Points[i].GetLeftSample();
		//	UltiDraw.DrawCircle(right, 0.01f, UltiDraw.Yellow);
		//	UltiDraw.DrawCircle(left, 0.01f, UltiDraw.Yellow);
		//}

		//Slopes
		//for(int i=0; i<Points.Length; i+=step) {
		//	UltiDraw.DrawLine(Points[i].GetPosition(), Points[i].GetPosition() + 1f * Points[i].GetSlope() * Vector3.up, 0.025f, 0f, UltiDraw.Blue.Transparent(0.75f));
		//}

		//Positions
		for(int i=0; i<Points.Length; i+=step) {
			UltiDraw.DrawCircle(Points[i].GetPosition(), 0.025f, UltiDraw.Black);
		}

		//Phase
		for(int i=0; i<Points.Length; i+=step) {
			//UltiDraw.DrawLine(Points[i].GetPosition(), Points[i].GetPosition() + Points[i].Phase*Vector3.up, UltiDraw.IndianRed);
			UltiDraw.DrawArrow(Points[i].GetPosition(), Points[i].GetPosition() + Points[i].Phase*Vector3.up, 0.8f, 0.025f, 0.05f, UltiDraw.IndianRed.Transparent(0.5f));
			//UltiDraw.DrawSphere(Points[i].GetPosition(), Quaternion.identity, Points[i].PhaseUpdate / 10f, UltiDraw.Purple.Transparent(0.25f));
		}

		/*
		List<float[]> signal = new List<float[]>();
		for(int i=0; i<Styles.Length; i++) {
			float[] s = new float[Points.Length];
			for(int j=0; j<Points.Length; j++) {
				s[j] = Points[j].Signals[i];
			}
			signal.Add(s);
		}
		List<float[]> signalInput = new List<float[]>();
		for(int i=0; i<Styles.Length; i++) {
			float[] s = new float[Points.Length];
			for(int j=0; j<Points.Length; j++) {
				s[j] = Points[j].Signals[i] - Points[j].Styles[i];
			}
			signalInput.Add(s);
		}
		List<float[]> stateInput = new List<float[]>();
		for(int i=0; i<Styles.Length; i++) {
			float[] s = new float[Points.Length];
			for(int j=0; j<Points.Length; j++) {
				s[j] = Points[j].Styles[i];
			}
			stateInput.Add(s);
		}
		UltiDraw.DrawGUIFunctions(new Vector2(0.5f, 0.4f), new Vector2(0.75f, 0.1f), signal, 0f, 1f, UltiDraw.DarkGrey, colors);
		UltiDraw.DrawGUIFunctions(new Vector2(0.5f, 0.25f), new Vector2(0.75f, 0.1f), stateInput, 0f, 1f, UltiDraw.DarkGrey, colors);
		UltiDraw.DrawGUIFunctions(new Vector2(0.5f, 0.1f), new Vector2(0.75f, 0.1f), signalInput, -1f, 1f, UltiDraw.DarkGrey, colors);
		*/

		UltiDraw.End();
	}

}
