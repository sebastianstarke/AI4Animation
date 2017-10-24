using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class Trajectory {

	public bool Inspect = false;
	
	public Transform Owner = null;

	public float Width = 0.5f;

	public float TargetSmoothing = 0.25f;
	public float GaitSmoothing = 0.25f;
	public float CorrectionSmoothing = 0.5f;

	public Vector3 TargetDirection;
	public Vector3 TargetVelocity;

	public Point[] Points;

	private const int PastPoints = 6;
	private const int FuturePoints = 5;
	private const int Density = 10;

	private Color PointColor {get {return Color.black;}}
	private Color ConnectionColor {get {return Color.black;}}
	private Color HeightColor {get {return Color.yellow;}}
	private Color RiseColor {get {return new Color(0f, 0f, 1f, 0.75f);}}
	private Color DirectionColor {get {return new Color(1f, 0.5f, 0f, 0.75f);}}
	private Color TargetDirectionColor {get {return new Color(1f, 0f, 0f, 0.75f);}}
	private Color TargetVelocityColor {get {return new Color(0f, 1f, 0f, 0.75f);}}

	public Trajectory(Transform owner) {
		Owner = owner;
		Initialise();
	}

	public void Initialise() {
		TargetDirection = Owner.forward;
		TargetVelocity = Vector3.zero;
		Points = new Point[GetPointCount()];
		for(int i=0; i<GetPointCount(); i++) {
			Points[i] = new Point(Owner.position, Owner.forward);
		}
	}

	public void UpdateTarget(Vector3 move, float turn) {
		//TargetDirection = Quaternion.AngleAxis(turn*120f*Time.deltaTime, Vector3.up) * TargetDirection;
		TargetDirection = Vector3.Lerp(TargetDirection, Quaternion.AngleAxis(turn*60f, Vector3.up) * GetRoot().GetDirection(), TargetSmoothing);
		TargetVelocity = Vector3.Lerp(TargetVelocity, (Quaternion.LookRotation(TargetDirection, Vector3.up) * move).normalized, TargetSmoothing);
	}

	/*
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
	*/

	public int GetPastPoints() {
		return PastPoints;
	}

	public int GetFuturePoints() {
		return FuturePoints;
	}

	public int GetDensity() {
		return Density;
	}

	public Point GetSample(int index) {
		return Points[index*Density];
	}

	public int GetSampleCount() {
		return PastPoints + FuturePoints + 1;
	}

	public int GetPointCount() {
		return Density*(PastPoints + FuturePoints) + 1;
	}

	public int GetRootSampleIndex() {
		return PastPoints;
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

	[System.Serializable]
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
				Jump = Utility.GetRise(Position.x, Position.z, LayerMask.GetMask("Ground"));
			}
		}

		public void SetPosition(Vector3 position, Transformation relativeTo, bool recalculateHeight = true) {
			Position = relativeTo.Position + relativeTo.Rotation * position;
			if(recalculateHeight) {
				Position.y = Utility.GetHeight(Position.x, Position.z, LayerMask.GetMask("Ground"));
				Jump = Utility.GetRise(Position.x, Position.z, LayerMask.GetMask("Ground"));
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

	public void Draw() {
		int step = Density;

		//Connections
		for(int i=0; i<GetPointCount()-step; i+=step) {
			UnityGL.DrawLine(Points[i].GetPosition(), Points[i+step].GetPosition(), 0.01f, ConnectionColor);
		}

		//Projections
		for(int i=0; i<GetPointCount(); i+=step) {
			Vector3 right = Points[i].Project(Width/2f);
			Vector3 left = Points[i].Project(-Width/2f);
			UnityGL.DrawCircle(right, 0.01f, HeightColor);
			UnityGL.DrawCircle(left, 0.01f, HeightColor);
		}

		//Directions
		for(int i=0; i<GetPointCount(); i+=step) {
			UnityGL.DrawLine(Points[i].GetPosition(), Points[i].GetPosition() + 0.25f * Points[i].GetDirection(), 0.025f, 0f, DirectionColor);
		}

		//Rises
		for(int i=0; i<GetPointCount(); i+=step) {
			UnityGL.DrawLine(Points[i].GetPosition(), Points[i].GetPosition() + 1f * Points[i].Jump * Vector3.up, 0.025f, 0f, RiseColor);
		}

		//Positions
		for(int i=0; i<GetPointCount(); i+=step) {
			if(i % GetDensity() == 0) {
				UnityGL.DrawCircle(Points[i].GetPosition(), 0.025f, PointColor);
			} else {
				UnityGL.DrawCircle(Points[i].GetPosition(), 0.005f, PointColor);
			}
		}

		//Target
		UnityGL.DrawLine(GetRoot().GetPosition(), GetRoot().GetPosition() + TargetDirection, 0.05f, 0f, TargetDirectionColor);
		UnityGL.DrawLine(GetRoot().GetPosition(), GetRoot().GetPosition() + TargetVelocity, 0.05f, 0f, TargetVelocityColor);
	}

	#if UNITY_EDITOR
	public void Inspector() {
		Utility.SetGUIColor(Color.grey);
		using(new GUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			if(GUILayout.Button("Trajectory")) {
				Inspect = !Inspect;
			}

			if(Inspect) {
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Width = EditorGUILayout.FloatField("Width", Width);
					TargetSmoothing = EditorGUILayout.Slider("Target Smoothing", TargetSmoothing, 0f, 1f);
					GaitSmoothing = EditorGUILayout.Slider("Gait Smoothing", GaitSmoothing, 0f, 1f);
					CorrectionSmoothing = EditorGUILayout.Slider("Correction Smoothing", CorrectionSmoothing, 0f, 1f);
				}
			}
		}
	}
	#endif

}
