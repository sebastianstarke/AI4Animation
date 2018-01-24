using UnityEngine;
using System.Collections;
using System.Collections.Generic;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class CatmullRomSpline : MonoBehaviour {

	public BioAnimation_APFNN Target;

	[Range(0f, 1f)] public float Correction = 1f;

	public Trajectory Trajectory;
	public Transform[] ControlPoints = new Transform[0];

	public bool DrawGUI = true;

	private bool Visualise = true;

	private List<Vector3> Positions = new List<Vector3>();
	private List<float> OffsetErrors = new List<float>();
	private List<float> AngleErrors = new List<float>();
	private List<float> Slidings = new List<float>();
	private int[] Feet = new int[4] {10, 15, 19, 23};
	private Vector3[] LastFeetPositions = new Vector3[4];

	private int Current = 0;

	void Start() {
		for(int i=0; i<Feet.Length; i++) {
			LastFeetPositions[i] = Target.Joints[i].position;
		}
	}

	void Update() {
		Target.TrajectoryCorrection = Correction;
		Target.TrajectoryControl = false;

		Trajectory.Point pivot = Trajectory.Points[Current];//GetClosestTrajectoryPoint(Target.transform.position);
		Trajectory.Point[] future = GetFutureTrajectory(pivot);

		//Target.SetTargetDirection((future[future.Length-1].GetPosition() - Target.transform.position).normalized);
		//Target.SetTargetVelocity(future[future.Length-1].GetPosition() - pivot.GetPosition());
		Target.SetTargetDirection(Vector3.zero);
		Target.SetTargetVelocity(Vector3.zero);
		
		Trajectory targetTrajectory = Target.GetTrajectory();
		for(int i=0; i<future.Length; i++) {
			float weight = (float)(i+1) / (float)future.Length;
			Trajectory.Point point = targetTrajectory.Points[60+i+1];
			point.SetPosition((1f-weight) * Target.transform.position + weight * future[i].GetPosition());
			//point.SetDirection(future[i].GetDirection());
			point.SetDirection((future[i].GetDirection() + (future[i].GetPosition()-point.GetPosition()).normalized).normalized);
			//point.SetDirection((future[i].GetDirection() + (future[future.Length-1].GetPosition() - point.GetPosition()).normalized).normalized);
			point.SetVelocity(Vector3.Distance(pivot.GetPosition(), future[future.Length-1].GetPosition()));
		}
		for(int i=60; i<targetTrajectory.Points.Length; i++) {
			targetTrajectory.Points[i].Styles[0] = 0f;
			targetTrajectory.Points[i].Styles[1] = 1f;
		}

		Current += 1;
		if(Current == Trajectory.Points.Length) {
			Current = 0;
		}

		/*
		for(int i=0; i<Feet.Length; i++) {
			float heightThreshold = i==0 || i==1 ? 0.025f : 0.05f;
			float velocityThreshold = i==0 || i==1 ? 0.015f : 0.015f;
			Vector3 oldPosition = LastFeetPositions[i];
			Vector3 newPosition = Target.Joints[Feet[i]].position;
			float velocityWeight = Utility.Exponential01((newPosition-oldPosition).magnitude / velocityThreshold);
			float heightWeight = Utility.Exponential01(newPosition.y / heightThreshold);
			float weight = 1f - Mathf.Min(velocityWeight, heightWeight);
			Vector3 slide = newPosition - oldPosition;
			slide.y = 0f;
			Slidings.Add(weight * slide.magnitude * 60f);
			LastFeetPositions[i] = newPosition;
		}

		Positions.Add(Target.transform.position);
		Vector3 p = pivot.GetPosition() - Target.transform.position;
		p.y = 0f;
		OffsetErrors.Add(p.magnitude);
		AngleErrors.Add(Mathf.Abs(Vector3.SignedAngle(pivot.GetDirection(), targetTrajectory.Points[60].GetDirection(), Vector3.up)));
		*/
	}

	void OnRenderObject() {
		if(IsInvalid()) {
			return;
		}

		if(HasChanged()) {
			for(int i=0; i<ControlPoints.Length; i++) {
				ControlPoints[i].position = Utility.ProjectGround(ControlPoints[i].position, LayerMask.GetMask("Ground"));
			}
			Create();
		}
		Trajectory.Draw(10);
		
		UnityGL.Start();
		if(Positions.Count > 1) {
			for(int i=1; i<Positions.Count; i++) {
				UnityGL.DrawLine(Positions[i-1], Positions[i], 0.025f, Utility.DarkGreen.Transparent(0.75f));
			}
		}
		UnityGL.Finish();

		if(Visualise) {
			UnityGL.Start();
			for(int i=10; i<Trajectory.Points.Length-60; i+=10) {
				UnityGL.DrawLine(Trajectory.Points[i-10].GetPosition(), Trajectory.Points[i].GetPosition(), 0.075f, Utility.Magenta);
			}
			//for(int i=0; i<ControlPoints.Length; i++) {
			//	UnityGL.DrawSphere(ControlPoints[i].position, 0.05f, Utility.Cyan.Transparent(0.75f));
			//}
			UnityGL.Finish();
		}

		/*
		Trajectory.Point pivot = GetClosestTrajectoryPoint(Target.transform.position);
		Trajectory.Point[] future = GetFutureTrajectory(pivot);
		UnityGL.Start();
		UnityGL.DrawSphere(pivot.GetPosition(), 0.05f, Utility.DarkRed.Transparent(0.75f));
		for(int i=0; i<future.Length; i++) {
			UnityGL.DrawSphere(future[i].GetPosition(), 0.025f, Utility.DarkGreen.Transparent(0.75f));
		}
		UnityGL.Finish();
		*/

		for(int i=0; i<ControlPoints.Length; i++) {
			ControlPoints[i].hasChanged = false;
		}
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			OnRenderObject();
		}
	}

	public Trajectory.Point GetClosestTrajectoryPoint(Vector3 position) {
		int index = 0;
		float distance = Vector3.Distance(position, Trajectory.Points[0].GetPosition());
		for(int i=1; i<Trajectory.Points.Length; i++) {
			float d = Vector3.Distance(position, Trajectory.Points[i].GetPosition());
			if(d <= distance) {
				index = i;
				distance = d;
			}
		}
		return Trajectory.Points[index];
	}

	public Trajectory.Point[] GetFutureTrajectory(Trajectory.Point pivot) {
		Trajectory.Point[] future = new Trajectory.Point[50];
		for(int i=0; i<50; i++) {
			int index = pivot.GetIndex() + i + 1;
			index = index % Trajectory.Points.Length;
			future[i] = Trajectory.Points[index];
		}
		return future;
	}

	private bool IsInvalid() {
		if(ControlPoints.Length < 4) {
			return true;
		}
		for(int i=0; i<ControlPoints.Length; i++) {
			if(ControlPoints[i] == null) {
				return true;
			}
		}
		return false;
	}

	private bool HasChanged() {
		for(int i=0; i<ControlPoints.Length; i++) {
			if(ControlPoints[i].hasChanged) {
				return true;
			}
		}
		return false;
	}

	private void Create() {
		Trajectory = new Trajectory(ControlPoints.Length * 60, 0);
		for(int pos=0; pos<ControlPoints.Length-1; pos++) {
			Vector3 p0 = ControlPoints[ClampListPos(pos - 1)].position;
			Vector3 p1 = ControlPoints[pos].position;
			Vector3 p2 = ControlPoints[ClampListPos(pos + 1)].position;
			Vector3 p3 = ControlPoints[ClampListPos(pos + 2)].position;
			Vector3 lastPos = p1;
			int loops = 60;
			for(int i=1; i<=loops; i++) {
				float t = i / (float)loops;
				Vector3 newPos = GetCatmullRomPosition(t, p0, p1, p2, p3);
				Trajectory.Points[pos * 60 + i -1].SetPosition(newPos);
				Trajectory.Points[pos * 60 + i -1].SetDirection(newPos-lastPos);
				lastPos = newPos;
			}
		}
		Trajectory.Postprocess();
	}

	private int ClampListPos(int pos) {
		if(pos < 0) {
			pos = ControlPoints.Length - 1;
		}
		if(pos > ControlPoints.Length) {
			pos = 1;
		} else if(pos > ControlPoints.Length - 1) {
			pos = 0;
		}
		return pos;
	}

	private Vector3 GetCatmullRomPosition(float t, Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3) {
		Vector3 a = 2f * p1;
		Vector3 b = p2 - p0;
		Vector3 c = 2f * p0 - 5f * p1 + 4f * p2 - p3;
		Vector3 d = -p0 + 3f * p1 - 3f * p2 + p3;
		Vector3 pos = 0.5f * (a + (b * t) + (c * t * t) + (d * t * t * t));
		return pos;
	}

	void OnGUI() {
		/*
		if(!DrawGUI) {
			return;
		}
		GUI.color = Color.black;
		if(GUI.Button(Utility.GetGUIRect(0f, 0, 0.05f, 0.025f), "Add CP")) {
			AddControlPoint();
		}
		if(GUI.Button(Utility.GetGUIRect(0f, 0.025f, 0.05f, 0.025f), "Remove CP")) {
			RemoveControlPoint();
		}
		if(GUI.Button(Utility.GetGUIRect(0f, 0.05f, 0.05f, 0.025f), "Visualise")) {
			Visualise = !Visualise;
		}
		*/
		/*
		if(GUI.Button(Utility.GetGUIRect(0f, 0.075f, 0.05f, 0.025f), "Reset")) {
			Positions.Clear();
			OffsetErrors.Clear();
			AngleErrors.Clear();
			Slidings.Clear();
		}
		*/
		//GUI.Label(Utility.GetGUIRect(0.025f, 0.925f, 0.2f, 0.025f), "Average Offset Error: " + Utility.ComputeMean(OffsetErrors.ToArray()));
		//GUI.Label(Utility.GetGUIRect(0.025f, 0.95f, 0.2f, 0.025f), "Average Angle Error: " + Utility.ComputeMean(AngleErrors.ToArray()));
		//GUI.Label(Utility.GetGUIRect(0.025f, 0.975f, 0.2f, 0.025f), "Average Sliding: " + Utility.ComputeMean(Slidings.ToArray()));
	}

	private Transform AddControlPoint() {
		Transform cp = GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
		cp.localScale = 0.05f * Vector3.one;
		cp.position = Vector3.zero;
		cp.rotation = Quaternion.identity;
		Utility.Destroy(cp.gameObject.GetComponent<MeshRenderer>());
		Utility.Destroy(cp.gameObject.GetComponent<MeshFilter>());
		cp.gameObject.GetComponent<Collider>().isTrigger = true;
		cp.name = "ControlPoint";
		cp.gameObject.AddComponent<MouseDrag>();
		cp.SetParent(transform);
		Utility.Add(ref ControlPoints, cp);
		//Create();
		return ControlPoints[ControlPoints.Length-1];
	}

	private void RemoveControlPoint() {
		Utility.Destroy(ControlPoints[ControlPoints.Length-1].gameObject);
		Utility.Shrink(ref ControlPoints);
		//Create();
	}
	
	#if UNITY_EDITOR
	[CustomEditor(typeof(CatmullRomSpline))]
	public class CatmullRomSpline_Editor : Editor {

		public CatmullRomSpline Target;

		void Awake() {
			Target = (CatmullRomSpline)target;
		}

		public override void OnInspectorGUI() {
			DrawDefaultInspector();
			if(GUILayout.Button("Create Circle")) {
				CreateCirlce();
			}
			if(GUILayout.Button("Create Slalom")) {
				CreateSlalom();
			}
			if(GUILayout.Button("Add Control Point")) {
				Target.AddControlPoint();
			}
			if(GUILayout.Button("Remove Control Point")) {
				Target.RemoveControlPoint();
			}
		}

		private void CreateCirlce() {
			Clear();
			int samples = 8;
			float radius = 1.5f;
			for(int i=0; i<samples; i++) {
				Transform t = Target.AddControlPoint();
				float angle = 2f * Mathf.PI * (float)i / (float)samples;
				t.position = radius * new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));
			}
		}

		private void CreateSlalom() {
			Clear();
			int samples = 10;
			float radius = 2f;
			float width = 0.5f;
			for(int i=0; i<samples; i++) {
				Transform t = Target.AddControlPoint();
				float angle = 2f * Mathf.PI * (float)i / (float)samples;
				t.position = radius * new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));
				t.position = t.position + width*t.position.normalized;
				width *= -1f;
			}
		}

		private void Clear() {
			while(Target.ControlPoints.Length > 0) {
				Target.RemoveControlPoint();
			}
		}
	}
	#endif

}