using UnityEngine;
using System.Collections;
using System.Collections.Generic;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class CatmullRomSpline : MonoBehaviour {

	public BioAnimation Target;

	public int Density = 50;

	[Range(0f, 1f)] public float Correction = 1f;

	public Trajectory Trajectory;
	public Transform[] ControlPoints = new Transform[0];

	public bool DrawGUI = true;

	private bool Visualise = true;

	private List<Vector3> Positions = new List<Vector3>();
	private List<float> OffsetErrors = new List<float>();
	private List<float> AngleErrors = new List<float>();

	private int Current = 0;

	void Update() {
		Target.TrajectoryCorrection = Correction;
		Target.TrajectoryControl = false;

		//Trajectory.Point pivot = Trajectory.Points[Current];
		Trajectory.Point pivot = GetClosestTrajectoryPoint(Target.transform.position);
		Trajectory.Point[] future = GetFutureTrajectory(pivot);
		
		Trajectory targetTrajectory = Target.GetTrajectory();
		for(int i=0; i<future.Length; i++) {
			//float weight = (float)(i+1) / (float)future.Length;
			Trajectory.Point point = targetTrajectory.Points[60+i+1];
			point.SetPosition(future[i].GetPosition());
			//point.SetDirection(future[i].GetDirection());
			if(i == future.Length-1) {
				point.SetDirection(future[i].GetDirection());
			} else {
				point.SetDirection(future[i].GetDirection());
				//point.SetDirection(future[future.Length-1].GetPosition() - point.GetPosition());
			}
			//point.SetPosition((1f-weight) * Target.transform.position + weight * future[i].GetPosition());
			//point.SetDirection((future[i].GetDirection() + (future[i].GetPosition()-point.GetPosition()).normalized).normalized);
			//point.SetDirection((future[i].GetDirection() + (future[future.Length-1].GetPosition() - point.GetPosition()).normalized).normalized);
			if(i==0) {
				point.SetVelocity((future[0].GetPosition() - pivot.GetPosition()) * 60f);
			} else {
				point.SetVelocity((future[i].GetPosition() - future[i-1].GetPosition()) * 60f);
			}
		}
		for(int i=60; i<targetTrajectory.Points.Length; i++) {
			targetTrajectory.Points[i].Styles[0] = 0f;
			targetTrajectory.Points[i].Styles[1] = 1f;
		}

		Current += 1;
		if(Current == Trajectory.Points.Length) {
			Current = 0;
		}

		Positions.Add(Target.transform.position);
		Vector3 p = pivot.GetPosition() - Target.transform.position;
		p.y = 0f;
		OffsetErrors.Add(p.magnitude);
		AngleErrors.Add(Mathf.Abs(Vector3.SignedAngle(pivot.GetDirection(), targetTrajectory.Points[60].GetDirection(), Vector3.up)));
		
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

		/*
		UltiDraw.Begin();
		if(Positions.Count > 1) {
			for(int i=1; i<Positions.Count; i++) {
				UltiDraw.DrawLine(Positions[i-1], Positions[i], 0.025f, UltiDraw.DarkGreen.Transparent(0.75f));
			}
		}
		UltiDraw.End();
		*/

		if(Visualise) {
			UltiDraw.Begin();
			for(int i=1; i<Trajectory.Points.Length; i++) {
				UltiDraw.DrawLine(Trajectory.Points[i-1].GetPosition(), Trajectory.Points[i].GetPosition(), 0.025f, UltiDraw.Brown);
			}
			for(int i=0; i<ControlPoints.Length; i++) {
				UltiDraw.DrawSphere(ControlPoints[i].position, Quaternion.identity, 0.1f, UltiDraw.Cyan.Transparent(0.75f));
			}
			UltiDraw.End();
		}

		
		Trajectory.Point pivot = GetClosestTrajectoryPoint(Target.transform.position);
		Trajectory.Point[] future = GetFutureTrajectory(pivot);
		UltiDraw.Begin();
		UltiDraw.DrawSphere(pivot.GetPosition(), Quaternion.identity, 0.05f, UltiDraw.Red.Transparent(0.75f));
		for(int i=0; i<future.Length; i++) {
			UltiDraw.DrawSphere(future[i].GetPosition(), Quaternion.identity, 0.025f, UltiDraw.Green.Transparent(0.75f));
		}
		UltiDraw.End();
		

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
		int length = ControlPoints.Length;
		Trajectory = new Trajectory(length * Density, 0);
		for(int pos=0; pos<length; pos++) {
			Vector3 p0 = ControlPoints[ClampListPos(pos - 1)].position;
			Vector3 p1 = ControlPoints[pos].position;
			Vector3 p2 = ControlPoints[ClampListPos(pos + 1)].position;
			Vector3 p3 = ControlPoints[ClampListPos(pos + 2)].position;
			Vector3 lastPos = p1;
			for(int i=1; i<=Density; i++) {
				float t = i / (float)Density;
				Vector3 newPos = GetCatmullRomPosition(t, p0, p1, p2, p3);
				Trajectory.Points[pos * Density + i -1].SetPosition(newPos);
				Trajectory.Points[pos * Density + i -1].SetDirection(newPos-lastPos);
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
		
		GUI.color = UltiDraw.Mustard;
		GUI.backgroundColor = UltiDraw.Black;
		GUI.Box(Utility.GetGUIRect(0.025f, 0.875f, 0.2f, 0.125f), "");
		if(GUI.Button(Utility.GetGUIRect(0.0375f, 0.89f, 0.175f, 0.025f), "Reset")) {
			Positions.Clear();
			OffsetErrors.Clear();
			AngleErrors.Clear();
		}
		GUI.Label(Utility.GetGUIRect(0.0375f, 0.925f, 0.175f, 0.025f), "Average Offset Error: " + Utility.ComputeMean(OffsetErrors.ToArray()));
		GUI.Label(Utility.GetGUIRect(0.0375f, 0.95f, 0.175f, 0.025f), "Average Angle Error: " + Utility.ComputeMean(AngleErrors.ToArray()));
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
		ArrayExtensions.Add(ref ControlPoints, cp);
		//Create();
		return ControlPoints[ControlPoints.Length-1];
	}

	private void RemoveControlPoint() {
		Utility.Destroy(ControlPoints[ControlPoints.Length-1].gameObject);
		ArrayExtensions.Shrink(ref ControlPoints);
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
			
			if(GUILayout.Button("Rebuild")) {
				Target.Create();
			}

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