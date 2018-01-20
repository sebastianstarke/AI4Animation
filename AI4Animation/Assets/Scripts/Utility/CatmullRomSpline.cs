using UnityEngine;
using System.Collections;

//#if UNITY_EDITOR
//using UnityEditor;
//#endif

public class CatmullRomSpline : MonoBehaviour {

	public BioAnimation_MLP Target;

	public float Correction = 0f;

	public Trajectory Trajectory;
	public Transform[] ControlPoints;

	void Update() {
		Target.TrajectoryCorrection = Correction;
		Target.TrajectoryControl = false;

		Trajectory.Point pivot = GetClosestTrajectoryPoint(Target.transform.position);
		Trajectory.Point[] future = GetFutureTrajectory(pivot);

		Trajectory trajectory = Target.GetTrajectory();
		for(int i=0; i<future.Length; i++) {
			float weight = (float)(i+1) / (float)future.Length;
			Trajectory.Point point = trajectory.Points[60+i+1];
			point.SetPosition((1f-weight) * Target.transform.position + weight * future[i].GetPosition());
			point.SetDirection((future[i].GetDirection() + (future[i].GetPosition()-point.GetPosition()).normalized).normalized);
			point.SetVelocity(Vector3.Distance(pivot.GetPosition(), future[future.Length-1].GetPosition()));
		}
		for(int i=60; i<trajectory.Points.Length; i++) {
			trajectory.Points[i].Styles[0] = 0f;
			trajectory.Points[i].Styles[1] = 1f;
		}
	}

	void OnRenderObject() {
		if(IsInvalid()) {
			return;
		}

		if(HasChanged()) {
			Create();
		}
		Trajectory.Draw(10);
		
		UnityGL.Start();
		for(int i=0; i<ControlPoints.Length; i++) {
			UnityGL.DrawSphere(ControlPoints[i].position, 0.05f, Utility.Cyan.Transparent(0.75f));
		}
		UnityGL.Finish();

		Trajectory.Point pivot = GetClosestTrajectoryPoint(Target.transform.position);
		Trajectory.Point[] future = GetFutureTrajectory(pivot);
		UnityGL.Start();
		UnityGL.DrawSphere(pivot.GetPosition(), 0.05f, Utility.DarkRed.Transparent(0.75f));
		for(int i=0; i<future.Length; i++) {
			UnityGL.DrawSphere(future[i].GetPosition(), 0.025f, Utility.DarkGreen.Transparent(0.75f));
		}
		UnityGL.Finish();

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
		for(int pos=0; pos<ControlPoints.Length; pos++) {
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

	/*
	#if UNITY_EDITOR
	[CustomEditor(typeof(CatmullRomSpline))]
	public class CatmullRomSpline_Editor : Editor {

		public CatmullRomSpline Target;

		void Awake() {
			Target = (CatmullRomSpline)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

			Inspector();

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

		private void Inspector() {			
			Utility.SetGUIColor(Utility.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

			}
		}
	}
	#endif
	*/
	
}