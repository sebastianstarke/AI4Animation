using UnityEngine;
using System.Collections;
using System.Collections.Generic;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class TrajectoryController : MonoBehaviour {

	public SIGGRAPH_2018.BioAnimation Target;

	public bool Visualise = true;
	public bool Loop = true;

	public float Correction = 1f;

	public Trajectory Trajectory;
	public ControlPoint[] ControlPoints = new ControlPoint[0];

	private const int Density = 60;

	//private int Current = 0;

	void Update() {
		if(IsInvalid()) {
			return;
		}

		Target.TrajectoryCorrection = Correction;
		Target.TrajectoryControl = false;

		//Trajectory.Point pivot = Trajectory.Points[Current];
		Trajectory.Point pivot = GetClosestTrajectoryPoint(Target.transform.position);
		Trajectory.Point[] future = GetFutureTrajectory(pivot);
		
		for(int i=0; i<future.Length; i++) {
			Trajectory.Point point = Target.GetTrajectory().Points[60+i+1];
			point.SetPosition(future[i].GetPosition());
			point.SetDirection(future[i].GetDirection());
			point.SetVelocity(future[i].GetVelocity());
			//float weight = (float)(i+1) / (float)future.Length;
			//point.SetPosition(future[i].GetPosition());
			//point.SetDirection(future[i].GetDirection());
			//if(i == future.Length-1) {
			//	point.SetDirection(future[i].GetDirection());
			//} else {
			//	point.SetDirection(future[i].GetDirection());
				//point.SetDirection(future[future.Length-1].GetPosition() - point.GetPosition());
			//}
			/*
			point.SetPosition((1f-weight) * Target.transform.position + weight * future[i].GetPosition());
			//point.SetDirection((future[i].GetDirection() + (future[i].GetPosition()-point.GetPosition()).normalized).normalized);
			point.SetDirection((future[i].GetDirection() + (future[future.Length-1].GetPosition() - point.GetPosition()).normalized).normalized);
			if(i==0) {
				point.SetVelocity((future[0].GetPosition() - pivot.GetPosition()) * 60f);
			} else {
				point.SetVelocity((future[i].GetPosition() - future[i-1].GetPosition()) * 60f);
			}
			*/
		}
		for(int i=60; i<Target.GetTrajectory().Points.Length; i++) {
			Target.GetTrajectory().Points[i].Styles[0] = 0f;
			Target.GetTrajectory().Points[i].Styles[1] = 1f;
		}

		//Current += 1;
		//if(Current == Trajectory.Points.Length) {
		//	Current = 0;
		//}	
	}

	void OnRenderObject() {
		if(HasChanged()) {
			for(int i=0; i<ControlPoints.Length; i++) {
				ControlPoints[i].Transform.position = Utility.ProjectGround(ControlPoints[i].Transform.position, LayerMask.GetMask("Ground"));
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

		/*
		if(Visualise) {
			UltiDraw.Begin();
			for(int i=1; i<Trajectory.Points.Length; i++) {
				UltiDraw.DrawLine(Trajectory.Points[i-1].GetPosition(), Trajectory.Points[i].GetPosition(), 0.025f, UltiDraw.Magenta.Transparent(0.75f));
			}
			for(int i=0; i<ControlPoints.Length; i++) {
				UltiDraw.DrawSphere(ControlPoints[i].Transform.position, Quaternion.identity, 0.1f, UltiDraw.Cyan.Transparent(0.75f));
			}
			UltiDraw.End();
		}
		*/

		if(Visualise) {
			Trajectory.Draw(10);
		}

		
		if(Target == null) {
			return;
		}
		Trajectory.Point pivot = GetClosestTrajectoryPoint(Target.transform.position);
		Trajectory.Point[] future = GetFutureTrajectory(pivot);
		UltiDraw.Begin();
		UltiDraw.DrawSphere(pivot.GetPosition(), Quaternion.identity, 0.1f, UltiDraw.Red.Transparent(0.75f));
		for(int i=0; i<future.Length-1; i++) {
			UltiDraw.DrawSphere(future[i].GetPosition(), Quaternion.identity, 0.05f, UltiDraw.Green.Transparent(0.75f));
		}
		UltiDraw.DrawSphere(future[future.Length-1].GetPosition(), Quaternion.identity, 0.1f, UltiDraw.Red.Transparent(0.75f));
		UltiDraw.End();
		

		for(int i=0; i<ControlPoints.Length; i++) {
			ControlPoints[i].Transform.hasChanged = false;
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
			if(Loop) {
				future[i] = Trajectory.Points[(int)Mathf.Repeat((float)(pivot.GetIndex() + i + 1), (float)Trajectory.Points.Length)];
			} else {
				future[i] = Trajectory.Points[Mathf.Clamp(pivot.GetIndex() + i + 1, 0, Trajectory.Points.Length-1)];
			}
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
			if(ControlPoints[i].Transform.hasChanged) {
				return true;
			}
		}
		return false;
	}

	private void Create() {
		if(IsInvalid()) {
			return;
		}

		if(Loop) {
			Trajectory = new Trajectory(ControlPoints.Length * Density, 0);
			int index = 0;
			for(int pos=0; pos<ControlPoints.Length; pos++) {
				Vector3 p0 = ControlPoints[ClampIndex(pos - 1)].Transform.position;
				Vector3 p1 = ControlPoints[pos].Transform.position;
				Vector3 p2 = ControlPoints[ClampIndex(pos + 1)].Transform.position;
				Vector3 p3 = ControlPoints[ClampIndex(pos + 2)].Transform.position;
				for(int i=0; i<Density; i++) {
					float t = i / (float)Density;
					Trajectory.Points[index].SetPosition(GetCatmullRomPosition(t, p0, p1, p2, p3));
					index += 1;
				}
			}
		} else {
			Trajectory = new Trajectory((ControlPoints.Length-1) * Density + 1, 0);
			int index = 0;
			for(int pos=0; pos<ControlPoints.Length-1; pos++) {
				Vector3 p0 = ControlPoints[ClampIndex(pos - 1)].Transform.position;
				Vector3 p1 = ControlPoints[pos].Transform.position;
				Vector3 p2 = ControlPoints[ClampIndex(pos + 1)].Transform.position;
				Vector3 p3 = ControlPoints[ClampIndex(pos + 2)].Transform.position;
				for(int i=0; i<Density; i++) {
					float t = i / (float)Density;
					Trajectory.Points[index].SetPosition(GetCatmullRomPosition(t, p0, p1, p2, p3));
					index += 1;
				}
			}
			Trajectory.Points[Trajectory.Points.Length-1].SetPosition(ControlPoints[ControlPoints.Length-1].Transform.position);
		}

		Trajectory.Postprocess();
		for(int i=0; i<Trajectory.Points.Length; i++) {
			Trajectory.Points[i].SetDirection(ComputeDirection(i));
			Trajectory.Points[i].SetVelocity(ComputeVelocity(i));
		}
	}

	private Vector3 ComputeDirection(int index) {
		Vector3 direction = Vector3.zero;
		if(index == 0) {
			if(Loop) {
				direction = Trajectory.Points[Trajectory.Points.Length-1].GetPosition() - Trajectory.Points[0].GetPosition();
			} else {
				direction = Trajectory.Points[index+1].GetPosition() - Trajectory.Points[index].GetPosition();
			}
		} else {
			direction = Trajectory.Points[index].GetPosition() - Trajectory.Points[index-1].GetPosition();
		}
		if(direction.magnitude == 0f) {
			return Vector3.forward;
		} else {
			return direction.normalized;
		}
	}

	private Vector3 ComputeVelocity(int index) {
		Vector3 velocity = Vector3.zero;
		if(index == 0) {
			if(Loop) {
				velocity = Trajectory.Points[Trajectory.Points.Length-1].GetPosition() - Trajectory.Points[0].GetPosition();
			} else {
				velocity = Trajectory.Points[index+1].GetPosition() - Trajectory.Points[index].GetPosition();
			}
		} else {
			velocity = Trajectory.Points[index].GetPosition() - Trajectory.Points[index-1].GetPosition();
		}
		return velocity * 60f;
	}

	private int ClampIndex(int i) {
		if(Loop) {
			return (int)Mathf.Repeat((float)i, (float)ControlPoints.Length);
		} else {
			return Mathf.Clamp(i, 0, ControlPoints.Length-1);
		}
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

	}

	private void AddControlPoint() {
		ControlPoint cp = new ControlPoint();
		cp.Transform = GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
		cp.Transform.localScale = 0.05f * Vector3.one;
		cp.Transform.position = ControlPoints.Length == 0 ? transform.position : ControlPoints[ControlPoints.Length-1].Transform.position;
		cp.Transform.rotation = ControlPoints.Length == 0 ? transform.rotation : ControlPoints[ControlPoints.Length-1].Transform.rotation;
		Utility.Destroy(cp.Transform.GetComponent<MeshRenderer>());
		Utility.Destroy(cp.Transform.GetComponent<MeshFilter>());
		cp.Transform.GetComponent<Collider>().isTrigger = true;
		cp.Transform.name = "ControlPoint";
		cp.Transform.gameObject.AddComponent<MouseDrag>();
		cp.Transform.SetParent(transform);
		ArrayExtensions.Add(ref ControlPoints, cp);
		Create();
	}

	private void RemoveControlPoint() {
		if(ControlPoints.Length > 0) {
			Utility.Destroy(ControlPoints[ControlPoints.Length-1].Transform.gameObject);
			ArrayExtensions.Shrink(ref ControlPoints);
		}
		Create();
	}
	
	[System.Serializable]
	public class ControlPoint {
		public Transform Transform;
		public float Velocity;
		public float Direction;
		public float[] Style;
	}

	#if UNITY_EDITOR
	[CustomEditor(typeof(TrajectoryController))]
	public class TrajectoryController_Editor : Editor {

		public TrajectoryController Target;

		void Awake() {
			Target = (TrajectoryController)target;
		}

		public override void OnInspectorGUI() {
			//DrawDefaultInspector();

			Target.Target = (SIGGRAPH_2018.BioAnimation)EditorGUILayout.ObjectField("Target", Target.Target, typeof(SIGGRAPH_2018.BioAnimation), true);

			Target.Visualise = EditorGUILayout.Toggle("Visualise", Target.Visualise);
			Target.Loop = EditorGUILayout.Toggle("Loop", Target.Loop);
			
			if(GUILayout.Button("Rebuild")) {
				Target.Create();
			}

			if(GUILayout.Button("Add Control Point")) {
				Target.AddControlPoint();
			}
			if(GUILayout.Button("Remove Control Point")) {
				Target.RemoveControlPoint();
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

			/*
			if(GUILayout.Button("Create Circle")) {
				CreateCirlce();
			}
			if(GUILayout.Button("Create Slalom")) {
				CreateSlalom();
			}
			*/

		/*
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
		*/