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

	private const int Density = 50;

	//private int Current = 0;

	void Reset() {
		while(transform.childCount > 0) {
			Utility.Destroy(transform.GetChild(0).gameObject);
		}
	}

	void Update() {
		if(IsInvalid()) {
			return;
		}

		Target.TrajectoryCorrection = Correction;
		Target.TrajectoryControl = false;

		//Trajectory.Point pivot = Trajectory.Points[Current];
		Trajectory.Point pivot = GetClosestTrajectoryPoint(Target.transform.position);
		Trajectory.Point[] future = GetFutureTrajectory(pivot);
		
		Trajectory.Points[60].SetSpeed(pivot.GetSpeed());
		for(int i=0; i<future.Length; i++) {
			Trajectory.Point point = Target.GetTrajectory().Points[60+i+1];
			point.SetPosition(future[i].GetPosition());
			point.SetDirection(future[i].GetDirection());
			point.SetVelocity(future[i].GetVelocity());
			point.SetSpeed(future[i].GetSpeed());
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

	public void SetVisualise(bool value) {
		if(Visualise != value) {
			Visualise = value;
			#if UNITY_EDITOR
			SceneView.RepaintAll();
			#endif
		}
	}

	public void SetLoop(bool value) {
		if(Loop != value) {
			Loop = value;
			Create();
		}
	}

	public void SetCorrection(float value) {
		if(Correction != value) {
			Correction = value;
		}
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

		for(int i=0; i<ControlPoints.Length; i++) {
			ControlPoints[i].Transform.SetSiblingIndex(i);
			ControlPoints[i].Transform.name = "Control Point " + (i+1);
		}

		if(Loop) {
			Trajectory = new Trajectory(ControlPoints.Length * Density, 0);
			int index = 0;
			for(int pos=0; pos<ControlPoints.Length; pos++) {
				Vector3 p0 = ControlPoints[GetControlPoint(pos - 1)].Transform.position;
				Vector3 p1 = ControlPoints[pos].Transform.position;
				Vector3 p2 = ControlPoints[GetControlPoint(pos + 1)].Transform.position;
				Vector3 p3 = ControlPoints[GetControlPoint(pos + 2)].Transform.position;
				for(int i=0; i<Density; i++) {
					float t = (float)i / (float)Density;
					Vector3 current = GetCatmullRomVector(t, p0, p1, p2, p3);
					Vector3 previous = GetCatmullRomVector(t - 1f/(float)Density, p0, p1, p2, p3);
					Trajectory.Points[index].SetPosition(current);
					Trajectory.Points[index].SetDirection((current-previous).normalized);
					Trajectory.Points[index].SetVelocity(60f*(current-previous));
					Trajectory.Points[index].SetSpeed(GetSpeed(index));
					index += 1;
				}
			}
		} else {
			Trajectory = new Trajectory((ControlPoints.Length-1) * Density + 1, 0);
			int index = 0;
			for(int pos=0; pos<ControlPoints.Length-1; pos++) {
				Vector3 p0 = ControlPoints[GetControlPoint(pos - 1)].Transform.position;
				Vector3 p1 = ControlPoints[pos].Transform.position;
				Vector3 p2 = ControlPoints[GetControlPoint(pos + 1)].Transform.position;
				Vector3 p3 = ControlPoints[GetControlPoint(pos + 2)].Transform.position;
				for(int i=0; i<Density; i++) {
					float t = (float)i / (float)Density;
					Vector3 current = GetCatmullRomVector(t, p0, p1, p2, p3);
					Vector3 previous = GetCatmullRomVector(t - 1f/(float)Density, p0, p1, p2, p3);
					Trajectory.Points[index].SetPosition(current);
					Trajectory.Points[index].SetDirection(Quaternion.Euler(0f, Random.Range(-10f, 10f), 0f) * (current-previous).normalized);
					Trajectory.Points[index].SetVelocity(60f*(current-previous));
					Trajectory.Points[index].SetSpeed(GetSpeed(index));
					if(pos == 0) {
						Trajectory.Points[index].SetVelocity(t * Trajectory.Points[index].GetVelocity());
					}
					if(pos == ControlPoints.Length-2) {
						Trajectory.Points[index].SetVelocity((1f-t) * Trajectory.Points[index].GetVelocity());
					}
					index += 1;
				}
			}
			int last = Trajectory.Points.Length-1;
			Trajectory.Points[last].SetPosition(ControlPoints[ControlPoints.Length-1].Transform.position);
			Trajectory.Points[last].SetDirection((Trajectory.Points[last].GetPosition() - Trajectory.Points[last-1].GetPosition()).normalized);
			Trajectory.Points[last].SetVelocity(Vector3.zero);
			Trajectory.Points[last].SetSpeed(GetSpeed(last));
		}

		Trajectory.Postprocess();

		#if UNITY_EDITOR
		if(!Application.isPlaying) {
			SceneView.RepaintAll();
		}
		#endif
	}

	private float GetSpeed(int index) {
		float speed = 0f;
		for(int i=index; i<index+50; i++) {
			speed += Vector3.Distance(GetTrajectoryPoint(i-1).GetPosition(), GetTrajectoryPoint(i).GetPosition());
		}
		return speed;
	}

	private Trajectory.Point GetTrajectoryPoint(int i) {
		if(Loop) {
			return Trajectory.Points[(int)Mathf.Repeat((float)i, (float)Trajectory.Points.Length)];
		} else {
			return Trajectory.Points[Mathf.Clamp(i, 0, Trajectory.Points.Length-1)];
		}
	}

	private int GetControlPoint(int i) {
		if(Loop) {
			return (int)Mathf.Repeat((float)i, (float)ControlPoints.Length);
		} else {
			return Mathf.Clamp(i, 0, ControlPoints.Length-1);
		}
	}

	private float GetCatmullRomValue(float t, float v0, float v1, float v2, float v3) {
		float a = 2f * v1;
		float b = v2 - v0;
		float c = 2f * v0 - 5f * v1 + 4f * v2 - v3;
		float d = -v0 + 3f * v1 - 3f * v2 + v3;
		return 0.5f * (a + (b * t) + (c * t * t) + (d * t * t * t));
	}

	private Vector3 GetCatmullRomVector(float t, Vector3 v0, Vector3 v1, Vector3 v2, Vector3 v3) {
		Vector3 a = 2f * v1;
		Vector3 b = v2 - v0;
		Vector3 c = 2f * v0 - 5f * v1 + 4f * v2 - v3;
		Vector3 d = -v0 + 3f * v1 - 3f * v2 + v3;
		return 0.5f * (a + (b * t) + (c * t * t) + (d * t * t * t));
	}

	private void AddControlPoint() {
		ControlPoint cp = new ControlPoint();
		cp.Transform = GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
		cp.Transform.localScale = 0.05f * Vector3.one;
		cp.Transform.position = ControlPoints.Length == 0 ? transform.position : Vector3.Lerp(ControlPoints[(ControlPoints.Length-1)].Transform.position, ControlPoints[GetControlPoint(ControlPoints.Length)].Transform.position, 0.5f);
		cp.Transform.rotation = ControlPoints.Length == 0 ? transform.rotation : Quaternion.Slerp(ControlPoints[GetControlPoint(ControlPoints.Length-1)].Transform.rotation, ControlPoints[GetControlPoint(ControlPoints.Length)].Transform.rotation, 0.5f);
		Utility.Destroy(cp.Transform.GetComponent<MeshRenderer>());
		Utility.Destroy(cp.Transform.GetComponent<MeshFilter>());
		cp.Transform.GetComponent<Collider>().isTrigger = true;
		cp.Transform.name = "ControlPoint";
		cp.Transform.gameObject.AddComponent<MouseDrag>();
		cp.Transform.SetParent(transform);
		ArrayExtensions.Add(ref ControlPoints, cp);
		Create();
	}

	private void InsertControlPoint(int index) {
		ControlPoint cp = new ControlPoint();
		cp.Transform = GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
		cp.Transform.localScale = 0.05f * Vector3.one;
		cp.Transform.position = ControlPoints.Length == 0 ? transform.position : Vector3.Lerp(ControlPoints[GetControlPoint(index-1)].Transform.position, ControlPoints[index].Transform.position, 0.5f);
		cp.Transform.rotation = ControlPoints.Length == 0 ? transform.rotation : Quaternion.Slerp(ControlPoints[GetControlPoint(index-1)].Transform.rotation, ControlPoints[index].Transform.rotation, 0.5f);
		Utility.Destroy(cp.Transform.GetComponent<MeshRenderer>());
		Utility.Destroy(cp.Transform.GetComponent<MeshFilter>());
		cp.Transform.GetComponent<Collider>().isTrigger = true;
		cp.Transform.name = "ControlPoint";
		cp.Transform.gameObject.AddComponent<MouseDrag>();
		cp.Transform.SetParent(transform);
		ArrayExtensions.Insert(ref ControlPoints, cp, index);
		Create();
	}

	private void RemoveControlPoint(int index) {
		if(index < 0 || index >= ControlPoints.Length) {
			return;
		}
		Utility.Destroy(ControlPoints[index].Transform.gameObject);
		ArrayExtensions.RemoveAt(ref ControlPoints, index);
		Create();
	}

	void OnGUI() {

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
			UltiDraw.Begin();
			for(int i=0; i<ControlPoints.Length; i++) {
				UltiDraw.DrawCuboid(ControlPoints[i].Transform.position, ControlPoints[i].Transform.rotation, ControlPoints[i].Transform.lossyScale, UltiDraw.Red.Transparent(0.9f));
			}
			UltiDraw.End();
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
	
	[System.Serializable]
	public class ControlPoint {
		public Transform Transform;
		public float[] Style = new float[0];
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

			Target.SetVisualise(EditorGUILayout.Toggle("Visualise", Target.Visualise));
			Target.SetLoop(EditorGUILayout.Toggle("Loop", Target.Loop));
			Target.SetCorrection(EditorGUILayout.Slider("Correction", Target.Correction, 0f, 1f));		
			
			if(GUILayout.Button("Create")) {
				Target.Create();
			}

			for(int i=0; i<Target.ControlPoints.Length; i++) {
				if(Utility.GUIButton("+", UltiDraw.DarkGreen, UltiDraw.White)) {
					Target.InsertControlPoint(i);
				}
				InspectControlPoint(i);
			}
			if(Utility.GUIButton("+", UltiDraw.DarkGreen, UltiDraw.White)) {
				Target.AddControlPoint();
			}
		}

		private void InspectControlPoint(int index) {
			ControlPoint cp = Target.ControlPoints[index];
			Utility.SetGUIColor(UltiDraw.LightGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				EditorGUILayout.BeginHorizontal();
				EditorGUILayout.LabelField("Control Point " + (index+1));
				if(Utility.GUIButton("-", UltiDraw.DarkRed, UltiDraw.White, 40f, 20f)) {
					Target.RemoveControlPoint(index);
				} else {
					EditorGUILayout.EndHorizontal();
					EditorGUILayout.ObjectField("Transform", cp.Transform, typeof(Transform), true);
					EditorGUILayout.LabelField("Style: " + cp.Style.Length);
				}
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

		private void Clear() {
			while(Target.ControlPoints.Length > 0) {
				Target.RemoveControlPoint();
			}
		}
		*/