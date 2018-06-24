using UnityEngine;
using System;
using System.Collections.Generic;
using DeepLearning;
#if UNITY_EDITOR
using UnityEditor;
#endif

[RequireComponent(typeof(Actor))]
public class BioAnimation : MonoBehaviour {

	public bool Inspect = false;

	public bool ShowTrajectory = true;
	public bool ShowVelocities = true;

	public float TargetGain = 0.25f;
	public float TargetDecay = 0.05f;
	public bool TrajectoryControl = true;
	public float TrajectoryCorrection = 1f;

	public int TrajectoryDimIn = 6;
	public int TrajectoryDimOut = 6;
	public int JointDimIn = 12;
	public int JointDimOut = 12;

	public Controller Controller;

	private Actor Actor;
	private NeuralNetwork NN;
	private Trajectory Trajectory;

	//private HeightMap HeightMap;

	private Vector3 TargetDirection;
	private Vector3 TargetVelocity;

	//State
	private Vector3[] Positions = new Vector3[0];
	private Vector3[] Forwards = new Vector3[0];
	private Vector3[] Ups = new Vector3[0];
	private Vector3[] Velocities = new Vector3[0];

	//Trajectory for 60 Hz framerate
	private const int Framerate = 60;
	private const int Points = 111;
	private const int PointSamples = 12;
	private const int PastPoints = 60;
	private const int FuturePoints = 50;
	private const int RootPointIndex = 60;
	private const int PointDensity = 10;

	//GUI
	private GUIStyle FontStyle;

	void Reset() {
		Controller = new Controller();
	}

	void Awake() {
		Actor = GetComponent<Actor>();
		NN = GetComponent<NeuralNetwork>();
		TargetDirection = new Vector3(transform.forward.x, 0f, transform.forward.z);
		TargetVelocity = Vector3.zero;
		Positions = new Vector3[Actor.Bones.Length];
		Forwards = new Vector3[Actor.Bones.Length];
		Ups = new Vector3[Actor.Bones.Length];
		Velocities = new Vector3[Actor.Bones.Length];
		Trajectory = new Trajectory(Points, Controller.Styles.Length, transform.position, TargetDirection);
		//HeightMap = new HeightMap(0.25f);
		if(Controller.Styles.Length > 0) {
			for(int i=0; i<Trajectory.Points.Length; i++) {
				Trajectory.Points[i].Styles[0] = 1f;
			}
		}
		for(int i=0; i<Actor.Bones.Length; i++) {
			Positions[i] = Actor.Bones[i].Transform.position;
			Forwards[i] = Actor.Bones[i].Transform.forward;
			Ups[i] = Actor.Bones[i].Transform.up;
			Velocities[i] = Vector3.zero;
		}

		if(NN.Parameters == null) {
			Debug.Log("No parameters saved.");
			return;
		}
		NN.LoadParameters();
	}

	void Start() {
		Utility.SetFPS(60);
		FontStyle = new GUIStyle();
		FontStyle.font = (Font)Resources.Load("Fonts/Coolvetica");
		FontStyle.normal.textColor = Color.white;
		FontStyle.alignment = TextAnchor.MiddleCenter;
	}

	public Trajectory GetTrajectory() {
		return Trajectory;
	}

	void Update() {
		if(NN.Parameters == null) {
			return;
		}

		if(TrajectoryControl) {
			PredictTrajectory();
		}

		if(NN.Parameters != null) {
			Animate();
		}
	}

	private void PredictTrajectory() {
		//Calculate Bias
		float bias = PoolBias();

		//Determine Control
		float turn = Controller.QueryTurn();
		Vector3 move = Controller.QueryMove();
		bool control = turn != 0f || move != Vector3.zero;

		//Update Target Direction / Velocity / Correction
		TargetDirection = Vector3.Lerp(TargetDirection, Quaternion.AngleAxis(turn * 60f, Vector3.up) * Trajectory.Points[RootPointIndex].GetDirection(), control ? TargetGain : TargetDecay);
		TargetVelocity = Vector3.Lerp(TargetVelocity, bias * (Quaternion.LookRotation(TargetDirection, Vector3.up) * move).normalized, control ? TargetGain : TargetDecay);
		TrajectoryCorrection = Utility.Interpolate(TrajectoryCorrection, Mathf.Max(move.normalized.magnitude, Mathf.Abs(turn)), control ? TargetGain : TargetDecay);

		//Predict Future Trajectory
		Vector3[] trajectory_positions_blend = new Vector3[Trajectory.Points.Length];
		trajectory_positions_blend[RootPointIndex] = Trajectory.Points[RootPointIndex].GetTransformation().GetPosition();
		for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
			float bias_pos = 0.75f;
			float bias_dir = 1.25f;
			float bias_vel = 1.50f;
			float weight = (float)(i - RootPointIndex) / (float)FuturePoints; //w between 1/FuturePoints and 1
			float scale_pos = 1.0f - Mathf.Pow(1.0f - weight, bias_pos);
			float scale_dir = 1.0f - Mathf.Pow(1.0f - weight, bias_dir);
			float scale_vel = 1.0f - Mathf.Pow(1.0f - weight, bias_vel);

			float scale = 1f / (Trajectory.Points.Length - (RootPointIndex + 1f));

			trajectory_positions_blend[i] = trajectory_positions_blend[i-1] + 
				Vector3.Lerp(
				Trajectory.Points[i].GetPosition() - Trajectory.Points[i-1].GetPosition(), 
				scale * TargetVelocity,
				scale_pos
				);

			Trajectory.Points[i].SetDirection(Vector3.Lerp(Trajectory.Points[i].GetDirection(), TargetDirection, scale_dir));
			Trajectory.Points[i].SetVelocity(Vector3.Lerp(Trajectory.Points[i].GetVelocity(), TargetVelocity, scale_vel));
		}
		for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
			Trajectory.Points[i].SetPosition(trajectory_positions_blend[i]);
		}
		//for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
		//	Trajectory.Points[i].SetVelocity((Trajectory.Points[i].GetPosition() - Trajectory.Points[i-1].GetPosition()) * Framerate);
		//}
		float[] style = Controller.GetStyle();
		for(int i=RootPointIndex; i<Trajectory.Points.Length; i++) {
			for(int j=0; j<Trajectory.Points[i].Styles.Length; j++) {
				Trajectory.Points[i].Styles[j] = Utility.Interpolate(Trajectory.Points[i].Styles[j], style[j], Controller.Styles[j].Transition);
			}
			//Utility.Normalise(ref Trajectory.Points[i].Styles);
			Trajectory.Points[i].SetSpeed(Utility.Interpolate(Trajectory.Points[i].GetSpeed(), TargetVelocity.magnitude, control ? TargetGain : TargetDecay));
		}
	}

	private void Animate() {
		//Calculate Root
		Matrix4x4 currentRoot = Trajectory.Points[RootPointIndex].GetTransformation();
		currentRoot[1,3] = 0f; //For flat terrain

		int start = 0;
		//Input Trajectory Positions / Directions / Velocities / Styles
		for(int i=0; i<PointSamples; i++) {
			Vector3 pos = GetSample(i).GetPosition().GetRelativePositionTo(currentRoot);
			Vector3 dir = GetSample(i).GetDirection().GetRelativeDirectionTo(currentRoot);
			Vector3 vel = GetSample(i).GetVelocity().GetRelativeDirectionTo(currentRoot);
			//float speed = GetSample(i).GetSpeed();
			//Debug.Log("Trajectory " + i + " starts at " + (start + i*TrajectoryDimIn));
			NN.SetInput(start + i*TrajectoryDimIn + 0, pos.x);
			NN.SetInput(start + i*TrajectoryDimIn + 1, pos.z);
			NN.SetInput(start + i*TrajectoryDimIn + 2, dir.x);
			NN.SetInput(start + i*TrajectoryDimIn + 3, dir.z);
			NN.SetInput(start + i*TrajectoryDimIn + 4, vel.x);
			NN.SetInput(start + i*TrajectoryDimIn + 5, vel.z);
			//NN.SetInput(start + i*TrajectoryDimIn + 6, speed);
			//for(int j=0; j<Controller.Styles.Length; j++) {
			//	NN.SetInput(start + i*TrajectoryDimIn + (TrajectoryDimIn - Controller.Styles.Length) + j, GetSample(i).Styles[j]);
			//}
		}
		start += TrajectoryDimIn*PointSamples;

		Matrix4x4 previousRoot = Trajectory.Points[RootPointIndex-1].GetTransformation();
		previousRoot[1,3] = 0f; //For flat terrain

		//Input Previous Bone Positions / Velocities
		for(int i=0; i<Actor.Bones.Length; i++) {
			//Debug.Log("Joint " + Actor.Bones[i].GetName() + " starts at " + (start + i*JointDimIn));
			Vector3 pos = Positions[i].GetRelativePositionTo(previousRoot);
			Vector3 forward = Forwards[i].GetRelativeDirectionTo(previousRoot);
			Vector3 up = Ups[i].GetRelativeDirectionTo(previousRoot);
			Vector3 vel = Velocities[i].GetRelativeDirectionTo(previousRoot);
			NN.SetInput(start + i*JointDimIn + 0, pos.x);
			NN.SetInput(start + i*JointDimIn + 1, pos.y);
			NN.SetInput(start + i*JointDimIn + 2, pos.z);
			NN.SetInput(start + i*JointDimIn + 3, forward.x);
			NN.SetInput(start + i*JointDimIn + 4, forward.y);
			NN.SetInput(start + i*JointDimIn + 5, forward.z);
			NN.SetInput(start + i*JointDimIn + 6, up.x);
			NN.SetInput(start + i*JointDimIn + 7, up.y);
			NN.SetInput(start + i*JointDimIn + 8, up.z);
			NN.SetInput(start + i*JointDimIn + 9, vel.x);
			NN.SetInput(start + i*JointDimIn + 10, vel.y);
			NN.SetInput(start + i*JointDimIn + 11, vel.z);
		}
		start += JointDimIn*Actor.Bones.Length;

		//Input Phase
		float phaseValue = GetComponent<PMANN>() != null ? GetComponent<PMANN>().GetPhase() : GetComponent<MPNN>().GetPhase();
		//float phaseUpdate = GetComponent<PMANN>() != null ? GetComponent<PMANN>().GetPhaseUpdate() : GetComponent<MPNN>().GetPhaseUpdate();
		//Vector2 phaseVector = Utility.GetCirclePhase(phaseValue);
		//Vector2 phaseVectorUpdate = Utility.GetCirclePhaseUpdate(phaseValue-phaseUpdate, phaseValue);
		//NN.SetInput(start + 0, phaseVector.x);
		//NN.SetInput(start + 1, phaseVector.y);
		//NN.SetInput(start + 2, phaseVectorUpdate.x);
		//NN.SetInput(start + 3, phaseVectorUpdate.y);
		//start += 4;

		float[] style = GetSample(6).Styles;
		float[] stylePhase = Utility.StylePhase(style, phaseValue);
		for(int i=0; i<stylePhase.Length; i++) {
			NN.SetInput(start + i, stylePhase[i]);
		}
		
		/*
		//Input Height Map
		HeightMap.Sense(Actor.Bones[0].Transform.GetWorldMatrix(), LayerMask.GetMask("Ground", "Object"));
		float[] distances = HeightMap.GetDistances();
		for(int i=0; i<HeightMap.Points.Length; i++) {
			NN.SetInput(start + i, distances[i]);
		}
		start += HeightMap.Points.Length;
		*/

		//Predict
		NN.Predict();

		//Update Past Trajectory
		for(int i=0; i<RootPointIndex; i++) {
			Trajectory.Points[i].SetPosition(Trajectory.Points[i+1].GetPosition());
			Trajectory.Points[i].SetDirection(Trajectory.Points[i+1].GetDirection());
			Trajectory.Points[i].SetVelocity(Trajectory.Points[i+1].GetVelocity());
			Trajectory.Points[i].SetSpeed(Trajectory.Points[i+1].GetSpeed());
			for(int j=0; j<Trajectory.Points[i].Styles.Length; j++) {
				Trajectory.Points[i].Styles[j] = Trajectory.Points[i+1].Styles[j];
			}
		}

		//Update Root
		Vector3 translationalOffset = Vector3.zero;
		float rotationalOffset = 0f;
		//float rest = Mathf.Pow(1f - Trajectory.Points[RootPointIndex].Styles[0], 0.25f);
		float rest = 1f;
		Vector3 rootMotion = new Vector3(NN.GetOutput(TrajectoryDimOut*6 + JointDimOut*Actor.Bones.Length + 0), NN.GetOutput(TrajectoryDimOut*6 + JointDimOut*Actor.Bones.Length + 1), NN.GetOutput(TrajectoryDimOut*6 + JointDimOut*Actor.Bones.Length + 2));
		rootMotion /= Framerate;
		translationalOffset = rest * new Vector3(rootMotion.x, 0f, rootMotion.z);
		rotationalOffset = rest * rootMotion.y;

		Trajectory.Points[RootPointIndex].SetPosition(translationalOffset.GetRelativePositionFrom(currentRoot));
		Trajectory.Points[RootPointIndex].SetDirection(Quaternion.AngleAxis(rotationalOffset, Vector3.up) * Trajectory.Points[RootPointIndex].GetDirection());
		Trajectory.Points[RootPointIndex].SetVelocity(translationalOffset.GetRelativeDirectionFrom(currentRoot) * Framerate);
		Matrix4x4 nextRoot = Trajectory.Points[RootPointIndex].GetTransformation();
		nextRoot[1,3] = 0f; //For flat terrain

		//Update Future Trajectory
		for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
			Trajectory.Points[i].SetPosition(Trajectory.Points[i].GetPosition() + translationalOffset.GetRelativeDirectionFrom(nextRoot));
			Trajectory.Points[i].SetDirection(Quaternion.AngleAxis(rotationalOffset, Vector3.up) * Trajectory.Points[i].GetDirection());
			Trajectory.Points[i].SetVelocity(Trajectory.Points[i].GetVelocity() + translationalOffset.GetRelativeDirectionFrom(nextRoot) * Framerate);
		}
		start = 0;
		for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
			//ROOT	1		2		3		4		5
			//.x....x.......x.......x.......x.......x
			int index = i;
			int prevSampleIndex = GetPreviousSample(index).GetIndex() / PointDensity;
			int nextSampleIndex = GetNextSample(index).GetIndex() / PointDensity;
			float factor = (float)(i % PointDensity) / PointDensity;

			Vector3 prevPos = new Vector3(
				NN.GetOutput(start + (prevSampleIndex-6)*TrajectoryDimOut + 0),
				0f,
				NN.GetOutput(start + (prevSampleIndex-6)*TrajectoryDimOut + 1)
			).GetRelativePositionFrom(nextRoot);
			Vector3 prevDir = new Vector3(
				NN.GetOutput(start + (prevSampleIndex-6)*TrajectoryDimOut + 2),
				0f,
				NN.GetOutput(start + (prevSampleIndex-6)*TrajectoryDimOut + 3)
			).normalized.GetRelativeDirectionFrom(nextRoot);
			Vector3 prevVel = new Vector3(
				NN.GetOutput(start + (prevSampleIndex-6)*TrajectoryDimOut + 4),
				0f,
				NN.GetOutput(start + (prevSampleIndex-6)*TrajectoryDimOut + 5)
			).GetRelativeDirectionFrom(nextRoot);

			Vector3 nextPos = new Vector3(
				NN.GetOutput(start + (nextSampleIndex-6)*TrajectoryDimOut + 0),
				0f,
				NN.GetOutput(start + (nextSampleIndex-6)*TrajectoryDimOut + 1)
			).GetRelativePositionFrom(nextRoot);
			Vector3 nextDir = new Vector3(
				NN.GetOutput(start + (nextSampleIndex-6)*TrajectoryDimOut + 2),
				0f,
				NN.GetOutput(start + (nextSampleIndex-6)*TrajectoryDimOut + 3)
			).normalized.GetRelativeDirectionFrom(nextRoot);
			Vector3 nextVel = new Vector3(
				NN.GetOutput(start + (nextSampleIndex-6)*TrajectoryDimOut + 4),
				0f,
				NN.GetOutput(start + (nextSampleIndex-6)*TrajectoryDimOut + 5)
			).GetRelativeDirectionFrom(nextRoot);

			Vector3 pos = (1f - factor) * prevPos + factor * nextPos;
			Vector3 dir = ((1f - factor) * prevDir + factor * nextDir).normalized;
			Vector3 vel = (1f - factor) * prevVel + factor * nextVel;

			pos = Vector3.Lerp(Trajectory.Points[i].GetPosition() + vel / Framerate, pos, 0.5f);

			Trajectory.Points[i].SetPosition(
				Utility.Interpolate(
					Trajectory.Points[i].GetPosition(),
					pos,
					TrajectoryCorrection
					)
				);
			Trajectory.Points[i].SetDirection(
				Utility.Interpolate(
					Trajectory.Points[i].GetDirection(),
					dir,
					TrajectoryCorrection
					)
				);
			Trajectory.Points[i].SetVelocity(
				Utility.Interpolate(
					Trajectory.Points[i].GetVelocity(),
					vel,
					TrajectoryCorrection
					)
				);
		}
		start += TrajectoryDimOut*6;

		//Compute Posture
		for(int i=0; i<Actor.Bones.Length; i++) {
			Vector3 position = new Vector3(NN.GetOutput(start + i*JointDimOut + 0), NN.GetOutput(start + i*JointDimOut + 1), NN.GetOutput(start + i*JointDimOut + 2)).GetRelativePositionFrom(currentRoot);
			Vector3 forward = new Vector3(NN.GetOutput(start + i*JointDimOut + 3), NN.GetOutput(start + i*JointDimOut + 4), NN.GetOutput(start + i*JointDimOut + 5)).normalized.GetRelativeDirectionFrom(currentRoot);
			Vector3 up = new Vector3(NN.GetOutput(start + i*JointDimOut + 6), NN.GetOutput(start + i*JointDimOut + 7), NN.GetOutput(start + i*JointDimOut + 8)).normalized.GetRelativeDirectionFrom(currentRoot);
			Vector3 velocity = new Vector3(NN.GetOutput(start + i*JointDimOut + 9), NN.GetOutput(start + i*JointDimOut + 10), NN.GetOutput(start + i*JointDimOut + 11)).GetRelativeDirectionFrom(currentRoot);

			Positions[i] = Vector3.Lerp(Positions[i] + velocity / Framerate, position, 0.5f);
			Forwards[i] = forward;
			Ups[i] = up;
			Velocities[i] = velocity;
		}
		start += JointDimOut*Actor.Bones.Length;
		
		//Assign Posture
		transform.position = nextRoot.GetPosition();
		transform.rotation = nextRoot.GetRotation();
		for(int i=0; i<Actor.Bones.Length; i++) {
			Actor.Bones[i].Transform.position = Positions[i];
			Actor.Bones[i].Transform.rotation = Quaternion.LookRotation(Forwards[i], Ups[i]);
		}
	}

	private float PoolBias() {
		float[] styles = Trajectory.Points[RootPointIndex].Styles;
		float bias = 0f;
		for(int i=0; i<styles.Length; i++) {
			float _bias = Controller.Styles[i].Bias;
			float max = 0f;
			for(int j=0; j<Controller.Styles[i].Multipliers.Length; j++) {
				if(Input.GetKey(Controller.Styles[i].Multipliers[j].Key)) {
					max = Mathf.Max(max, Controller.Styles[i].Bias * Controller.Styles[i].Multipliers[j].Value);
				}
			}
			for(int j=0; j<Controller.Styles[i].Multipliers.Length; j++) {
				if(Input.GetKey(Controller.Styles[i].Multipliers[j].Key)) {
					_bias = Mathf.Min(max, _bias * Controller.Styles[i].Multipliers[j].Value);
				}
			}
			bias += styles[i] * _bias;
		}
		return bias;
	}

	private Trajectory.Point GetSample(int index) {
		return Trajectory.Points[Mathf.Clamp(index*10, 0, Trajectory.Points.Length-1)];
	}

	private Trajectory.Point GetPreviousSample(int index) {
		return GetSample(index / 10);
	}

	private Trajectory.Point GetNextSample(int index) {
		if(index % 10 == 0) {
			return GetSample(index / 10);
		} else {
			return GetSample(index / 10 + 1);
		}
	}

	void OnGUI() {
		if(NN.Parameters == null) {
			return;
		}
		FontStyle.fontSize = Mathf.RoundToInt(0.0125f * Screen.width);

		/*
		float[] style = Trajectory.Points[RootPointIndex].Styles;
		//Color[] colors = UltiDraw.GetRainbowColors(style.Length);
		Rect center = Utility.GetGUIRect(0.25f, 0.05f, 0.5f + 0.5f / style.Length, 0.15f);
		GUI.Box(center, "");
		for(int i=0; i<style.Length; i++) {
			GUI.Box(Utility.GetGUIRect(
				0.25f + (float)i / (float)(style.Length-1) * 0.5f - 0.5f / style.Length / 2f, 
				0.05f, 
				0.5f / style.Length, 
				0.15f * style[i]), 
				"");
		}
		*/
	}

	void OnRenderObject() {
		if(Application.isPlaying) {
			if(NN.Parameters == null) {
				return;
			}

			if(ShowTrajectory) {
				UltiDraw.Begin();
				UltiDraw.DrawLine(Trajectory.Points[RootPointIndex].GetPosition(), Trajectory.Points[RootPointIndex].GetPosition() + TargetDirection, 0.05f, 0f, UltiDraw.Red.Transparent(0.75f));
				UltiDraw.DrawLine(Trajectory.Points[RootPointIndex].GetPosition(), Trajectory.Points[RootPointIndex].GetPosition() + TargetVelocity, 0.05f, 0f, UltiDraw.Green.Transparent(0.75f));
				UltiDraw.End();
				Trajectory.Draw(10);
			}

			if(ShowVelocities) {
				UltiDraw.Begin();
				for(int i=0; i<Actor.Bones.Length; i++) {
					UltiDraw.DrawArrow(
						Actor.Bones[i].Transform.position,
						Actor.Bones[i].Transform.position + Velocities[i],
						0.75f,
						0.0075f,
						0.05f,
						UltiDraw.Purple.Transparent(0.5f)
					);
				}
				UltiDraw.End();
			}

			UltiDraw.Begin();
			Color[] colors = UltiDraw.GetRainbowColors(Controller.Styles.Length);
			for(int i=0; i<Controller.Styles.Length; i++) {
				float x = (float)i/(float)(Controller.Styles.Length-1);
				x = Utility.Normalise(x, 0f, 1f, 0.3f, 0.7f);
				float y = 0.85f;
				float phase = GetComponent<PMANN>() != null ? GetComponent<PMANN>().GetPhase() : GetComponent<MPNN>().GetPhase();
				UltiDraw.DrawGUICircularPivot(new Vector2(x, y), 0.04f, UltiDraw.DarkGrey, phase * 360f, GetSample(6).Styles[i], colors[i]);
			}
			UltiDraw.End();

			//HeightMap.Draw();
		}
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			OnRenderObject();
		}
	}

	#if UNITY_EDITOR
	[CustomEditor(typeof(BioAnimation))]
	public class BioAnimation_Editor : Editor {

		public BioAnimation Target;

		void Awake() {
			Target = (BioAnimation)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

			Inspector();
			Target.Controller.Inspector();

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

		private void Inspector() {
			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				if(Utility.GUIButton("Animation", UltiDraw.DarkGrey, UltiDraw.White)) {
					Target.Inspect = !Target.Inspect;
				}

				if(Target.Inspect) {
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Target.TrajectoryDimIn = EditorGUILayout.IntField("Trajectory Dim X", Target.TrajectoryDimIn);
						Target.TrajectoryDimOut = EditorGUILayout.IntField("Trajectory Dim Y", Target.TrajectoryDimOut);
						Target.JointDimIn = EditorGUILayout.IntField("Joint Dim X", Target.JointDimIn);
						Target.JointDimOut = EditorGUILayout.IntField("Joint Dim Y", Target.JointDimOut);
						Target.ShowTrajectory = EditorGUILayout.Toggle("Show Trajectory", Target.ShowTrajectory);
						Target.ShowVelocities = EditorGUILayout.Toggle("Show Velocities", Target.ShowVelocities);
						Target.TargetGain = EditorGUILayout.Slider("Target Gain", Target.TargetGain, 0f, 1f);
						Target.TargetDecay = EditorGUILayout.Slider("Target Decay", Target.TargetDecay, 0f, 1f);
						Target.TrajectoryControl = EditorGUILayout.Toggle("Trajectory Control", Target.TrajectoryControl);
						Target.TrajectoryCorrection = EditorGUILayout.Slider("Trajectory Correction", Target.TrajectoryCorrection, 0f, 1f);
					}
				}
			}
		}
	}
	#endif
}