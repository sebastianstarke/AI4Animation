using UnityEngine;
using System;
using System.Collections.Generic;
using DeepLearning;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class BioAnimation : MonoBehaviour {

	public bool Inspect = false;

	public int Framerate = 60;
	public bool ShowTrajectory = true;
	public bool ShowVelocities = true;

	public float TargetBlending = 0.25f;
	public float StyleTransition = 0.25f;
	public bool TrajectoryControl = true;
	public float TrajectoryCorrection = 1f;

	public int TrajectoryDimIn = 6;
	public int TrajectoryDimOut = 6;
	public int JointDimIn = 12;
	public int JointDimOut = 12;

	public Controller Controller;
	public Character Character;
	public NeuralNetwork NN;

	public bool MotionEditing = true;
	public SerialCCD[] IKSolvers = new SerialCCD[0];

	private Trajectory Trajectory;

	private Vector3 TargetDirection;
	private Vector3 TargetVelocity;

	//State
	private Transform[] Joints;

	private Vector3[] Positions = new Vector3[0];
	private Vector3[] Forwards = new Vector3[0];
	private Vector3[] Ups = new Vector3[0];
	private Vector3[] Velocities = new Vector3[0];

	private List<Vector3>[] PastPositions = new List<Vector3>[0];
	private List<Vector3>[] PastForwards = new List<Vector3>[0];
	private List<Vector3>[] PastUps = new List<Vector3>[0];
	private List<Vector3>[] PastVelocities = new List<Vector3>[0];
	
	private List<Vector3>[] FuturePositions = new List<Vector3>[0];
	private List<Vector3>[] FutureForwards = new List<Vector3>[0];
	private List<Vector3>[] FutureUps = new List<Vector3>[0];
	private List<Vector3>[] FutureVelocities = new List<Vector3>[0];

	//Trajectory for 60 Hz framerate
	private const int Points = 111;
	private const int PointSamples = 12;
	private const int RootPointIndex = 60;
	private const int PointDensity = 10;

	void Reset() {
		Controller = new Controller();
		Character = new Character();
		Character.BuildHierarchy(transform);
		NN = new NeuralNetwork(TYPE.Vanilla);
	}

	void Awake() {
		DetectJoints();
		
		TargetDirection = new Vector3(transform.forward.x, 0f, transform.forward.z);
		TargetVelocity = Vector3.zero;
		Positions = new Vector3[Joints.Length];
		Forwards = new Vector3[Joints.Length];
		Ups = new Vector3[Joints.Length];
		Velocities = new Vector3[Joints.Length];
		Trajectory = new Trajectory(Points, Controller.Styles.Length, transform.position, TargetDirection);
		if(Controller.Styles.Length > 0) {
			for(int i=0; i<Trajectory.Points.Length; i++) {
				Trajectory.Points[i].Styles[0] = 1f;
			}
		}
		Trajectory.Postprocess();
		for(int i=0; i<Joints.Length; i++) {
			Positions[i] = Joints[i].position;
			Forwards[i] = Joints[i].forward;
			Ups[i] = Joints[i].up;
			Velocities[i] = Vector3.zero;
		}
		//
		PastPositions = new List<Vector3>[Joints.Length];
		PastForwards = new List<Vector3>[Joints.Length];
		PastUps = new List<Vector3>[Joints.Length];
		PastVelocities = new List<Vector3>[Joints.Length];
		for(int i=0; i<Joints.Length; i++) {
			PastPositions[i] = new List<Vector3>();
			PastForwards[i] = new List<Vector3>();
			PastUps[i] = new List<Vector3>();
			PastVelocities[i] = new List<Vector3>();
			for(int j=0; j<60; j++) {
				PastPositions[i].Add(Positions[i]);
				PastForwards[i].Add(Forwards[i]);
				PastUps[i].Add(Ups[i]);
				PastVelocities[i].Add(Velocities[i]);
			}
		}
		//
		//
		FuturePositions = new List<Vector3>[Joints.Length];
		FutureForwards = new List<Vector3>[Joints.Length];
		FutureUps = new List<Vector3>[Joints.Length];
		FutureVelocities = new List<Vector3>[Joints.Length];
		for(int i=0; i<Joints.Length; i++) {
			FuturePositions[i] = new List<Vector3>();
			FutureForwards[i] = new List<Vector3>();
			FutureUps[i] = new List<Vector3>();
			FutureVelocities[i] = new List<Vector3>();
			for(int j=0; j<5; j++) {
				FuturePositions[i].Add(Positions[i]);
				FutureForwards[i].Add(Forwards[i]);
				FutureUps[i].Add(Ups[i]);
				FutureVelocities[i].Add(Velocities[i]);
			}
		}
		//
		if(NN.Model.Parameters == null) {
			Debug.Log("No parameters saved.");
			return;
		}
		NN.Model.LoadParameters();
	}

	void Start() {
		Utility.SetFPS(60);
	}

	public Trajectory GetTrajectory() {
		return Trajectory;
	}

	public void EditMotion(bool value) {
		MotionEditing = value;
		if(MotionEditing) {
			for(int i=0; i<IKSolvers.Length; i++) {
				IKSolvers[i].TargetPosition = IKSolvers[i].EndEffector.position;
			}
		}
	}

	public void DetectJoints() {
		System.Array.Resize(ref Joints, 0);
		Action<Transform> recursion = null;
		recursion = new Action<Transform>((transform) => {
			if(Character.FindSegment(transform.name) != null) {
				System.Array.Resize(ref Joints, Joints.Length+1);
				Joints[Joints.Length-1] = transform;
			}
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i));
			}
		});
		recursion(transform);
	}

	void Update() {
		if(TrajectoryControl) {
			PredictTrajectory();
		}

		if(NN.Model.Parameters != null) {
			Animate();
		}

		if(MotionEditing) {
			EditMotion();
		}
		
		//Update Skeleton
		Character.FetchTransformations(transform);
	}

	private void PredictTrajectory() {
		//Calculate Bias
		float bias = PoolBias();

		//Update Target Direction / Velocity 
		TargetDirection = Vector3.Lerp(TargetDirection, Quaternion.AngleAxis(Controller.QueryTurn() * 60f, Vector3.up) * Trajectory.Points[RootPointIndex].GetDirection(), TargetBlending);
		TargetVelocity = Vector3.Lerp(TargetVelocity, bias * (Quaternion.LookRotation(TargetDirection, Vector3.up) * Controller.QueryMove()).normalized, TargetBlending);

		//Update Trajectory Correction
		TrajectoryCorrection = Utility.Interpolate(TrajectoryCorrection, Mathf.Max(Controller.QueryMove().normalized.magnitude, Mathf.Abs(Controller.QueryTurn())), TargetBlending);

		//Update Style
		float[] style = new float[Controller.Styles.Length];
		for(int i=0; i<Controller.Styles.Length; i++) {
			style[i] = Controller.Styles[i].Query() ? 1f : 0f;
			Trajectory.Points[RootPointIndex].Styles[i] = Utility.Interpolate(Trajectory.Points[RootPointIndex].Styles[i], Controller.Styles[i].Query() ? 1f : 0f, StyleTransition);
		}

		//Predict Future Trajectory
		Vector3[] trajectory_positions_blend = new Vector3[Trajectory.Points.Length];
		trajectory_positions_blend[RootPointIndex] = Trajectory.Points[RootPointIndex].GetTransformation().GetPosition();
		for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
			float bias_pos = 0.75f;
			float bias_dir = 1.25f;
			float scale_pos = (1.0f - Mathf.Pow(1.0f - ((float)(i - RootPointIndex) / (RootPointIndex)), bias_pos));
			float scale_dir = (1.0f - Mathf.Pow(1.0f - ((float)(i - RootPointIndex) / (RootPointIndex)), bias_dir));
			
			float scale = 1f / (Trajectory.Points.Length - (RootPointIndex + 1f));

			trajectory_positions_blend[i] = trajectory_positions_blend[i-1] + Vector3.Lerp(
				Trajectory.Points[i].GetPosition() - Trajectory.Points[i-1].GetPosition(), 
				scale * TargetVelocity,
				scale_pos);

			Trajectory.Points[i].SetDirection(Vector3.Lerp(Trajectory.Points[i].GetDirection(), TargetDirection, scale_dir));

			float weight = (float)(i-60) / (float)(Trajectory.Points.Length-61);
			for(int j=0; j<Trajectory.Points[i].Styles.Length; j++) {
				Trajectory.Points[i].Styles[j] = Utility.Interpolate(Trajectory.Points[RootPointIndex].Styles[j], style[j], weight);
			}
		}
		for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
			Trajectory.Points[i].SetPosition(trajectory_positions_blend[i]);
		}
		for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
			Trajectory.Points[i].SetVelocity((Trajectory.Points[i].GetPosition() - Trajectory.Points[i-1].GetPosition()) * Framerate);
		}
		for(int i=RootPointIndex; i<Trajectory.Points.Length; i+=PointDensity) {
			Trajectory.Points[i].Postprocess();
		}
		/*
		for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
			Trajectory.Point prev = GetPreviousSample(i);
			Trajectory.Point next = GetNextSample(i);
			float factor = (float)(i % PointDensity) / PointDensity;

			Trajectory.Points[i].SetPosition(((1f-factor)*prev.GetPosition() + factor*next.GetPosition()));
			Trajectory.Points[i].SetDirection(((1f-factor)*prev.GetDirection() + factor*next.GetDirection()));
			Trajectory.Points[i].SetVelocity(((1f-factor)*prev.GetVelocity() + factor*next.GetVelocity()));
			Trajectory.Points[i].SetLeftsample((1f-factor)*prev.GetLeftSample() + factor*next.GetLeftSample());
			Trajectory.Points[i].SetRightSample((1f-factor)*prev.GetRightSample() + factor*next.GetRightSample());
			Trajectory.Points[i].SetSlope((1f-factor)*prev.GetSlope() + factor*next.GetSlope());
		}
		*/
	}

	private void Animate() {
		//Calculate Root
		Matrix4x4 currentRoot = Trajectory.Points[RootPointIndex].GetTransformation();
		currentRoot[1,3] = 0f; //Fix for flat terrain

		int start = 0;
		//Input Trajectory Positions / Directions / Velocities / Styles
		for(int i=0; i<PointSamples; i++) {
			//Debug.Log("Trajectory sample " + i + " starts at " + (start + i*TrajectoryDimIn));
			Vector3 pos = GetSample(i).GetPosition().GetRelativePositionTo(currentRoot);
			Vector3 dir = GetSample(i).GetDirection().GetRelativeDirectionTo(currentRoot);
			Vector3 vel = GetSample(i).GetVelocity().GetRelativeDirectionTo(currentRoot);
			NN.Model.SetInput(start + i*TrajectoryDimIn + 0, pos.x);
			NN.Model.SetInput(start + i*TrajectoryDimIn + 1, pos.z);
			NN.Model.SetInput(start + i*TrajectoryDimIn + 2, dir.x);
			NN.Model.SetInput(start + i*TrajectoryDimIn + 3, dir.z);
			NN.Model.SetInput(start + i*TrajectoryDimIn + 4, vel.x);
			NN.Model.SetInput(start + i*TrajectoryDimIn + 5, vel.z);
			for(int j=0; j<Controller.Styles.Length; j++) {
				//Debug.Log(start + i*TrajectoryDimIn + (TrajectoryDimIn - Controller.Styles.Length) + j);
				NN.Model.SetInput(start + i*TrajectoryDimIn + (TrajectoryDimIn - Controller.Styles.Length) + j, GetSample(i).Styles[j]);
			}
		}
		start += TrajectoryDimIn*PointSamples;

		Matrix4x4 previousRoot = Trajectory.Points[RootPointIndex-1].GetTransformation();
		previousRoot[1,3] = 0f; //Fix for flat terrain

		//Input Previous Bone Positions / Velocities
		for(int i=0; i<Joints.Length; i++) {
			//Debug.Log("Joint " + Joints[i].name + " starts at " + (start + i*JointDimIn));
			Vector3 pos = Positions[i].GetRelativePositionTo(previousRoot);
			Vector3 forward = Forwards[i].GetRelativeDirectionTo(previousRoot);
			Vector3 up = Ups[i].GetRelativeDirectionTo(previousRoot);
			Vector3 vel = Velocities[i].GetRelativeDirectionTo(previousRoot);
			NN.Model.SetInput(start + i*JointDimIn + 0, pos.x);
			NN.Model.SetInput(start + i*JointDimIn + 1, pos.y);
			NN.Model.SetInput(start + i*JointDimIn + 2, pos.z);
			NN.Model.SetInput(start + i*JointDimIn + 3, forward.x);
			NN.Model.SetInput(start + i*JointDimIn + 4, forward.y);
			NN.Model.SetInput(start + i*JointDimIn + 5, forward.z);
			NN.Model.SetInput(start + i*JointDimIn + 6, up.x);
			NN.Model.SetInput(start + i*JointDimIn + 7, up.y);
			NN.Model.SetInput(start + i*JointDimIn + 8, up.z);
			NN.Model.SetInput(start + i*JointDimIn + 9, vel.x);
			NN.Model.SetInput(start + i*JointDimIn + 10, vel.y);
			NN.Model.SetInput(start + i*JointDimIn + 11, vel.z);
			//if(Joints[i].name == "LeftHand" || Joints[i].name == "RightHand" || Joints[i].name == "LeftToeBase" || Joints[i].name == "RightToeBase") {
			//	Debug.Log(start + i*JointDimIn + 9);
			//	Debug.Log(start + i*JointDimIn + 10);
			//	Debug.Log(start + i*JointDimIn + 11);
			//}
		}
		start += JointDimIn*Joints.Length;

		/*
		//Input Past Posture
		for(int i=5; i<6; i++) {
			for(int j=0; j<Joints.Length; j++) {
				Vector3 pos = PastPositions[j][i*10].GetRelativePositionTo(previousRoot);
				Vector3 forward = PastForwards[j][i*10].GetRelativeDirectionTo(previousRoot);
				Vector3 up = PastUps[j][i*10].GetRelativeDirectionTo(previousRoot);
				Vector3 vel = PastVelocities[j][i*10].GetRelativeDirectionTo(previousRoot);
				NN.Model.SetInput(start + j*JointDimIn + 0, pos.x);
				NN.Model.SetInput(start + j*JointDimIn + 1, pos.y);
				NN.Model.SetInput(start + j*JointDimIn + 2, pos.z);
				NN.Model.SetInput(start + j*JointDimIn + 3, forward.x);
				NN.Model.SetInput(start + j*JointDimIn + 4, forward.y);
				NN.Model.SetInput(start + j*JointDimIn + 5, forward.z);
				NN.Model.SetInput(start + j*JointDimIn + 6, up.x);
				NN.Model.SetInput(start + j*JointDimIn + 7, up.y);
				NN.Model.SetInput(start + j*JointDimIn + 8, up.z);
				NN.Model.SetInput(start + j*JointDimIn + 9, vel.x);
				NN.Model.SetInput(start + j*JointDimIn + 10, vel.y);
				NN.Model.SetInput(start + j*JointDimIn + 11, vel.z);
			}
			start += JointDimIn*Joints.Length;
		}
		*/

		//Predict
		NN.Model.Predict();

		//Update Past Trajectory
		for(int i=0; i<RootPointIndex; i++) {
			Trajectory.Points[i].SetPosition(Trajectory.Points[i+1].GetPosition());
			Trajectory.Points[i].SetDirection(Trajectory.Points[i+1].GetDirection());
			Trajectory.Points[i].SetVelocity(Trajectory.Points[i+1].GetVelocity());
			Trajectory.Points[i].SetLeftsample(Trajectory.Points[i+1].GetLeftSample());
			Trajectory.Points[i].SetRightSample(Trajectory.Points[i+1].GetRightSample());
			Trajectory.Points[i].SetSlope(Trajectory.Points[i+1].GetSlope());
			for(int j=0; j<Trajectory.Points[i].Styles.Length; j++) {
				Trajectory.Points[i].Styles[j] = Trajectory.Points[i+1].Styles[j];
			}
		}

		//Update Current Trajectory
		Vector3 translationalOffset = new Vector3(NN.Model.GetOutput(TrajectoryDimOut*6 + JointDimOut*Joints.Length + 0), 0f, NN.Model.GetOutput(TrajectoryDimOut*6 + JointDimOut*Joints.Length + 1));
		float rotationalOffset = NN.Model.GetOutput(TrajectoryDimOut*6 + JointDimOut*Joints.Length + 2);

		//translationalOffset *= Utility.Exponential01(translationalOffset.magnitude / 0.001f);
		//rotationalOffset *= Utility.Exponential01(Mathf.Abs(rotationalOffset) / 0.01f);
		
		Trajectory.Points[RootPointIndex].SetPosition(translationalOffset.GetRelativePositionFrom(currentRoot));
		Trajectory.Points[RootPointIndex].SetDirection(Quaternion.AngleAxis(rotationalOffset, Vector3.up) * Trajectory.Points[RootPointIndex].GetDirection());
		Trajectory.Points[RootPointIndex].SetVelocity(translationalOffset.GetRelativeDirectionFrom(currentRoot) * Framerate);
		Trajectory.Points[RootPointIndex].Postprocess();
		Matrix4x4 nextRoot = Trajectory.Points[RootPointIndex].GetTransformation();
		nextRoot[1,3] = 0f; //Fix for flat terrain

		//Update Future Trajectory
		for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
			Trajectory.Points[i].SetPosition(Trajectory.Points[i].GetPosition() + translationalOffset.GetRelativeDirectionFrom(nextRoot));
			Trajectory.Points[i].SetDirection(Quaternion.AngleAxis(rotationalOffset, Vector3.up) * Trajectory.Points[i].GetDirection());
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
				NN.Model.GetOutput(start + (prevSampleIndex-6)*TrajectoryDimOut + 0),
				0f,
				NN.Model.GetOutput(start + (prevSampleIndex-6)*TrajectoryDimOut + 1)
			).GetRelativePositionFrom(nextRoot);
			Vector3 prevDir = new Vector3(
				NN.Model.GetOutput(start + (prevSampleIndex-6)*TrajectoryDimOut + 2),
				0f,
				NN.Model.GetOutput(start + (prevSampleIndex-6)*TrajectoryDimOut + 3)
			).normalized.GetRelativeDirectionFrom(nextRoot);
			Vector3 prevVel = new Vector3(
				NN.Model.GetOutput(start + (prevSampleIndex-6)*TrajectoryDimOut + 4),
				0f,
				NN.Model.GetOutput(start + (prevSampleIndex-6)*TrajectoryDimOut + 5)
			).GetRelativeDirectionFrom(nextRoot);

			Vector3 nextPos = new Vector3(
				NN.Model.GetOutput(start + (nextSampleIndex-6)*TrajectoryDimOut + 0),
				0f,
				NN.Model.GetOutput(start + (nextSampleIndex-6)*TrajectoryDimOut + 1)
			).GetRelativePositionFrom(nextRoot);
			Vector3 nextDir = new Vector3(
				NN.Model.GetOutput(start + (nextSampleIndex-6)*TrajectoryDimOut + 2),
				0f,
				NN.Model.GetOutput(start + (nextSampleIndex-6)*TrajectoryDimOut + 3)
			).normalized.GetRelativeDirectionFrom(nextRoot);
			Vector3 nextVel = new Vector3(
				NN.Model.GetOutput(start + (nextSampleIndex-6)*TrajectoryDimOut + 4),
				0f,
				NN.Model.GetOutput(start + (nextSampleIndex-6)*TrajectoryDimOut + 5)
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
		}
		start += TrajectoryDimOut*6;
		for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
			Trajectory.Points[i].SetVelocity((Trajectory.Points[i].GetPosition() - Trajectory.Points[i-1].GetPosition()) * Framerate);
		}
		for(int i=RootPointIndex+PointDensity; i<Trajectory.Points.Length; i+=PointDensity) {
			Trajectory.Points[i].Postprocess();
		}
		
		/*
		for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
			Trajectory.Point prev = GetPreviousSample(i);
			Trajectory.Point next = GetNextSample(i);
			float factor = (float)(i % PointDensity) / PointDensity;

			Trajectory.Points[i].SetPosition(((1f-factor)*prev.GetPosition() + factor*next.GetPosition()));
			Trajectory.Points[i].SetDirection(((1f-factor)*prev.GetDirection() + factor*next.GetDirection()));
			Trajectory.Points[i].SetVelocity(((1f-factor)*prev.GetVelocity() + factor*next.GetVelocity()));
			Trajectory.Points[i].SetLeftsample((1f-factor)*prev.GetLeftSample() + factor*next.GetLeftSample());
			Trajectory.Points[i].SetRightSample((1f-factor)*prev.GetRightSample() + factor*next.GetRightSample());
			Trajectory.Points[i].SetSlope((1f-factor)*prev.GetSlope() + factor*next.GetSlope());
		}
		*/

		//Update Previous Postures
		for(int i=0; i<Joints.Length; i++) {
			PastPositions[i].RemoveAt(0);
			PastPositions[i].Add(Positions[i]);
			PastForwards[i].RemoveAt(0);
			PastForwards[i].Add(Forwards[i]);
			PastUps[i].RemoveAt(0);
			PastUps[i].Add(Ups[i]);
			PastVelocities[i].RemoveAt(0);
			PastVelocities[i].Add(Velocities[i]);
		}

		/*
		//Compute Future Postures
		for(int p=0; p<5; p++) {
			int pivot = TrajectoryDimOut*6 + JointDimOut*Joints.Length + 3 + p*JointDimOut*Joints.Length;
			for(int i=0; i<Joints.Length; i++) {
				Vector3 position = new Vector3(NN.Model.GetOutput(pivot + 0), NN.Model.GetOutput(pivot + 1), NN.Model.GetOutput(pivot + 2)).GetRelativePositionFrom(currentRoot);
				Vector3 forward = new Vector3(NN.Model.GetOutput(pivot + 3), NN.Model.GetOutput(pivot + 4), NN.Model.GetOutput(pivot + 5)).normalized.GetRelativeDirectionFrom(currentRoot);
				Vector3 up = new Vector3(NN.Model.GetOutput(pivot + 6), NN.Model.GetOutput(pivot + 7), NN.Model.GetOutput(pivot + 8)).normalized.GetRelativeDirectionFrom(currentRoot);
				Vector3 velocity = new Vector3(NN.Model.GetOutput(pivot + 9), NN.Model.GetOutput(pivot + 10), NN.Model.GetOutput(pivot + 11)).GetRelativeDirectionFrom(currentRoot);
				pivot += JointDimOut;
				FuturePositions[i][p] = Vector3.Lerp(FuturePositions[i][p] + velocity / Framerate, position, 0.5f);
				FutureForwards[i][p] = forward;
				FutureUps[i][p] = up;
				FutureVelocities[i][p] = velocity;
			}
		}
		*/

		//Compute Posture
		for(int i=0; i<Joints.Length; i++) {
			Vector3 position = new Vector3(NN.Model.GetOutput(start + i*JointDimOut + 0), NN.Model.GetOutput(start + i*JointDimOut + 1), NN.Model.GetOutput(start + i*JointDimOut + 2)).GetRelativePositionFrom(currentRoot);
			Vector3 forward = new Vector3(NN.Model.GetOutput(start + i*JointDimOut + 3), NN.Model.GetOutput(start + i*JointDimOut + 4), NN.Model.GetOutput(start + i*JointDimOut + 5)).normalized.GetRelativeDirectionFrom(currentRoot);
			Vector3 up = new Vector3(NN.Model.GetOutput(start + i*JointDimOut + 6), NN.Model.GetOutput(start + i*JointDimOut + 7), NN.Model.GetOutput(start + i*JointDimOut + 8)).normalized.GetRelativeDirectionFrom(currentRoot);
			Vector3 velocity = new Vector3(NN.Model.GetOutput(start + i*JointDimOut + 9), NN.Model.GetOutput(start + i*JointDimOut + 10), NN.Model.GetOutput(start + i*JointDimOut + 11)).GetRelativeDirectionFrom(currentRoot);

			Positions[i] = Vector3.Lerp(Positions[i] + velocity / Framerate, position, 0.5f);
			Forwards[i] = forward;
			Ups[i] = up;
			Velocities[i] = velocity;
		}
		start += JointDimOut*Joints.Length;
		
		//Assign Posture
		transform.position = nextRoot.GetPosition();
		transform.rotation = nextRoot.GetRotation();
		for(int i=0; i<Joints.Length; i++) {
			Joints[i].position = Positions[i];
			Joints[i].rotation = Quaternion.LookRotation(Forwards[i], Ups[i]);
		}
	}

	private void EditMotion() {

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
		GUI.color = UltiDraw.Mustard;
		GUI.backgroundColor = UltiDraw.Black;
		float height = 0.05f;
		GUI.Box(Utility.GetGUIRect(0.025f, 0.05f, 0.3f, Controller.Styles.Length*height), "");
		for(int i=0; i<Controller.Styles.Length; i++) {
			GUI.Label(Utility.GetGUIRect(0.05f, 0.075f + i*0.05f, 0.025f, height), Controller.Styles[i].Name);
			string keys = string.Empty;
			for(int j=0; j<Controller.Styles[i].Keys.Length; j++) {
				keys += Controller.Styles[i].Keys[j].ToString() + " ";
			}
			GUI.Label(Utility.GetGUIRect(0.075f, 0.075f + i*0.05f, 0.05f, height), keys);
			GUI.HorizontalSlider(Utility.GetGUIRect(0.125f, 0.075f + i*0.05f, 0.15f, height), Trajectory.Points[RootPointIndex].Styles[i], 0f, 1f);
		}
	}

	void OnRenderObject() {
		if(ShowTrajectory) {
			if(Application.isPlaying) {
				//UltiDraw.Begin();
				//UltiDraw.DrawLine(Trajectory.Points[RootPointIndex].GetPosition(), Trajectory.Points[RootPointIndex].GetPosition() + TargetDirection, 0.05f, 0f, UltiDraw.Orange.Transparent(0.75f));
				//UltiDraw.DrawLine(Trajectory.Points[RootPointIndex].GetPosition(), Trajectory.Points[RootPointIndex].GetPosition() + TargetVelocity, 0.05f, 0f, UltiDraw.Green.Transparent(0.75f));
				//UltiDraw.End();
				Trajectory.Draw(10);
			}
		}

		if(Application.isPlaying) {
			/*
			for(int i=5; i<6; i++) {
				for(int j=0; j<Character.Hierarchy.Length; j++) {
					Matrix4x4 mat = Matrix4x4.TRS(PastPositions[j][i*10], Quaternion.LookRotation(PastForwards[j][i*10], PastUps[j][i*10]), Vector3.one);
					Character.Hierarchy[j].SetTransformation(mat);
				}
				Character.DrawSimple(Color.Lerp(UltiDraw.Blue, UltiDraw.Cyan, 1f - (float)(i+1)/6f).Transparent(0.75f));
			}
			*/
			/*
			for(int i=0; i<5; i++) {
				for(int j=0; j<Character.Hierarchy.Length; j++) {
					Matrix4x4 mat = Matrix4x4.TRS(FuturePositions[j][i], Quaternion.LookRotation(FutureForwards[j][i], FutureUps[j][i]), Vector3.one);
					Character.Hierarchy[j].SetTransformation(mat);
				}
				Character.DrawSimple(Color.Lerp(UltiDraw.Red, UltiDraw.Orange, (float)(i+1)/5f).Transparent(0.75f));
			}
			*/
		}
		
		Character.FetchTransformations(transform);
		Character.Draw();

		if(ShowVelocities) {
			if(Application.isPlaying) {
				UltiDraw.Begin();
				for(int i=0; i<Joints.Length; i++) {
					Character.Segment segment = Character.FindSegment(Joints[i].name);
					if(segment != null) {
						UltiDraw.DrawArrow(
							Joints[i].position,
							Joints[i].position + Velocities[i],
							0.75f,
							0.0075f,
							0.05f,
							UltiDraw.Purple.Transparent(0.5f)
						);
					}
				}
				UltiDraw.End();
			}
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
			Target.Character.Inspector(Target.transform);
			Target.NN.Inspector();

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

		private void Inspector() {
			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				if(Target.Character.RebuildRequired(Target.transform)) {
					EditorGUILayout.HelpBox("Rebuild required because hierarchy was changed externally.", MessageType.Error);
					if(Utility.GUIButton("Build Hierarchy", Color.grey, Color.white)) {
						Target.Character.BuildHierarchy(Target.transform);
					}
				}

				if(Utility.GUIButton("Animation", UltiDraw.DarkGrey, UltiDraw.White)) {
					Target.Inspect = !Target.Inspect;
				}

				if(Target.Inspect) {
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Target.Framerate = EditorGUILayout.IntField("Framerate", Target.Framerate);
						Target.TrajectoryDimIn = EditorGUILayout.IntField("Trajectory Dim X", Target.TrajectoryDimIn);
						Target.TrajectoryDimOut = EditorGUILayout.IntField("Trajectory Dim Y", Target.TrajectoryDimOut);
						Target.JointDimIn = EditorGUILayout.IntField("Joint Dim X", Target.JointDimIn);
						Target.JointDimOut = EditorGUILayout.IntField("Joint Dim Y", Target.JointDimOut);
						Target.ShowTrajectory = EditorGUILayout.Toggle("Show Trajectory", Target.ShowTrajectory);
						Target.ShowVelocities = EditorGUILayout.Toggle("Show Velocities", Target.ShowVelocities);
						Target.TargetBlending = EditorGUILayout.Slider("Target Blending", Target.TargetBlending, 0f, 1f);
						Target.StyleTransition = EditorGUILayout.Slider("Style Transition", Target.StyleTransition, 0f, 1f);
						Target.TrajectoryControl = EditorGUILayout.Toggle("Trajectory Control", Target.TrajectoryControl);
						Target.TrajectoryCorrection = EditorGUILayout.Slider("Trajectory Correction", Target.TrajectoryCorrection, 0f, 1f);

						Target.MotionEditing = EditorGUILayout.Toggle("Motion Editing", Target.MotionEditing);
						EditorGUILayout.BeginHorizontal();
						if(Utility.GUIButton("Add IK Solver", UltiDraw.Brown, UltiDraw.White)) {
							ArrayExtensions.Expand(ref Target.IKSolvers);
						}
						if(Utility.GUIButton("Remove IK Solver", UltiDraw.Brown, UltiDraw.White)) {
							ArrayExtensions.Shrink(ref Target.IKSolvers);
						}
						EditorGUILayout.EndHorizontal();
						for(int i=0; i<Target.IKSolvers.Length; i++) {
							Target.IKSolvers[i] = (SerialCCD)EditorGUILayout.ObjectField(Target.IKSolvers[i], typeof(SerialCCD), true);
						}
					}
				}
			}
		}
	}
	#endif
}