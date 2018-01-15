using UnityEngine;
using System;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class BioAnimation_APFNN : MonoBehaviour {

	public bool Inspect = false;
	public bool ShowTrajectory = true;
	public bool ShowVelocities = true;

	public float TargetBlending = 0.25f;
	public float StyleTransition = 0.25f;
	public float TrajectoryCorrection = 1f;

	public Transform Root;
	public Transform[] Joints = new Transform[0];

	public Controller Controller;
	public Character Character;
	public APFNN_Eigen APFNN;

	public SerialIK[] IKSolvers = new SerialIK[0];

	private Trajectory Trajectory;

	private Vector3 TargetDirection;
	private Vector3 TargetVelocity;

	private Vector3[] Positions = new Vector3[0];
	private Vector3[] Forwards = new Vector3[0];
	private Vector3[] Ups = new Vector3[0];
	private Vector3[] Velocities = new Vector3[0];
	
	//Trajectory for 60 Hz framerate
	private const int PointSamples = 12;
	private const int RootPointIndex = 60;
	private const int PointDensity = 10;

	private const int JointDimIn = 12;
	private const int JointDimOut = 12;

	void Reset() {
		Root = transform;
		Controller = new Controller();
		Character = new Character();
		Character.BuildHierarchy(transform);
		APFNN = new APFNN_Eigen();
	}

	void Awake() {
		TargetDirection = new Vector3(Root.forward.x, 0f, Root.forward.z);
		TargetVelocity = Vector3.zero;
		Positions = new Vector3[Joints.Length];
		Forwards = new Vector3[Joints.Length];
		Ups = new Vector3[Joints.Length];
		Velocities = new Vector3[Joints.Length];
		Trajectory = new Trajectory(111, Controller.Styles.Length, Root.position, TargetDirection);
		Trajectory.Postprocess();
		APFNN.Initialise();
		for(int i=309; i<312; i++) {
			APFNN.AddControlNeuron(i);
		}
		for(int i=369; i<372; i++) {
			APFNN.AddControlNeuron(i);
		}
		for(int i=417; i<420; i++) {
			APFNN.AddControlNeuron(i);
		}
		for(int i=465; i<468; i++) {
			APFNN.AddControlNeuron(i);
		}
	}

	void Start() {
		Utility.SetFPS(60);
	}

	public Trajectory GetTrajectory() {
		return Trajectory;
	}

	public void AutoDetect() {
		SetJointCount(0);
		Action<Transform> recursion = null;
		recursion = new Action<Transform>((transform) => {
			if(Character.FindSegment(transform.name) != null) {
				AddJoint(transform);
			}
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i));
			}
		});
		recursion(Root);
	}

	void Update() {	
		//Update Target Direction / Velocity
		TargetDirection = Vector3.Lerp(TargetDirection, Quaternion.AngleAxis(Controller.QueryTurn()*60f, Vector3.up) * Trajectory.Points[RootPointIndex].GetDirection(), TargetBlending);
		TargetVelocity = Vector3.Lerp(TargetVelocity, (Quaternion.LookRotation(TargetDirection, Vector3.up) * Controller.QueryMove()).normalized, TargetBlending);
		
		//Update Style
		for(int i=0; i<Controller.Styles.Length; i++) {
			Trajectory.Points[RootPointIndex].Styles[i] = Utility.Interpolate(Trajectory.Points[RootPointIndex].Styles[i], Controller.Styles[i].Query(), StyleTransition);
		}
		//Only for current project
		if(Controller.Styles.Length > 0) {
			if(Controller.Styles[2].Query() == 1f) {
				Trajectory.Points[RootPointIndex].Styles[1] = 1f - Trajectory.Points[RootPointIndex].Styles[2];
			}
		}
		//

		//Predict Future Trajectory
		Vector3[] trajectory_positions_blend = new Vector3[Trajectory.Points.Length];
		trajectory_positions_blend[RootPointIndex] = Trajectory.Points[RootPointIndex].GetTransformation().GetPosition();
		for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
			float bias_pos = 0.75f;
			float bias_dir = 1.25f;
			float scale_pos = (1.0f - Mathf.Pow(1.0f - ((float)(i - RootPointIndex) / (RootPointIndex)), bias_pos));
			float scale_dir = (1.0f - Mathf.Pow(1.0f - ((float)(i - RootPointIndex) / (RootPointIndex)), bias_dir));
			float vel_boost = 2f;
			
			float rescale = 1f / (Trajectory.Points.Length - (RootPointIndex + 1f));

			trajectory_positions_blend[i] = trajectory_positions_blend[i-1] + Vector3.Lerp(
				Trajectory.Points[i].GetPosition() - Trajectory.Points[i-1].GetPosition(), 
				vel_boost * rescale * TargetVelocity,
				scale_pos);

			Trajectory.Points[i].SetDirection(Vector3.Lerp(Trajectory.Points[i].GetDirection(), TargetDirection, scale_dir));

			for(int j=0; j<Trajectory.Points[i].Styles.Length; j++) {
				Trajectory.Points[i].Styles[j] = Trajectory.Points[RootPointIndex].Styles[j];
			}
		}
		for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
			Trajectory.Points[i].SetPosition(trajectory_positions_blend[i]);
		}

		for(int i=RootPointIndex; i<Trajectory.Points.Length; i+=PointDensity) {
			Trajectory.Points[i].Postprocess();
		}
		for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
			Trajectory.Point prev = GetPreviousSample(i);
			Trajectory.Point next = GetNextSample(i);
			float factor = (float)(i % PointDensity) / PointDensity;

			Trajectory.Points[i].SetPosition(((1f-factor)*prev.GetPosition() + factor*next.GetPosition()));
			Trajectory.Points[i].SetDirection(((1f-factor)*prev.GetDirection() + factor*next.GetDirection()));
			Trajectory.Points[i].SetLeftsample((1f-factor)*prev.GetLeftSample() + factor*next.GetLeftSample());
			Trajectory.Points[i].SetRightSample((1f-factor)*prev.GetRightSample() + factor*next.GetRightSample());
			Trajectory.Points[i].SetRise((1f-factor)*prev.GetRise() + factor*next.GetRise());
		}

		if(APFNN.Parameters != null) {
			//Calculate Root
			Matrix4x4 currentRoot = Trajectory.Points[RootPointIndex].GetTransformation();
			Matrix4x4 previousRoot = Trajectory.Points[RootPointIndex-1].GetTransformation();
			//Fix for flat terrain
			Transformations.SetPosition(
				ref previousRoot, 
				previousRoot.GetPosition() - new Vector3(0f, currentRoot.GetPosition().y - previousRoot.GetPosition().y, 0f)
				);
			
			//Input Trajectory Positions / Directions
			int start = 0;
			for(int i=0; i<PointSamples; i++) {
				Vector3 pos = GetSample(i).GetPosition().GetRelativePositionTo(currentRoot);
				Vector3 dir = GetSample(i).GetDirection().GetRelativeDirectionTo(currentRoot);
				APFNN.SetInput(start + i*6 + 0, pos.x);
				//APFNN.SetInput(start + i*6 + 1, pos.y);
				APFNN.SetInput(start + i*6 + 1, 0f);
				APFNN.SetInput(start + i*6 + 2, pos.z);
				APFNN.SetInput(start + i*6 + 3, dir.x);
				APFNN.SetInput(start + i*6 + 4, 0f);
				APFNN.SetInput(start + i*6 + 5, dir.z);
			}
			start += 6*PointSamples;

			//Input Trajectory Heights
			for(int i=0; i<PointSamples; i++) {
				//APFNN.SetInput(start + i*2 + 0, GetSample(i).GetLeftSample().y - currentRoot.GetPosition().y);
				//APFNN.SetInput(start + i*2 + 1, GetSample(i).GetRightSample().y - currentRoot.GetPosition().y);
				APFNN.SetInput(start + i*2 + 0, 0f);
				APFNN.SetInput(start + i*2 + 1, 0f);
			}
			start += 2*PointSamples;

			//Input Trajectory Styles
			for (int i=0; i<PointSamples; i++) {
				for(int j=0; j<GetSample(i).Styles.Length; j++) {
					APFNN.SetInput(start + (i*GetSample(i).Styles.Length) + j, GetSample(i).Styles[j]);
				}
			}
			start += Controller.Styles.Length * PointSamples;
			//180

			//Input Previous Bone Positions / Velocities
			for(int i=0; i<Joints.Length; i++) {
				Vector3 pos = Positions[i].GetRelativePositionTo(previousRoot);
				Vector3 forward = Forwards[i].GetRelativeDirectionTo(previousRoot);
				Vector3 up = Ups[i].GetRelativeDirectionTo(previousRoot);
				Vector3 vel = Velocities[i].GetRelativeDirectionTo(previousRoot);
				APFNN.SetInput(start + i*JointDimIn + 0, pos.x);
				APFNN.SetInput(start + i*JointDimIn + 1, pos.y);
				APFNN.SetInput(start + i*JointDimIn + 2, pos.z);
				APFNN.SetInput(start + i*JointDimIn + 3, forward.x);
				APFNN.SetInput(start + i*JointDimIn + 4, forward.y);
				APFNN.SetInput(start + i*JointDimIn + 5, forward.z);
				APFNN.SetInput(start + i*JointDimIn + 6, up.x);
				APFNN.SetInput(start + i*JointDimIn + 7, up.y);
				APFNN.SetInput(start + i*JointDimIn + 8, up.z);
				APFNN.SetInput(start + i*JointDimIn + 9, vel.x);
				APFNN.SetInput(start + i*JointDimIn + 10, vel.y);
				APFNN.SetInput(start + i*JointDimIn + 11, vel.z);
			}
			start += JointDimIn*Joints.Length;
			
			//Predict
			APFNN.Predict();

			//Update Past Trajectory
			for(int i=0; i<RootPointIndex; i++) {
				Trajectory.Points[i].SetPosition(Trajectory.Points[i+1].GetPosition());
				Trajectory.Points[i].SetDirection(Trajectory.Points[i+1].GetDirection());
				Trajectory.Points[i].SetLeftsample(Trajectory.Points[i+1].GetLeftSample());
				Trajectory.Points[i].SetRightSample(Trajectory.Points[i+1].GetRightSample());
				Trajectory.Points[i].SetRise(Trajectory.Points[i+1].GetRise());
				for(int j=0; j<Trajectory.Points[i].Styles.Length; j++) {
					Trajectory.Points[i].Styles[j] = Trajectory.Points[i+1].Styles[j];
				}
			}

			//Update Current Trajectory
			int end = 6*4 + JointDimOut*Joints.Length;
			Vector3 translationalVelocity = new Vector3(APFNN.GetOutput(end+0), 0f, APFNN.GetOutput(end+1));
			float angularVelocity = APFNN.GetOutput(end+2);
			
			Trajectory.Points[RootPointIndex].SetPosition(translationalVelocity.GetRelativePositionFrom(currentRoot));
			Trajectory.Points[RootPointIndex].SetDirection(Quaternion.AngleAxis(angularVelocity, Vector3.up) * Trajectory.Points[RootPointIndex].GetDirection());
			Trajectory.Points[RootPointIndex].Postprocess();
			Matrix4x4 nextRoot = Trajectory.Points[RootPointIndex].GetTransformation();

			//Update Future Trajectory
			for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
				Trajectory.Points[i].SetPosition(Trajectory.Points[i].GetPosition() + translationalVelocity.GetRelativeDirectionFrom(nextRoot));
			}
			start = 0;
			for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
				//ROOT	1		2		3		4		5
				//.x....x.......x.......x.......x.......x
				int index = i;
				int prevSampleIndex = GetPreviousSample(index).GetIndex() / PointDensity;
				int nextSampleIndex = GetNextSample(index).GetIndex() / PointDensity;
				float factor = (float)(i % PointDensity) / PointDensity;

				float prevPosX = APFNN.GetOutput(start + (prevSampleIndex-6)*4 + 0);
				float prevPosZ = APFNN.GetOutput(start + (prevSampleIndex-6)*4 + 1);
				float prevDirX = APFNN.GetOutput(start + (prevSampleIndex-6)*4 + 2);
				float prevDirZ = APFNN.GetOutput(start + (prevSampleIndex-6)*4 + 3);

				float nextPosX = APFNN.GetOutput(start + (nextSampleIndex-6)*4 + 0);
				float nextPosZ = APFNN.GetOutput(start + (nextSampleIndex-6)*4 + 1);
				float nextDirX = APFNN.GetOutput(start + (nextSampleIndex-6)*4 + 2);
				float nextDirZ = APFNN.GetOutput(start + (nextSampleIndex-6)*4 + 3);

				float posX = (1f - factor) * prevPosX + factor * nextPosX;
				float posZ = (1f - factor) * prevPosZ + factor * nextPosZ;
				float dirX = (1f - factor) * prevDirX + factor * nextDirX;
				float dirZ = (1f - factor) * prevDirZ + factor * nextDirZ;

				Trajectory.Points[i].SetPosition(
					Utility.Interpolate(
						Trajectory.Points[i].GetPosition(),
						new Vector3(posX, 0f, posZ).GetRelativePositionFrom(nextRoot),
						TrajectoryCorrection
						)
					);
				Trajectory.Points[i].SetDirection(
					Utility.Interpolate(
						Trajectory.Points[i].GetDirection(),
						new Vector3(dirX, 0f, dirZ).normalized.GetRelativeDirectionFrom(nextRoot),
						TrajectoryCorrection
						)
					);
			}
			start += 6 * 4;
			for(int i=RootPointIndex+PointDensity; i<Trajectory.Points.Length; i+=PointDensity) {
				Trajectory.Points[i].Postprocess();
			}
			for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
				Trajectory.Point prev = GetPreviousSample(i);
				Trajectory.Point next = GetNextSample(i);
				float factor = (float)(i % PointDensity) / PointDensity;

				Trajectory.Points[i].SetPosition(((1f-factor)*prev.GetPosition() + factor*next.GetPosition()));
				Trajectory.Points[i].SetDirection(((1f-factor)*prev.GetDirection() + factor*next.GetDirection()));
				Trajectory.Points[i].SetLeftsample((1f-factor)*prev.GetLeftSample() + factor*next.GetLeftSample());
				Trajectory.Points[i].SetRightSample((1f-factor)*prev.GetRightSample() + factor*next.GetRightSample());
				Trajectory.Points[i].SetRise((1f-factor)*prev.GetRise() + factor*next.GetRise());
			}

			//Compute Posture
			for(int i=0; i<Joints.Length; i++) {
				Vector3 position = new Vector3(APFNN.GetOutput(start + i*JointDimOut + 0), APFNN.GetOutput(start + i*JointDimOut + 1), APFNN.GetOutput(start + i*JointDimOut + 2));
				Vector3 forward = new Vector3(APFNN.GetOutput(start + i*JointDimOut + 3), APFNN.GetOutput(start + i*JointDimOut + 4), APFNN.GetOutput(start + i*JointDimOut + 5)).normalized;
				Vector3 up = new Vector3(APFNN.GetOutput(start + i*JointDimOut + 6), APFNN.GetOutput(start + i*JointDimOut + 7), APFNN.GetOutput(start + i*JointDimOut + 8)).normalized;
				Vector3 velocity = new Vector3(APFNN.GetOutput(start + i*JointDimOut + 9), APFNN.GetOutput(start + i*JointDimOut + 10), APFNN.GetOutput(start + i*JointDimOut + 11));
				/*
				if(i==0 || i==1) {
					position.x = 0f;
					position.z = 0f;
					velocity.x = 0f;
					velocity.z = 0f;
				}
				if(i==3) {
					position.x = 0f;
					velocity.x = 0f;
				}
				*/
				Positions[i] = Vector3.Lerp(Positions[i].GetRelativePositionTo(currentRoot) + velocity, position, 0.5f).GetRelativePositionFrom(currentRoot);
				Forwards[i] = forward.GetRelativeDirectionFrom(currentRoot);
				Ups[i] = up.GetRelativeDirectionFrom(currentRoot);
				Velocities[i] = velocity.GetRelativeDirectionFrom(currentRoot);
			}
			start += JointDimOut*Joints.Length;
			
			//Update Posture
			Root.position = nextRoot.GetPosition();
			Root.rotation = nextRoot.GetRotation();
			for(int i=0; i<Joints.Length; i++) {
				Joints[i].position = Positions[i];
				Joints[i].rotation = Quaternion.LookRotation(Forwards[i], Ups[i]);
			}
			
			//Motion Editing
			for(int i=0; i<IKSolvers.Length; i++) {
				IKSolvers[i].UpdateGoal();
			}

			Transform spine = Array.Find(Joints, x => x.name == "Spine1");
			Transform neck = Array.Find(Joints, x => x.name == "Neck");
			Transform leftShoulder = Array.Find(Joints, x => x.name == "LeftShoulder");
			Transform rightShoulder = Array.Find(Joints, x => x.name == "RightShoulder");
			Vector3 spinePosition = spine.position;
			Vector3 neckPosition = neck.position;
			Vector3 leftShoulderPosition = leftShoulder.position;
			Vector3 rightShoulderPosition = rightShoulder.position;
			float spineHeight = Utility.GetHeight(spine.position, LayerMask.GetMask("Ground"));
			float neckHeight = Utility.GetHeight(neck.position, LayerMask.GetMask("Ground"));
			float leftShoulderHeight = Utility.GetHeight(leftShoulder.position, LayerMask.GetMask("Ground"));
			float rightShoulderHeight = Utility.GetHeight(rightShoulder.position, LayerMask.GetMask("Ground"));

			spine.rotation = Quaternion.Slerp(spine.rotation, Quaternion.FromToRotation(neckPosition - spinePosition, new Vector3(neckPosition.x, neckHeight + (neckPosition.y - Root.position.y), neckPosition.z) - spinePosition) * spine.rotation, 0.5f);

			spine.position = new Vector3(spinePosition.x, spineHeight + (spinePosition.y - Root.position.y), spinePosition.z);
			neck.position = new Vector3(neckPosition.x, neckHeight + (neckPosition.y - Root.position.y), neckPosition.z);
			leftShoulder.position = new Vector3(leftShoulderPosition.x, leftShoulderHeight + (leftShoulderPosition.y - Root.position.y), leftShoulderPosition.z);
			rightShoulder.position = new Vector3(rightShoulderPosition.x, rightShoulderHeight + (rightShoulderPosition.y - Root.position.y), rightShoulderPosition.z);

			for(int i=0; i<IKSolvers.Length; i++) {
				IKSolvers[i].ProcessIK();
			}

			//Update Skeleton
			Character.FetchTransformations(Root);		
		}
	}

	private int GetJointIndex(string name) {
		return System.Array.FindIndex(Joints, x => x.name == name);
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

	public void AddJoint(Transform joint) {
		System.Array.Resize(ref Joints, Joints.Length+1);
		Joints[Joints.Length-1] = joint;
	}

	public void SetJoint(int index, Transform t) {
		if(index < 0 || index >= Joints.Length) {
			return;
		}
		Joints[index] = t;
	}

	public void SetJointCount(int count) {
		count = Mathf.Max(0, count);
		if(Joints.Length != count) {
			System.Array.Resize(ref Joints, count);
		}
	}

	void OnGUI() {
		float height = 0.05f;
		GUI.Box(Utility.GetGUIRect(0.7f, 0.025f, 0.3f, Controller.Styles.Length*height), "");
		for(int i=0; i<Controller.Styles.Length; i++) {
			GUI.Label(Utility.GetGUIRect(0.725f, 0.05f + i*0.05f, 0.025f, height), Controller.Styles[i].Name);
			string keys = string.Empty;
			for(int j=0; j<Controller.Styles[i].Keys.Length; j++) {
				keys += Controller.Styles[i].Keys[j].ToString() + " ";
			}
			GUI.Label(Utility.GetGUIRect(0.75f, 0.05f + i*0.05f, 0.05f, height), keys);
			GUI.HorizontalSlider(Utility.GetGUIRect(0.8f, 0.05f + i*0.05f, 0.15f, height), Trajectory.Points[RootPointIndex].Styles[i], 0f, 1f);
		}
	}

	void OnRenderObject() {
		if(Root == null) {
			Root = transform;
		}

		if(ShowTrajectory) {
			if(Application.isPlaying) {
				UnityGL.Start();
				UnityGL.DrawLine(Trajectory.Points[RootPointIndex].GetPosition(), Trajectory.Points[RootPointIndex].GetPosition() + TargetDirection, 0.05f, 0f, new Color(Utility.Red.r, Utility.Red.g, Utility.Red.b, 0.75f));
				UnityGL.DrawLine(Trajectory.Points[RootPointIndex].GetPosition(), Trajectory.Points[RootPointIndex].GetPosition() + TargetVelocity, 0.05f, 0f, new Color(Utility.Green.r, Utility.Green.g, Utility.Green.b, 0.75f));
				UnityGL.Finish();
				Trajectory.Draw(10);
			}
		}
		
		if(!Application.isPlaying) {
			Character.FetchTransformations(Root);
		}
		Character.Draw();

		if(ShowVelocities) {
			if(Application.isPlaying) {
				UnityGL.Start();
				for(int i=0; i<Joints.Length; i++) {
					Character.Segment segment = Character.FindSegment(Joints[i].name);
					if(segment != null) {
						UnityGL.DrawArrow(
							Joints[i].position,
							Joints[i].position + Velocities[i]/Time.deltaTime,
							0.75f,
							0.0075f,
							0.05f,
							new Color(0f, 1f, 1f, 0.5f)
						);
					}
				}
				UnityGL.Finish();
			}
		}
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			OnRenderObject();
		}
	}

	#if UNITY_EDITOR
	[CustomEditor(typeof(BioAnimation_APFNN))]
	public class BioAnimation_APFNN_Editor : Editor {

		public BioAnimation_APFNN Target;

		void Awake() {
			Target = (BioAnimation_APFNN)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

			Inspector();
			Target.Controller.Inspector();
			Target.Character.Inspector(Target.Root);
			Target.APFNN.Inspector();

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

		private void Inspector() {			
			Utility.SetGUIColor(Utility.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				if(Target.Character.RebuildRequired(Target.Root)) {
					EditorGUILayout.HelpBox("Rebuild required because hierarchy was changed externally.", MessageType.Error);
					if(Utility.GUIButton("Build Hierarchy", Color.grey, Color.white)) {
						Target.Character.BuildHierarchy(Target.Root);
					}
				}

				/*
				if(Utility.GUIButton("Rebuild Hierarchy", Utility.DarkGreen, Utility.White)) {
					Target.Character.BuildHierarchy(Target.Root);
				}
				if(Utility.GUIButton("Clear Hierarchy", Utility.DarkRed, Utility.White)) {
					Utility.Clear(ref Target.Character.Hierarchy);
				}
				*/

				if(Utility.GUIButton("Animation", Utility.DarkGrey, Utility.White)) {
					Target.Inspect = !Target.Inspect;
				}

				if(Target.Inspect) {
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Target.ShowTrajectory = EditorGUILayout.Toggle("Show Trajectory", Target.ShowTrajectory);
						Target.ShowVelocities = EditorGUILayout.Toggle("Show Velocities", Target.ShowVelocities);
						Target.TargetBlending = EditorGUILayout.Slider("Target Blending", Target.TargetBlending, 0f, 1f);
						Target.StyleTransition = EditorGUILayout.Slider("Style Transition", Target.StyleTransition, 0f, 1f);
						Target.TrajectoryCorrection = EditorGUILayout.Slider("Trajectory Correction", Target.TrajectoryCorrection, 0f, 1f);

						EditorGUILayout.BeginHorizontal();
						if(Utility.GUIButton("Add IK Solver", Utility.Brown, Utility.White)) {
							Utility.Expand(ref Target.IKSolvers);
						}
						if(Utility.GUIButton("Remove IK Solver", Utility.Brown, Utility.White)) {
							Utility.Shrink(ref Target.IKSolvers);
						}
						EditorGUILayout.EndHorizontal();
						for(int i=0; i<Target.IKSolvers.Length; i++) {
							Target.IKSolvers[i] = (SerialIK)EditorGUILayout.ObjectField(Target.IKSolvers[i], typeof(SerialIK), true);
						}

						EditorGUI.BeginDisabledGroup(true);
						EditorGUILayout.ObjectField("Root", Target.Root, typeof(Transform), true);
						EditorGUI.EndDisabledGroup();
						Target.SetJointCount(EditorGUILayout.IntField("Joint Count", Target.Joints.Length));
						if(Utility.GUIButton("Auto Detect", Utility.DarkGrey, Utility.White)) {
							Target.AutoDetect();
						}
						for(int i=0; i<Target.Joints.Length; i++) {
							if(Target.Joints[i] != null) {
								Utility.SetGUIColor(Utility.Green);
							} else {
								Utility.SetGUIColor(Utility.Red);
							}
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField("Joint " + (i+1), GUILayout.Width(50f));
							Target.SetJoint(i, (Transform)EditorGUILayout.ObjectField(Target.Joints[i], typeof(Transform), true));
							EditorGUILayout.EndHorizontal();
							Utility.ResetGUIColor();
						}
					}
				}
			}
		}
	}
	#endif
}