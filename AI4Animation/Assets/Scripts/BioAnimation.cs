using UnityEngine;

public class BioAnimation : MonoBehaviour {

	public bool Inspect = false;

	public float TargetBlending = 0.25f;
	public float GaitTransition = 0.25f;
	public float TrajectoryCorrection = 0.75f;

	public Transform Root;
	public Transform[] Joints = new Transform[0];

	public Controller Controller;
	public Character Character;
	public PFNN PFNN;

	private Trajectory Trajectory;

	private float Phase = 0f;
	private Vector3 TargetDirection;
	private Vector3 TargetVelocity;
	private Vector3[] Velocities = new Vector3[0];
	
	//Trajectory for 60 Hz framerate
	private const int PointSamples = 12;
	private const int RootPointIndex = 60;
	private const int PointDensity = 10;

	void Reset() {
		Root = transform;
		Controller = new Controller();
		Character = new Character();
		Character.BuildHierarchy(transform);
		PFNN = new PFNN();
	}

	void Awake() {
		TargetDirection = new Vector3(Root.forward.x, 0f, Root.forward.z);
		TargetVelocity = Vector3.zero;
		Velocities = new Vector3[Joints.Length];
		Trajectory = new Trajectory(111, Controller.Styles.Length, Root.position, TargetDirection);
		Trajectory.Postprocess();
		PFNN.Initialise();
	}

	void Start() {
		Utility.SetFPS(60);
	}

	void Update() {	
		//Update Target Direction / Velocity
		TargetDirection = Vector3.Lerp(TargetDirection, Quaternion.AngleAxis(Controller.QueryTurn()*60f, Vector3.up) * Trajectory.Points[RootPointIndex].GetDirection(), TargetBlending);
		TargetVelocity = Vector3.Lerp(TargetVelocity, (Quaternion.LookRotation(TargetDirection, Vector3.up) * Controller.QueryMove()).normalized, TargetBlending);
		
		//Update Gait
		for(int i=0; i<Controller.Styles.Length; i++) {
			Trajectory.Points[RootPointIndex].Styles[i] = Utility.Interpolate(Trajectory.Points[RootPointIndex].Styles[i], Controller.Styles[i].Query(), GaitTransition);
		}

		//Predict Future Trajectory
		Vector3[] trajectory_positions_blend = new Vector3[Trajectory.Points.Length];
		trajectory_positions_blend[RootPointIndex] = Trajectory.Points[RootPointIndex].GetPosition();
		for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
			float bias_pos = 0.75f;
			float bias_dir = 1.25f;
			float scale_pos = (1.0f - Mathf.Pow(1.0f - ((float)(i - RootPointIndex) / (RootPointIndex)), bias_pos));
			float scale_dir = (1.0f - Mathf.Pow(1.0f - ((float)(i - RootPointIndex) / (RootPointIndex)), bias_dir));
			float vel_boost = 1f;
			
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

		if(PFNN.Parameters != null) {
			//Calculate Root
			Transformation currentRoot = Trajectory.Points[RootPointIndex].GetTransformation();
			Transformation previousRoot = Trajectory.Points[RootPointIndex-1].GetTransformation();
			
			//Input Trajectory Positions / Directions
			int start = 0;
			for(int i=0; i<PointSamples; i++) {
				Vector3 pos = GetSample(i).GetPosition().RelativePositionTo(currentRoot);
				Vector3 dir = GetSample(i).GetDirection().RelativeDirectionTo(currentRoot);
				PFNN.SetInput(start + i*6 + 0, pos.x);
				PFNN.SetInput(start + i*6 + 1, pos.y);
				PFNN.SetInput(start + i*6 + 2, pos.z);
				PFNN.SetInput(start + i*6 + 3, dir.x);
				PFNN.SetInput(start + i*6 + 4, dir.y);
				PFNN.SetInput(start + i*6 + 5, dir.z);
			}
			start += 6*PointSamples;

			//Input Trajectory Heights
			for(int i=0; i<PointSamples; i++) {
				PFNN.SetInput(start + i*2 + 0, GetSample(i).GetLeftSample().y - currentRoot.Position.y);
				PFNN.SetInput(start + i*2 + 1, GetSample(i).GetRightSample().y - currentRoot.Position.y);
			}
			start += 2*PointSamples;

			//Input Trajectory Gaits
			for (int i=0; i<PointSamples; i++) {
				for(int j=0; j<GetSample(i).Styles.Length; j++) {
					PFNN.SetInput(start + (i*GetSample(i).Styles.Length) + j, GetSample(i).Styles[j]);
				}
			}
			start += Controller.Styles.Length * PointSamples;

			//Input Previous Bone Positions / Velocities
			for(int i=0; i<Joints.Length; i++) {
				Vector3 pos = Joints[i].position.RelativePositionTo(previousRoot);
				Vector3 vel = Velocities[i].RelativeDirectionTo(previousRoot);
				PFNN.SetInput(start + i*6 + 0, pos.x);
				PFNN.SetInput(start + i*6 + 1, pos.y);
				PFNN.SetInput(start + i*6 + 2, pos.z);
				PFNN.SetInput(start + i*6 + 3, vel.x);
				PFNN.SetInput(start + i*6 + 4, vel.y);
				PFNN.SetInput(start + i*6 + 5, vel.z);
			}
			start += 6*Joints.Length;

			//Predict
			PFNN.Predict(Phase);

			//Update Past Trajectory
			for(int i=0; i<RootPointIndex; i++) {
				Trajectory.Points[i].Position = Trajectory.Points[i+1].Position;
				Trajectory.Points[i].Direction = Trajectory.Points[i+1].Direction;
				Trajectory.Points[i].RightSample = Trajectory.Points[i+1].RightSample;
				Trajectory.Points[i].LeftSample = Trajectory.Points[i+1].LeftSample;
				Trajectory.Points[i].Rise = Trajectory.Points[i+1].Rise;
				for(int j=0; j<Trajectory.Points[i].Styles.Length; j++) {
					Trajectory.Points[i].Styles[j] = Trajectory.Points[i+1].Styles[j];
				}
			}

			//Update Current Trajectory
			int end = 6 * 4 + Joints.Length * 6;
			Vector3 translationalVelocity = new Vector3(PFNN.GetOutput(end+0), 0f, PFNN.GetOutput(end+1));
			float angularVelocity = PFNN.GetOutput(end+2);
			float rest = Mathf.Pow(1.0f-Trajectory.Points[RootPointIndex].Styles[0], 0.25f);
			rest = Mathf.Min(rest, Mathf.Pow(1.0f-Trajectory.Points[RootPointIndex].Styles[5], 0.25f));
			rest = Mathf.Min(rest, Mathf.Pow(1.0f-Trajectory.Points[RootPointIndex].Styles[6], 0.25f));
			rest = Mathf.Min(rest, Mathf.Pow(1.0f-Trajectory.Points[RootPointIndex].Styles[7], 0.25f));
			rest = Mathf.Min(rest, Mathf.Pow(1.0f-Trajectory.Points[RootPointIndex].Styles[8], 0.25f));
			Trajectory.Points[RootPointIndex].SetPosition((rest * translationalVelocity).RelativePositionFrom(currentRoot));
			Trajectory.Points[RootPointIndex].SetDirection(Quaternion.AngleAxis(rest * angularVelocity, Vector3.up) * Trajectory.Points[RootPointIndex].GetDirection());
			Trajectory.Points[RootPointIndex].Postprocess();
			Transformation nextRoot = Trajectory.Points[RootPointIndex].GetTransformation();

			//Update Future Trajectory
			for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
				Trajectory.Points[i].SetPosition(Trajectory.Points[i].GetPosition() + (rest * translationalVelocity).RelativeDirectionFrom(nextRoot));
			}
			start = 0;
			for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
				//ROOT	1		2		3		4		5
				//.x....x.......x.......x.......x.......x
				int index = i;
				int prevSampleIndex = GetPreviousSample(index).GetIndex() / PointDensity;
				int nextSampleIndex = GetNextSample(index).GetIndex() / PointDensity;
				float factor = (float)(i % PointDensity) / PointDensity;

				float prevPosX = PFNN.GetOutput(start + (prevSampleIndex-6)*4 + 0);
				float prevPosZ = PFNN.GetOutput(start + (prevSampleIndex-6)*4 + 1);
				float prevDirX = PFNN.GetOutput(start + (prevSampleIndex-6)*4 + 2);
				float prevDirZ = PFNN.GetOutput(start + (prevSampleIndex-6)*4 + 3);

				float nextPosX = PFNN.GetOutput(start + (nextSampleIndex-6)*4 + 0);
				float nextPosZ = PFNN.GetOutput(start + (nextSampleIndex-6)*4 + 1);
				float nextDirX = PFNN.GetOutput(start + (nextSampleIndex-6)*4 + 2);
				float nextDirZ = PFNN.GetOutput(start + (nextSampleIndex-6)*4 + 3);

				float posX = (1f - factor) * prevPosX + factor * nextPosX;
				float posZ = (1f - factor) * prevPosZ + factor * nextPosZ;
				float dirX = (1f - factor) * prevDirX + factor * nextDirX;
				float dirZ = (1f - factor) * prevDirZ + factor * nextDirZ;

				Trajectory.Points[i].SetPosition(
					Utility.Interpolate(
						Trajectory.Points[i].GetPosition(),
						new Vector3(posX, 0f, posZ).RelativePositionFrom(nextRoot),
						TrajectoryCorrection
						)
					);
				Trajectory.Points[i].SetDirection(
					Utility.Interpolate(
						Trajectory.Points[i].GetDirection(),
						new Vector3(dirX, 0f, dirZ).normalized.RelativeDirectionFrom(nextRoot),
						TrajectoryCorrection
						)
					);
			}
			start += 6 * 4;
			for(int i=RootPointIndex+PointDensity; i<Trajectory.Points.Length; i+=PointDensity) {
				Trajectory.Points[i].Postprocess();
			}

			//Compute Posture
			Vector3[] positions = new Vector3[Joints.Length];
			for(int i=0; i<Joints.Length; i++) {			
				Vector3 position = new Vector3(PFNN.GetOutput(start + i*6 + 0), PFNN.GetOutput(start + i*6 + 1), PFNN.GetOutput(start + i*6 + 2));
				Vector3 velocity = new Vector3(PFNN.GetOutput(start + i*6 + 3), PFNN.GetOutput(start + i*6 + 4), PFNN.GetOutput(start + i*6 + 5));
				positions[i] = Vector3.Lerp(Joints[i].position.RelativePositionTo(currentRoot) + velocity/60f, position, 0.5f).RelativePositionFrom(currentRoot);
				Velocities[i] = velocity.RelativeDirectionFrom(currentRoot);
			}
			start += Joints.Length * 6;
			
			//Update Posture
			Root.position = nextRoot.Position;
			Root.rotation = nextRoot.Rotation;
			for(int i=0; i<Joints.Length; i++) {
				Joints[i].position = positions[i];
			}

			//Map to Character
			Character.ForwardKinematics(Root);

			/* Update Phase */
			Phase = Mathf.Repeat(Phase + PFNN.GetOutput(end+3) * 2f*Mathf.PI, 2f*Mathf.PI);
		}
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
		GUI.HorizontalSlider(Utility.GetGUIRect(0.45f, 0.1f, 0.1f, height), Phase, 0f, 2f*Mathf.PI);
		GUI.Box(Utility.GetGUIRect(0.725f, 0.025f, 0.25f, Controller.Styles.Length*height), "");
		for(int i=0; i<Controller.Styles.Length; i++) {
			GUI.Label(Utility.GetGUIRect(0.75f, 0.05f + i*0.05f, 0.05f, height), Controller.Styles[i].Name);
			GUI.HorizontalSlider(Utility.GetGUIRect(0.8f, 0.05f + i*0.05f, 0.15f, height), Trajectory.Points[RootPointIndex].Styles[i], 0f, 1f);
		}
	}

	void OnRenderObject() {
		if(Application.isPlaying) {
			UnityGL.Start();
			UnityGL.DrawLine(Trajectory.Points[RootPointIndex].GetPosition(), Trajectory.Points[RootPointIndex].GetPosition() + TargetDirection, 0.05f, 0f, new Color(Utility.Red.r, Utility.Red.g, Utility.Red.b, 0.75f));
			UnityGL.DrawLine(Trajectory.Points[RootPointIndex].GetPosition(), Trajectory.Points[RootPointIndex].GetPosition() + TargetVelocity, 0.05f, 0f, new Color(Utility.Green.r, Utility.Green.g, Utility.Green.b, 0.75f));
			UnityGL.Finish();
			Trajectory.Draw(10);
		}
		
		if(!Application.isPlaying) {
			Character.ForwardKinematics(Root);
		}
		Character.Draw();

		if(Application.isPlaying) {
			UnityGL.Start();
			for(int i=0; i<Joints.Length; i++) {
				Character.Bone bone = Character.FindBone(Joints[i].name);
				if(bone != null) {
					if(bone.Draw) {
						UnityGL.DrawArrow(
							Joints[i].position,
							Joints[i].position + Velocities[i],
							0.75f,
							0.0075f,
							0.05f,
							new Color(0f, 1f, 0f, 0.5f)
						);
					}
				}
			}
			UnityGL.Finish();
		}
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			OnRenderObject();
		}
	}
	
}
