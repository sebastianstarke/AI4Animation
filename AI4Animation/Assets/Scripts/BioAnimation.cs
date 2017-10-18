using UnityEngine;

public class BioAnimation : MonoBehaviour {

	public Controller Controller;
	public Character Character;
	public Trajectory Trajectory;
	public PFNN PFNN;

	private float extra_direction_smooth = 0.9f;
	private float extra_velocity_smooth = 0.9f;
	private float extra_strafe_smooth = 0.9f;
	private float extra_crouched_smooth = 0.9f;
	private float extra_gait_smooth = 0.1f;
	private float extra_joint_smooth = 0.5f;

	private float UnitScale = 100f;

	private float Phase = 0f;

	private enum DrawingMode {Scene, Game};

	void Awake() {
		#if UNITY_EDITOR
		QualitySettings.vSyncCount = 0;
		#endif
		Application.targetFrameRate = 60;
		Trajectory.Initialise(transform.position, transform.forward);
		PFNN.Initialise();
	}

	void Start() {

	}

	void Update() {
		if(PFNN.Parameters == null) {
			return;
		}
		
		//Update Target Direction / Velocity
		Trajectory.UpdateTarget(Controller.QueryMove(), Controller.QueryTurn());

		//TODO: Update strafe etc.
		
		//Update Gait
		if(Vector3.Magnitude(Trajectory.TargetVelocity) < 0.1f) {
			float standAmount = 1.0f - Mathf.Clamp(Vector3.Magnitude(Trajectory.TargetVelocity) / 0.1f, 0.0f, 1.0f);
			Trajectory.GetCurrent().Stand = Utility.Interpolate(Trajectory.GetCurrent().Stand, standAmount, extra_gait_smooth);
			Trajectory.GetCurrent().Walk = Utility.Interpolate(Trajectory.GetCurrent().Walk, 0f, extra_gait_smooth);
			Trajectory.GetCurrent().Jog = Utility.Interpolate(Trajectory.GetCurrent().Jog, 0f, extra_gait_smooth);
			Trajectory.GetCurrent().Crouch = Utility.Interpolate(Trajectory.GetCurrent().Crouch, Controller.QueryCrouch(), extra_gait_smooth);
			Trajectory.GetCurrent().Jump = Utility.Interpolate(Trajectory.GetCurrent().Jump, 0f, extra_gait_smooth);
			Trajectory.GetCurrent().Bump = Utility.Interpolate(Trajectory.GetCurrent().Bump, 0f, extra_gait_smooth);
		} else {
			float standAmount = 1.0f - Mathf.Clamp(Vector3.Magnitude(Trajectory.TargetVelocity) / 0.1f, 0.0f, 1.0f);
			Trajectory.GetCurrent().Stand = Utility.Interpolate(Trajectory.GetCurrent().Stand, standAmount, extra_gait_smooth);
			Trajectory.GetCurrent().Walk = Utility.Interpolate(Trajectory.GetCurrent().Walk, 1f-Controller.QueryJog(), extra_gait_smooth);
			Trajectory.GetCurrent().Jog = Utility.Interpolate(Trajectory.GetCurrent().Jog, Controller.QueryJog(), extra_gait_smooth);
			Trajectory.GetCurrent().Crouch = Utility.Interpolate(Trajectory.GetCurrent().Crouch, Controller.QueryCrouch(), extra_gait_smooth);
			Trajectory.GetCurrent().Jump = Utility.Interpolate(Trajectory.GetCurrent().Jump, 0f, extra_gait_smooth);
			Trajectory.GetCurrent().Bump = Utility.Interpolate(Trajectory.GetCurrent().Bump, 0f, extra_gait_smooth);
		}
		//TODO: Update for jog, crouch, ...

		//Blend Trajectory Offset
		//Vector3 positionOffset = transform.position - Trajectory.GetCurrent().GetPosition();
		//for(int i=Trajectory.Size/2; i<Trajectory.Size; i++) {
		//	float factor = 1f - (i - Trajectory.Size/2)/(Trajectory.Size/2 - 1f);
		//	Trajectory.Points[i].SetPosition(Trajectory.Points[i].GetPosition() + factor*positionOffset);
		//}

		//Predict Future Trajectory
		Vector3[] trajectory_positions_blend = new Vector3[Trajectory.Size];
		trajectory_positions_blend[Trajectory.Size/2] = Trajectory.GetCurrent().GetPosition();

		for(int i=Trajectory.Size/2+1; i<Trajectory.Size; i++) {
			float bias_pos = 0.75f;
			float bias_dir = 1.25f;
			float scale_pos = (1.0f - Mathf.Pow(1.0f - ((float)(i - Trajectory.Size/2) / (Trajectory.Size/2)), bias_pos));
			float scale_dir = (1.0f - Mathf.Pow(1.0f - ((float)(i - Trajectory.Size/2) / (Trajectory.Size/2)), bias_dir));

			trajectory_positions_blend[i] = trajectory_positions_blend[i-1] + Vector3.Lerp(
				Trajectory.Points[i].GetPosition() - Trajectory.Points[i-1].GetPosition(), 
				Trajectory.TargetVelocity / (Trajectory.Size - (Trajectory.Size/2+1f)),
				scale_pos);
				
			Trajectory.Points[i].SetDirection(Vector3.Lerp(Trajectory.Points[i].GetDirection(), Trajectory.TargetDirection, scale_dir));

			Trajectory.Points[i].Stand = Trajectory.GetCurrent().Stand;
			Trajectory.Points[i].Walk = Trajectory.GetCurrent().Walk;
			Trajectory.Points[i].Jog = Trajectory.GetCurrent().Jog;
			Trajectory.Points[i].Crouch = Trajectory.GetCurrent().Crouch;
			Trajectory.Points[i].Jump = Trajectory.GetCurrent().Jump;
			Trajectory.Points[i].Bump = Trajectory.GetCurrent().Bump;
		}
		
		for(int i=Trajectory.Size/2+1; i<Trajectory.Size; i++) {
			Trajectory.Points[i].SetPosition(trajectory_positions_blend[i]);
		}

		//Calculate Root
		Transformation prevRoot = new Transformation(Trajectory.GetPrevious().GetPosition(), Trajectory.GetPrevious().GetRotation());
		Transformation currRoot = new Transformation(Trajectory.GetCurrent().GetPosition(), Trajectory.GetCurrent().GetRotation());

		//Input Trajectory Positions / Directions
		for(int i=0; i<Trajectory.Size; i+=10) {
			int w = (Trajectory.Size)/10;
			Vector3 pos = Trajectory.Points[i].GetPosition(currRoot);
			Vector3 dir = Trajectory.Points[i].GetDirection(currRoot);
			PFNN.SetInput((w*0)+i/10, UnitScale * pos.x);
			PFNN.SetInput((w*1)+i/10, UnitScale * pos.z);
			PFNN.SetInput((w*2)+i/10, dir.x);
			PFNN.SetInput((w*3)+i/10, dir.z);
		}

		//Input Trajectory Gaits
		for (int i=0; i<Trajectory.Size; i+=10) {
			int w = (Trajectory.Size)/10;
			PFNN.SetInput((w*4)+i/10, Trajectory.Points[i].Stand);
			PFNN.SetInput((w*5)+i/10, Trajectory.Points[i].Walk);
			PFNN.SetInput((w*6)+i/10, Trajectory.Points[i].Jog);
			PFNN.SetInput((w*7)+i/10, Trajectory.Points[i].Crouch);
			PFNN.SetInput((w*8)+i/10, Trajectory.Points[i].Jump);
			PFNN.SetInput((w*9)+i/10, Trajectory.Points[i].Bump);
		}

		//Input Joint Previous Positions / Velocities / Rotations
		for(int i=0; i<Character.Joints.Length; i++) {
			int o = (((Trajectory.Size)/10)*10);  

			Vector3 pos = Character.Joints[i].GetPosition(prevRoot);
			Vector3 vel = Character.Joints[i].GetVelocity(prevRoot);

			PFNN.SetInput(o+(Character.Joints.Length*3*0)+i*3+0, UnitScale * pos.x);
			PFNN.SetInput(o+(Character.Joints.Length*3*0)+i*3+1, UnitScale * pos.y);
			PFNN.SetInput(o+(Character.Joints.Length*3*0)+i*3+2, UnitScale * pos.z);
			PFNN.SetInput(o+(Character.Joints.Length*3*1)+i*3+0, UnitScale * vel.x);
			PFNN.SetInput(o+(Character.Joints.Length*3*1)+i*3+1, UnitScale * vel.y);
			PFNN.SetInput(o+(Character.Joints.Length*3*1)+i*3+2, UnitScale * vel.z);
		}

		//Input Trajectory Heights
		for (int i=0; i<Trajectory.Size; i+=10) {
			int o = (((Trajectory.Size)/10)*10) + Character.Joints.Length*3*2;
			int w = Trajectory.Size/10;
			PFNN.SetInput(o+(w*0)+(i/10), UnitScale * (Trajectory.Points[i].Project(Trajectory.Width/2f).y - currRoot.Position.y));
			PFNN.SetInput(o+(w*1)+(i/10), UnitScale * (Trajectory.Points[i].GetHeight() - currRoot.Position.y));
			PFNN.SetInput(o+(w*2)+(i/10), UnitScale * (Trajectory.Points[i].Project(-Trajectory.Width/2f).y - currRoot.Position.y));
		}

		//Predict
		PFNN.Predict(Phase);

		//Read Output
		int opos = 8+(((Trajectory.Size/2)/10)*4)+(Character.Joints.Length*3*0);
		int ovel = 8+(((Trajectory.Size/2)/10)*4)+(Character.Joints.Length*3*1);
		for(int i=0; i<Character.Joints.Length; i++) {			
			Vector3 position = new Vector3(PFNN.GetOutput(opos+i*3+0), PFNN.GetOutput(opos+i*3+1), PFNN.GetOutput(opos+i*3+2));
			Vector3 velocity = new Vector3(PFNN.GetOutput(ovel+i*3+0), PFNN.GetOutput(ovel+i*3+1), PFNN.GetOutput(ovel+i*3+2));

			Character.Joints[i].SetPosition(position / UnitScale, currRoot);
			Character.Joints[i].SetVelocity(velocity / UnitScale, currRoot);
		}

		//Root Position
		transform.position = Trajectory.GetCurrent().GetPosition();
		transform.rotation = Trajectory.GetCurrent().GetRotation();

		//Update Posture
		Character.ForwardKinematics();

		//Update Past Trajectory
		for(int i=0; i<Trajectory.Size/2; i++) {
			Trajectory.Points[i].SetPosition(Trajectory.Points[i+1].GetPosition());
			Trajectory.Points[i].SetDirection(Trajectory.Points[i+1].GetDirection());
			Trajectory.Points[i].Stand = Trajectory.Points[i+1].Stand;
			Trajectory.Points[i].Walk = Trajectory.Points[i+1].Walk;
			Trajectory.Points[i].Jog = Trajectory.Points[i+1].Jog;
			Trajectory.Points[i].Crouch = Trajectory.Points[i+1].Crouch;
			Trajectory.Points[i].Jump = Trajectory.Points[i+1].Jump;
			Trajectory.Points[i].Bump = Trajectory.Points[i+1].Bump;
		}

		//Update Current Trajectory
		float stand_amount = Mathf.Pow(1.0f-Trajectory.GetCurrent().Stand, 0.25f);
		Trajectory.GetCurrent().SetPosition(Trajectory.GetCurrent().GetPosition() + stand_amount * (Trajectory.GetCurrent().GetRotation() * new Vector3(PFNN.GetOutput(0) / UnitScale, 0f, PFNN.GetOutput(1) / UnitScale)));
		Trajectory.GetCurrent().SetDirection(Quaternion.AngleAxis(stand_amount * Mathf.Rad2Deg * (-PFNN.GetOutput(2)), Vector3.up) * Trajectory.GetCurrent().GetDirection());
		
		for(int i=Trajectory.Size/2+1; i<Trajectory.Size; i++) {
		//	Trajectory.Points[i].SetPosition(Trajectory.Points[i].GetPosition() + stand_amount * (Trajectory.GetCurrent().GetRotation() * new Vector3(PFNN.GetOutput(0) / UnitScale, 0f, PFNN.GetOutput(1) / UnitScale)));
		}

		
		//Update Future Trajectory
		for(int i=Trajectory.Size/2+1; i<Trajectory.Size; i++) {
			int w = (Trajectory.Size/2)/10;
			float m = Mathf.Repeat(((float)i - (Trajectory.Size/2f)) / 10.0f, 1.0f);
			float posX = (1-m) * PFNN.GetOutput(8+(w*0)+(i/10)-w) + m * PFNN.GetOutput(8+(w*0)+(i/10)-w+1);
			float posZ = (1-m) * PFNN.GetOutput(8+(w*1)+(i/10)-w) + m * PFNN.GetOutput(8+(w*1)+(i/10)-w+1);
			float dirX = (1-m) * PFNN.GetOutput(8+(w*2)+(i/10)-w) + m * PFNN.GetOutput(8+(w*2)+(i/10)-w+1);
			float dirZ = (1-m) * PFNN.GetOutput(8+(w*3)+(i/10)-w) + m * PFNN.GetOutput(8+(w*3)+(i/10)-w+1);
			Trajectory.Points[i].SetPosition(new Vector3(posX / UnitScale, 0f, posZ / UnitScale), new Transformation(Trajectory.GetCurrent().GetPosition(), Trajectory.GetCurrent().GetRotation()));
			Trajectory.Points[i].SetDirection((Trajectory.GetCurrent().GetRotation() * new Vector3(dirX, 0f, dirZ)).normalized);
		}
		
		
		/* Update Phase */
		Phase = Mathf.Repeat(Phase + (stand_amount * 0.9f + 0.1f) * PFNN.GetOutput(3) * 2f*Mathf.PI, 2f*Mathf.PI);
		
	}
	
	void OnGUI() {
		/*
		if(Camera.main == null) {
			return;
		}
		DrawCharacterSkeleton(DrawingMode.Game, transform);
		DrawCharacterJoints(DrawingMode.Game, transform);
		*/
	}

	void OnDrawGizmos() {
		DrawCharacterSkeleton(DrawingMode.Scene, transform);
		DrawCharacterJoints(DrawingMode.Scene, transform);
		DrawTrajectory();
	}

	private void DrawCharacterSkeleton(DrawingMode mode, Transform bone, Transform parent = null) {
		bool isJoint = System.Array.Find(Character.Joints, x => x.Transform == bone) != null;
		if(parent != null && isJoint) {
			if(mode == DrawingMode.Scene) {
				Gizmos.color = Color.cyan;
				Gizmos.DrawLine(parent.position, bone.position);
			}
			if(mode == DrawingMode.Game) {
				Drawing.DrawLine(Drawing.WorldToScreen(parent.position), Drawing.WorldToScreen(bone.position), Color.cyan, 2f, true);
			}
		}
		for(int i=0; i<bone.childCount; i++) {
			if(isJoint) {
				DrawCharacterSkeleton(mode, bone.GetChild(i), bone);
			} else {
				DrawCharacterSkeleton(mode, bone.GetChild(i), parent);
			}
		}
	}

	private void DrawCharacterJoints(DrawingMode mode, Transform bone) {
		bool isJoint = System.Array.Find(Character.Joints, x => x.Transform == bone) != null;
		if(isJoint) {
			if(mode == DrawingMode.Scene) {
				Gizmos.color = Color.black;
				Gizmos.DrawSphere(bone.position, 0.01f);
			}
			if(mode == DrawingMode.Game) {
				Drawing.DrawCircle(Drawing.WorldToScreen(bone.position), 15, Color.red, 1f, true, 5);
				//Gizmos.DrawSphere(bone.position, 0.01f);
			}
		}
		for(int i=0; i<bone.childCount; i++) {
			DrawCharacterJoints(mode, bone.GetChild(i));
		}
	}

	private void DrawTrajectory() {
		if(!Application.isPlaying) {
			return;
		}
		
		//Projections
		for(int i=0; i<Trajectory.Points.Length; i++) {
			Vector3 right = Trajectory.Points[i].Project(Trajectory.Width/2f);
			Vector3 left = Trajectory.Points[i].Project(-Trajectory.Width/2f);
			//Gizmos.color = Color.white;
			//Gizmos.DrawLine(left, right);
			Gizmos.color = Color.black;
			Gizmos.DrawSphere(right, 0.005f);
			Gizmos.DrawSphere(left, 0.005f);
		}

		//Connections
		Gizmos.color = Color.cyan;
		for(int i=0; i<Trajectory.Points.Length-1; i++) {
			Gizmos.DrawLine(Trajectory.Points[i].GetPosition(), Trajectory.Points[i+1].GetPosition());
		}

		//Directions
		Gizmos.color = Color.red;
		for(int i=0; i<Trajectory.Points.Length; i++) {
			Gizmos.DrawLine(Trajectory.Points[i].GetPosition(), Trajectory.Points[i].GetPosition() + 0.1f * Trajectory.Points[i].GetDirection());
		}

		//Positions
		Gizmos.color = Color.black;
		for(int i=0; i<Trajectory.Points.Length; i++) {
			Gizmos.DrawSphere(Trajectory.Points[i].GetPosition(), 0.005f);
		}

		//Target
		Gizmos.color = Color.red;
		Gizmos.DrawLine(Trajectory.GetCurrent().GetPosition(), Trajectory.GetCurrent().GetPosition() + Trajectory.TargetDirection);
		Gizmos.color = Color.green;
		Gizmos.DrawLine(Trajectory.GetCurrent().GetPosition(), Trajectory.GetCurrent().GetPosition() + Trajectory.TargetVelocity);

		//Root
		Gizmos.color = Color.magenta;
		Gizmos.DrawSphere(Trajectory.GetCurrent().GetPosition(), 0.01f);

		/*
		Gizmos.color = Color.blue;
		for(int i=0; i<Points.Length; i++) {
			Vector3 center = Points[i].Position;
			Vector3 left = Points[i].ProjectLeft(Width/2f);
			Vector3 right = Points[i].ProjectRight(Width/2f);
			Gizmos.DrawLine(center, left);
			Gizmos.DrawLine(center, right);
			Gizmos.DrawSphere(left, 0.01f);
			Gizmos.DrawSphere(right, 0.01f);
		}
		*/
		//Gizmos.color = Color.green;
		//for(int i=0; i<Points.Length; i++) {
		//	Gizmos.DrawLine(Points[i].Position, Points[i].Position + Points[i].Velocity);
		//}
	}

}
