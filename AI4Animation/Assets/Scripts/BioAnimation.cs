using UnityEngine;

public class BioAnimation : MonoBehaviour {

	public Controller Controller;
	public Character Character;
	public Trajectory Trajectory;
	public PFNN PFNN;

	//private float extra_strafe_smooth = 0.9f;
	//private float extra_crouched_smooth = 0.9f;

	private float UnitScale = 100f;

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
			Trajectory.GetRoot().Stand = Utility.Interpolate(Trajectory.GetRoot().Stand, standAmount, Trajectory.GaitSmoothing);
			Trajectory.GetRoot().Walk = Utility.Interpolate(Trajectory.GetRoot().Walk, 0f, Trajectory.GaitSmoothing);
			Trajectory.GetRoot().Jog = Utility.Interpolate(Trajectory.GetRoot().Jog, 0f, Trajectory.GaitSmoothing);
			Trajectory.GetRoot().Crouch = Utility.Interpolate(Trajectory.GetRoot().Crouch, Controller.QueryCrouch(), Trajectory.GaitSmoothing);
			Trajectory.GetRoot().Jump = Utility.Interpolate(Trajectory.GetRoot().Jump, 0f, Trajectory.GaitSmoothing);
			Trajectory.GetRoot().Bump = Utility.Interpolate(Trajectory.GetRoot().Bump, 0f, Trajectory.GaitSmoothing);
		} else {
			float standAmount = 1.0f - Mathf.Clamp(Vector3.Magnitude(Trajectory.TargetVelocity) / 0.1f, 0.0f, 1.0f);
			Trajectory.GetRoot().Stand = Utility.Interpolate(Trajectory.GetRoot().Stand, standAmount, Trajectory.GaitSmoothing);
			Trajectory.GetRoot().Walk = Utility.Interpolate(Trajectory.GetRoot().Walk, 1f-Controller.QueryJog(), Trajectory.GaitSmoothing);
			Trajectory.GetRoot().Jog = Utility.Interpolate(Trajectory.GetRoot().Jog, Controller.QueryJog(), Trajectory.GaitSmoothing);
			Trajectory.GetRoot().Crouch = Utility.Interpolate(Trajectory.GetRoot().Crouch, Controller.QueryCrouch(), Trajectory.GaitSmoothing);
			Trajectory.GetRoot().Jump = Utility.Interpolate(Trajectory.GetRoot().Jump, 0f, Trajectory.GaitSmoothing);
			Trajectory.GetRoot().Bump = Utility.Interpolate(Trajectory.GetRoot().Bump, 0f, Trajectory.GaitSmoothing);
		}
		//TODO: Update gait for jog, crouch, ...

		//Blend Trajectory Offset
		Vector3 positionOffset = transform.position - Trajectory.GetRoot().GetPosition();
		Quaternion rotationOffset = Quaternion.Inverse(Trajectory.GetRoot().GetRotation()) * transform.rotation;
		Trajectory.GetRoot().SetPosition(Trajectory.GetRoot().GetPosition() + positionOffset, false);
		Trajectory.GetRoot().SetDirection(rotationOffset * Trajectory.GetRoot().GetDirection());
		//for(int i=Trajectory.GetRootPointIndex(); i<Trajectory.GetPointCount(); i++) {
			//float factor = 1f - (i - Trajectory.GetRootPointIndex())/(Trajectory.GetRootPointIndex() - 1f);
			//Trajectory.Points[i].SetPosition(Trajectory.Points[i].GetPosition() + factor*positionOffset, false);
		//}

		//Predict Future Trajectory
		Vector3[] trajectory_positions_blend = new Vector3[Trajectory.GetPointCount()];
		trajectory_positions_blend[Trajectory.GetRootPointIndex()] = Trajectory.GetRoot().GetPosition();

		for(int i=Trajectory.GetRootPointIndex()+1; i<Trajectory.GetPointCount(); i++) {
			float bias_pos = 0.75f;
			float bias_dir = 1.25f;
			float scale_pos = (1.0f - Mathf.Pow(1.0f - ((float)(i - Trajectory.GetRootPointIndex()) / (Trajectory.GetRootPointIndex())), bias_pos));
			float scale_dir = (1.0f - Mathf.Pow(1.0f - ((float)(i - Trajectory.GetRootPointIndex()) / (Trajectory.GetRootPointIndex())), bias_dir));
			float vel_boost = 1f;

			float rescale = 1f / (Trajectory.GetPointCount() - (Trajectory.GetRootPointIndex() + 1f));
			trajectory_positions_blend[i] = trajectory_positions_blend[i-1] + Vector3.Lerp(
				Trajectory.Points[i].GetPosition() - Trajectory.Points[i-1].GetPosition(), 
				vel_boost * rescale * Trajectory.TargetVelocity,
				scale_pos);
				
			Trajectory.Points[i].SetDirection(Vector3.Lerp(Trajectory.Points[i].GetDirection(), Trajectory.TargetDirection, scale_dir));

			Trajectory.Points[i].Stand = Trajectory.GetRoot().Stand;
			Trajectory.Points[i].Walk = Trajectory.GetRoot().Walk;
			Trajectory.Points[i].Jog = Trajectory.GetRoot().Jog;
			Trajectory.Points[i].Crouch = Trajectory.GetRoot().Crouch;
			Trajectory.Points[i].Jump = Trajectory.GetRoot().Jump;
			Trajectory.Points[i].Bump = Trajectory.GetRoot().Bump;
		}
		
		for(int i=Trajectory.GetRootPointIndex()+1; i<Trajectory.GetPointCount(); i++) {
			Trajectory.Points[i].SetPosition(trajectory_positions_blend[i]);
		}

		CollisionChecks(Trajectory.GetRootPointIndex()+1);

		//Calculate Current and Previous Root
		Transformation prevRoot = new Transformation(Trajectory.GetPrevious().GetPosition(), Trajectory.GetPrevious().GetRotation());
		Transformation currRoot = new Transformation(Trajectory.GetRoot().GetPosition(), Trajectory.GetRoot().GetRotation());

		//Input Trajectory Positions / Directions
		for(int i=0; i<Trajectory.GetSampleCount(); i++) {
			Vector3 pos = Trajectory.Points[Trajectory.Density*i].GetPosition(currRoot);
			Vector3 dir = Trajectory.Points[Trajectory.Density*i].GetDirection(currRoot);
			PFNN.SetInput(Trajectory.GetSampleCount()*0 + i, UnitScale * pos.x);
			PFNN.SetInput(Trajectory.GetSampleCount()*1 + i, UnitScale * pos.z);
			PFNN.SetInput(Trajectory.GetSampleCount()*2 + i, dir.x);
			PFNN.SetInput(Trajectory.GetSampleCount()*3 + i, dir.z);
		}

		//Input Trajectory Gaits
		for (int i=0; i<Trajectory.GetSampleCount(); i++) {
			PFNN.SetInput(Trajectory.GetSampleCount()*4 + i, Trajectory.Points[Trajectory.Density*i].Stand);
			PFNN.SetInput(Trajectory.GetSampleCount()*5 + i, Trajectory.Points[Trajectory.Density*i].Walk);
			PFNN.SetInput(Trajectory.GetSampleCount()*6 + i, Trajectory.Points[Trajectory.Density*i].Jog);
			PFNN.SetInput(Trajectory.GetSampleCount()*7 + i, Trajectory.Points[Trajectory.Density*i].Crouch);
			PFNN.SetInput(Trajectory.GetSampleCount()*8 + i, Trajectory.Points[Trajectory.Density*i].Jump);
			PFNN.SetInput(Trajectory.GetSampleCount()*9 + i, Trajectory.Points[Trajectory.Density*i].Bump);
		}

		//Input Joint Previous Positions / Velocities / Rotations
		for(int i=0; i<Character.Joints.Length; i++) {
			int o = 10*Trajectory.GetSampleCount();
			Vector3 pos = Character.Joints[i].GetPosition(prevRoot);
			Vector3 vel = Character.Joints[i].GetVelocity(prevRoot);
			PFNN.SetInput(o + Character.Joints.Length*3*0 + i*3+0, UnitScale * pos.x);
			PFNN.SetInput(o + Character.Joints.Length*3*0 + i*3+1, UnitScale * pos.y);
			PFNN.SetInput(o + Character.Joints.Length*3*0 + i*3+2, UnitScale * pos.z);
			PFNN.SetInput(o + Character.Joints.Length*3*1 + i*3+0, UnitScale * vel.x);
			PFNN.SetInput(o + Character.Joints.Length*3*1 + i*3+1, UnitScale * vel.y);
			PFNN.SetInput(o + Character.Joints.Length*3*1 + i*3+2, UnitScale * vel.z);
		}

		//Input Trajectory Heights
		for(int i=0; i<Trajectory.GetSampleCount(); i++) {
			int o = 10*Trajectory.GetSampleCount() + Character.Joints.Length*3*2;
			PFNN.SetInput(o + Trajectory.GetSampleCount()*0 + i, UnitScale * (Trajectory.Points[Trajectory.Density*i].Project(Trajectory.Width/2f).y - currRoot.Position.y));
			PFNN.SetInput(o + Trajectory.GetSampleCount()*1 + i, UnitScale * (Trajectory.Points[Trajectory.Density*i].GetHeight() - currRoot.Position.y));
			PFNN.SetInput(o + Trajectory.GetSampleCount()*2 + i, UnitScale * (Trajectory.Points[Trajectory.Density*i].Project(-Trajectory.Width/2f).y - currRoot.Position.y));
		}

		//Predict
		PFNN.Predict(Character.Phase);

		//Read Output
		int opos = 8 + 4*Trajectory.GetRootSampleIndex() + Character.Joints.Length*3*0;
		int ovel = 8 + 4*Trajectory.GetRootSampleIndex() + Character.Joints.Length*3*1;
		for(int i=0; i<Character.Joints.Length; i++) {			
			Vector3 position = new Vector3(PFNN.GetOutput(opos+i*3+0), PFNN.GetOutput(opos+i*3+1), PFNN.GetOutput(opos+i*3+2));
			Vector3 velocity = new Vector3(PFNN.GetOutput(ovel+i*3+0), PFNN.GetOutput(ovel+i*3+1), PFNN.GetOutput(ovel+i*3+2));

			Character.Joints[i].SetVelocity(velocity / UnitScale, currRoot);
			Character.Joints[i].SetPosition(Vector3.Lerp(Character.Joints[i].GetPosition(currRoot) + velocity / UnitScale, position / UnitScale, Character.JointSmoothing), currRoot);
		}

		//Update Past Trajectory
		for(int i=0; i<Trajectory.GetRootPointIndex(); i++) {
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
		//60
		float stand_amount = Mathf.Pow(1.0f-Trajectory.GetRoot().Stand, 0.25f);
		Trajectory.GetRoot().SetPosition(Trajectory.GetRoot().GetPosition() + stand_amount * (Trajectory.GetRoot().GetRotation() * new Vector3(PFNN.GetOutput(0) / UnitScale, 0f, PFNN.GetOutput(1) / UnitScale)));
		Trajectory.GetRoot().SetDirection(Quaternion.AngleAxis(stand_amount * Mathf.Rad2Deg * (-PFNN.GetOutput(2)), Vector3.up) * Trajectory.GetRoot().GetDirection());
		
		for(int i=Trajectory.GetRootPointIndex()+1; i<Trajectory.GetPointCount(); i++) {
			Trajectory.Points[i].SetPosition(Trajectory.Points[i].GetPosition() + stand_amount * (Trajectory.GetRoot().GetRotation() * new Vector3(PFNN.GetOutput(0) / UnitScale, 0f, PFNN.GetOutput(1) / UnitScale)));
		}
		
		//Update Future Trajectory
		Transformation reference = new Transformation(Trajectory.GetRoot().GetPosition(), Trajectory.GetRoot().GetRotation());
		for(int i=Trajectory.GetRootPointIndex()+1; i<Trajectory.GetPointCount(); i++) {
			int w = Trajectory.GetRootSampleIndex();
			float m = Mathf.Repeat(((float)i - (float)Trajectory.GetRootPointIndex()) / (float)Trajectory.Density, 1.0f);
			float posX = (1-m) * PFNN.GetOutput(8+(w*0)+(i/Trajectory.Density)-w) + m * PFNN.GetOutput(8+(w*0)+(i/Trajectory.Density)-w+1);
			float posZ = (1-m) * PFNN.GetOutput(8+(w*1)+(i/Trajectory.Density)-w) + m * PFNN.GetOutput(8+(w*1)+(i/Trajectory.Density)-w+1);
			float dirX = (1-m) * PFNN.GetOutput(8+(w*2)+(i/Trajectory.Density)-w) + m * PFNN.GetOutput(8+(w*2)+(i/Trajectory.Density)-w+1);
			float dirZ = (1-m) * PFNN.GetOutput(8+(w*3)+(i/Trajectory.Density)-w) + m * PFNN.GetOutput(8+(w*3)+(i/Trajectory.Density)-w+1);
			Trajectory.Points[i].SetPosition(
				Utility.Interpolate(
					Trajectory.Points[i].GetPosition(),
					reference.Position + reference.Rotation * new Vector3(posX / UnitScale, 0f, posZ / UnitScale),
					1f - Trajectory.CorrectionSmoothing
					)
				);
			Trajectory.Points[i].SetDirection(
				Utility.Interpolate(
					Trajectory.Points[i].GetDirection(),
					reference.Rotation * new Vector3(dirX, 0f, dirZ).normalized,
					1f - Trajectory.CorrectionSmoothing
					)
				);
		}

		CollisionChecks(Trajectory.GetRootPointIndex());
		
		/* Update Phase */
		Character.Phase = Mathf.Repeat(Character.Phase + (stand_amount * 0.9f + 0.1f) * PFNN.GetOutput(3) * 2f*Mathf.PI, 2f*Mathf.PI);
		
		//Root Position
		transform.position = Trajectory.GetRoot().GetPosition();
		transform.rotation = Trajectory.GetRoot().GetRotation();

		//Update Posture
		Character.ForwardKinematics();

		//PFNN.Finish();
	}

	private void CollisionChecks(int start) {
		for(int i=start; i<Trajectory.GetPointCount(); i++) {
			float safety = 0.5f;
			Vector3 previousPos = Trajectory.Points[i-1].GetPosition();
			Vector3 currentPos = Trajectory.Points[i].GetPosition();
			
			Vector3 testPos = previousPos + safety*(currentPos-previousPos).normalized;
			Vector3 projectedPos = Utility.ProjectCollision(previousPos, testPos, LayerMask.GetMask("Obstacles"));
			if(testPos != projectedPos) {
				Vector3 correctedPos = testPos + safety * (previousPos-testPos).normalized;
				Trajectory.Points[i].SetPosition(correctedPos);
			}
		}
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

	private void DrawCharacterSkeleton(DrawingMode mode, Transform bone) {
		Character.Joint joint = System.Array.Find(Character.Joints, x => x.Transform == bone);
		if(joint != null) {
			if(joint.Parent != null) {
				if(mode == DrawingMode.Scene) {
					Gizmos.color = Color.cyan;
					Gizmos.DrawLine(bone.parent.position, bone.position);
				}
				if(mode == DrawingMode.Game) {
				//	Drawing.DrawLine(Drawing.WorldToScreen(bone.parent.position), Drawing.WorldToScreen(bone.position), Color.cyan, 2f, true);
				}
			}
		}
		for(int i=0; i<bone.childCount; i++) {
			DrawCharacterSkeleton(mode, bone.GetChild(i));
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
				//Drawing.DrawCircle(Drawing.WorldToScreen(bone.position), 15, Color.red, 1f, true, 5);
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
		
		//int step = Trajectory.Density;
		int step = 1;

		//Projections
		for(int i=0; i<Trajectory.GetPointCount(); i+=step) {
			Vector3 right = Trajectory.Points[i].Project(Trajectory.Width/2f);
			Vector3 left = Trajectory.Points[i].Project(-Trajectory.Width/2f);
			//Gizmos.color = Color.white;
			//Gizmos.DrawLine(left, right);
			Gizmos.color = Color.yellow;
			Gizmos.DrawSphere(right, 0.005f);
			Gizmos.DrawSphere(left, 0.005f);
		}

		//Connections
		Gizmos.color = Color.cyan;
		for(int i=0; i<Trajectory.GetPointCount()-1; i+=step) {
			Gizmos.DrawLine(Trajectory.Points[i].GetPosition(), Trajectory.Points[i+1].GetPosition());
		}

		//Directions
		Gizmos.color = new Color(1f, 0.5f, 0f, 1f);
		for(int i=0; i<Trajectory.GetPointCount(); i+=step) {
			Gizmos.DrawLine(Trajectory.Points[i].GetPosition(), Trajectory.Points[i].GetPosition() + 0.25f * Trajectory.Points[i].GetDirection());
		}

		//Positions
		for(int i=0; i<Trajectory.GetPointCount(); i+=step) {
			if(i % Trajectory.Density == 0) {
				Gizmos.color = Color.magenta;
				Gizmos.DrawSphere(Trajectory.Points[i].GetPosition(), 0.015f);
			} else {
				Gizmos.color = Color.black;
				Gizmos.DrawSphere(Trajectory.Points[i].GetPosition(), 0.005f);
			}
		}

		//Target
		Gizmos.color = Color.red;
		Gizmos.DrawLine(Trajectory.GetRoot().GetPosition(), Trajectory.GetRoot().GetPosition() + Trajectory.TargetDirection);
		Gizmos.color = Color.green;
		Gizmos.DrawLine(Trajectory.GetRoot().GetPosition(), Trajectory.GetRoot().GetPosition() + Trajectory.TargetVelocity);
	}

}
