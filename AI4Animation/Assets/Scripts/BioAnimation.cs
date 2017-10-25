using UnityEngine;

public class BioAnimation : MonoBehaviour {

	public Controller Controller;
	public Character Character;
	public Trajectory Trajectory;
	public PFNN PFNN;

	private float UnitScale = 100f;

	private enum DrawingMode {Scene, Game};

	void Reset() {
		Trajectory = new Trajectory(transform);
		Character = new Character(transform);
	}

	void Awake() {
		Trajectory.Initialise();
		PFNN.Initialise();
	}

	void Start() {
		Utility.SetFPS(60);
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
			//Trajectory.GetRoot().Jump = Utility.Interpolate(Trajectory.GetRoot().Jump, Controller.QueryJump(), Trajectory.GaitSmoothing);
			Trajectory.GetRoot().Bump = Utility.Interpolate(Trajectory.GetRoot().Bump, 0f, Trajectory.GaitSmoothing);
		} else {
			float standAmount = 1.0f - Mathf.Clamp(Vector3.Magnitude(Trajectory.TargetVelocity) / 0.1f, 0.0f, 1.0f);
			Trajectory.GetRoot().Stand = Utility.Interpolate(Trajectory.GetRoot().Stand, standAmount, Trajectory.GaitSmoothing);
			Trajectory.GetRoot().Walk = Utility.Interpolate(Trajectory.GetRoot().Walk, 1f-Controller.QueryJog(), Trajectory.GaitSmoothing);
			Trajectory.GetRoot().Jog = Utility.Interpolate(Trajectory.GetRoot().Jog, Controller.QueryJog(), Trajectory.GaitSmoothing);
			Trajectory.GetRoot().Crouch = Utility.Interpolate(Trajectory.GetRoot().Crouch, Controller.QueryCrouch(), Trajectory.GaitSmoothing);
			//Trajectory.GetRoot().Jump = Utility.Interpolate(Trajectory.GetRoot().Jump, Controller.QueryJump(), Trajectory.GaitSmoothing);
			Trajectory.GetRoot().Bump = Utility.Interpolate(Trajectory.GetRoot().Bump, 0f, Trajectory.GaitSmoothing);
		}
		//TODO: Update gait for jog, crouch, ...

		//Blend Trajectory Offset
		/*
		Vector3 positionOffset = transform.position - Trajectory.GetRoot().GetPosition();
		Quaternion rotationOffset = Quaternion.Inverse(Trajectory.GetRoot().GetRotation()) * transform.rotation;
		Trajectory.GetRoot().SetPosition(Trajectory.GetRoot().GetPosition() + positionOffset, false);
		Trajectory.GetRoot().SetDirection(rotationOffset * Trajectory.GetRoot().GetDirection());
		*/
		/*
		for(int i=Trajectory.GetRootPointIndex(); i<Trajectory.GetPointCount(); i++) {
			float factor = 1f - (i - Trajectory.GetRootPointIndex())/(Trajectory.GetRootPointIndex() - 1f);
			Trajectory.Points[i].SetPosition(Trajectory.Points[i].GetPosition() + factor*positionOffset, false);
		}
		*/

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
			//Trajectory.Points[i].Jump = Trajectory.GetRoot().Jump;
			Trajectory.Points[i].Bump = Trajectory.GetRoot().Bump;
		}
		
		for(int i=Trajectory.GetRootPointIndex()+1; i<Trajectory.GetPointCount(); i++) {
			Trajectory.Points[i].SetPosition(trajectory_positions_blend[i], true);
		}

		//Post-Correct Trajectory
		CollisionChecks(Trajectory.GetRootPointIndex()+1);

		//Calculate Root
		Transformation currentRoot = new Transformation(Trajectory.GetRoot().GetPosition(), Trajectory.GetRoot().GetRotation());

		//Input Trajectory Positions / Directions
		for(int i=0; i<Trajectory.GetSampleCount(); i++) {
			Vector3 pos = Trajectory.Points[Trajectory.GetDensity()*i].GetPosition().RelativePositionTo(currentRoot);
			Vector3 dir = Trajectory.Points[Trajectory.GetDensity()*i].GetDirection().RelativeDirectionTo(currentRoot);
			PFNN.SetInput(Trajectory.GetSampleCount()*0 + i, UnitScale * pos.x);
			PFNN.SetInput(Trajectory.GetSampleCount()*1 + i, UnitScale * pos.z);
			PFNN.SetInput(Trajectory.GetSampleCount()*2 + i, dir.x);
			PFNN.SetInput(Trajectory.GetSampleCount()*3 + i, dir.z);
		}

		//Input Trajectory Gaits
		for (int i=0; i<Trajectory.GetSampleCount(); i++) {
			PFNN.SetInput(Trajectory.GetSampleCount()*4 + i, Trajectory.Points[Trajectory.GetDensity()*i].Stand);
			PFNN.SetInput(Trajectory.GetSampleCount()*5 + i, Trajectory.Points[Trajectory.GetDensity()*i].Walk);
			PFNN.SetInput(Trajectory.GetSampleCount()*6 + i, Trajectory.Points[Trajectory.GetDensity()*i].Jog);
			PFNN.SetInput(Trajectory.GetSampleCount()*7 + i, Trajectory.Points[Trajectory.GetDensity()*i].Crouch);
			PFNN.SetInput(Trajectory.GetSampleCount()*8 + i, Trajectory.Points[Trajectory.GetDensity()*i].Jump);
			PFNN.SetInput(Trajectory.GetSampleCount()*9 + i, Trajectory.Points[Trajectory.GetDensity()*i].Bump);
		}

		//Input Joint Previous Positions / Velocities / Rotations
		Transformation previousRoot = new Transformation(Trajectory.GetPrevious().GetPosition(), Trajectory.GetPrevious().GetRotation());
		for(int i=0; i<Character.Joints.Length; i++) {
			int o = 10*Trajectory.GetSampleCount();
			Vector3 pos = Character.Joints[i].GetPosition().RelativePositionTo(previousRoot);
			Vector3 vel = Character.Joints[i].GetVelocity().RelativeDirectionTo(previousRoot);
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
			PFNN.SetInput(o + Trajectory.GetSampleCount()*0 + i, UnitScale * (Trajectory.Points[Trajectory.GetDensity()*i].Project(Trajectory.Width/2f).y - currentRoot.Position.y));
			PFNN.SetInput(o + Trajectory.GetSampleCount()*1 + i, UnitScale * (Trajectory.Points[Trajectory.GetDensity()*i].GetHeight() - currentRoot.Position.y));
			PFNN.SetInput(o + Trajectory.GetSampleCount()*2 + i, UnitScale * (Trajectory.Points[Trajectory.GetDensity()*i].Project(-Trajectory.Width/2f).y - currentRoot.Position.y));
		}

		//Predict
		PFNN.Predict(Character.Phase);

		//Update Past Trajectory
		for(int i=0; i<Trajectory.GetRootPointIndex(); i++) {
			Trajectory.Points[i].SetPosition(Trajectory.Points[i+1].GetPosition(), true);
			Trajectory.Points[i].SetDirection(Trajectory.Points[i+1].GetDirection());
			Trajectory.Points[i].Stand = Trajectory.Points[i+1].Stand;
			Trajectory.Points[i].Walk = Trajectory.Points[i+1].Walk;
			Trajectory.Points[i].Jog = Trajectory.Points[i+1].Jog;
			Trajectory.Points[i].Crouch = Trajectory.Points[i+1].Crouch;
			//Trajectory.Points[i].Jump = Trajectory.Points[i+1].Jump;
			Trajectory.Points[i].Bump = Trajectory.Points[i+1].Bump;
		}

		//Update Current Trajectory
		float stand_amount = Mathf.Pow(1.0f-Trajectory.GetRoot().Stand, 0.25f);
		Trajectory.GetRoot().SetPosition((stand_amount * new Vector3(PFNN.GetOutput(0) / UnitScale, 0f, PFNN.GetOutput(1) / UnitScale)).RelativePositionFrom(currentRoot), true);
		Trajectory.GetRoot().SetDirection(Quaternion.AngleAxis(stand_amount * Mathf.Rad2Deg * (-PFNN.GetOutput(2)), Vector3.up) * Trajectory.GetRoot().GetDirection());
		Transformation newRoot = new Transformation(Trajectory.GetRoot().GetPosition(), Trajectory.GetRoot().GetRotation());

		for(int i=Trajectory.GetRootPointIndex()+1; i<Trajectory.GetPointCount(); i++) {
			Trajectory.Points[i].SetPosition(Trajectory.Points[i].GetPosition() + (stand_amount * new Vector3(PFNN.GetOutput(0) / UnitScale, 0f, PFNN.GetOutput(1) / UnitScale)).RelativeDirectionFrom(newRoot), true);
		}
		
		//Update Future Trajectory
		for(int i=Trajectory.GetRootPointIndex()+1; i<Trajectory.GetPointCount(); i++) {
			int w = Trajectory.GetRootSampleIndex();
			float m = Mathf.Repeat(((float)i - (float)Trajectory.GetRootPointIndex()) / (float)Trajectory.GetDensity(), 1.0f);
			float posX = (1-m) * PFNN.GetOutput(8+(w*0)+(i/Trajectory.GetDensity())-w) + m * PFNN.GetOutput(8+(w*0)+(i/Trajectory.GetDensity())-w+1);
			float posZ = (1-m) * PFNN.GetOutput(8+(w*1)+(i/Trajectory.GetDensity())-w) + m * PFNN.GetOutput(8+(w*1)+(i/Trajectory.GetDensity())-w+1);
			float dirX = (1-m) * PFNN.GetOutput(8+(w*2)+(i/Trajectory.GetDensity())-w) + m * PFNN.GetOutput(8+(w*2)+(i/Trajectory.GetDensity())-w+1);
			float dirZ = (1-m) * PFNN.GetOutput(8+(w*3)+(i/Trajectory.GetDensity())-w) + m * PFNN.GetOutput(8+(w*3)+(i/Trajectory.GetDensity())-w+1);
			Trajectory.Points[i].SetPosition(
				Utility.Interpolate(
					Trajectory.Points[i].GetPosition(),
					new Vector3(posX / UnitScale, 0f, posZ / UnitScale).RelativePositionFrom(newRoot),
					1f - Trajectory.CorrectionSmoothing
					),
					true
				);
			Trajectory.Points[i].SetDirection(
				Utility.Interpolate(
					Trajectory.Points[i].GetDirection(),
					new Vector3(dirX, 0f, dirZ).normalized.RelativeDirectionFrom(newRoot),
					1f - Trajectory.CorrectionSmoothing
					)
				);
		}

		//Post-Correct Trajectory
		CollisionChecks(Trajectory.GetRootPointIndex());
		
		//Update Posture via Forward Kinematics
		transform.OverridePosition(newRoot.Position);
		transform.OverrideRotation(newRoot.Rotation);
		int opos = 8 + 4*Trajectory.GetRootSampleIndex() + Character.Joints.Length*3*0;
		int ovel = 8 + 4*Trajectory.GetRootSampleIndex() + Character.Joints.Length*3*1;
		for(int i=0; i<Character.Joints.Length; i++) {			
			Vector3 position = new Vector3(PFNN.GetOutput(opos+i*3+0), PFNN.GetOutput(opos+i*3+1), PFNN.GetOutput(opos+i*3+2)) / UnitScale;
			Vector3 velocity = new Vector3(PFNN.GetOutput(ovel+i*3+0), PFNN.GetOutput(ovel+i*3+1), PFNN.GetOutput(ovel+i*3+2)) / UnitScale;

			Character.Joints[i].SetPosition(
				Vector3.Lerp(Character.Joints[i].GetPosition().RelativePositionTo(currentRoot) + velocity, position, 0.5f).RelativePositionFrom(currentRoot)
			);
			Character.Joints[i].SetVelocity(
				velocity.RelativeDirectionFrom(currentRoot)
			);
		}
		//Character.ForwardKinematics();

		/* Update Phase */
		Character.Phase = Mathf.Repeat(Character.Phase + (stand_amount * 0.9f + 0.1f) * PFNN.GetOutput(3) * 2f*Mathf.PI, 2f*Mathf.PI);

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
				Trajectory.Points[i].SetPosition(correctedPos, true);
			}
		}
	}

	void OnRenderObject() {
		Trajectory.Draw();
		Character.Draw();
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			OnRenderObject();
		}
	}

}
