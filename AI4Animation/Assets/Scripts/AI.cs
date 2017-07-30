using UnityEngine;
using MathNet.Numerics.LinearAlgebra;

public class AI : MonoBehaviour {

	private PFNN Network;
	private Character Character;
	private Trajectory Trajectory;

	public float XAxis = 0f;
	public float YAxis = 0f;
	public float Turn = 0f;

	private const float M_PI = 3.14159265358979323846f;

	void Start() {
		//Network = new PFNN(PFNN.MODE.CONSTANT);
		//Network.Load();
		
		Character = new Character(transform);
		Trajectory = new Trajectory(transform);
	}

	void Update() {
		PreUpdate();
		RegularUpdate();
		PostUpdate();
	}

	private void PreUpdate() {
		HandleInput();
		HandleTrajectory();
	}

	private void RegularUpdate() {
		Character.Move(new Vector2(XAxis, YAxis));
		Character.Turn(Turn);
		//Network.Predict(0.5f);
	}

	private void PostUpdate() {
		//Character.Phase = GetPhase();
	}

	private float GetPhase() {
		float stand_amount = 0f;
		float factor = 0.9f;
		return Mathf.Repeat(Character.Phase + stand_amount*factor + (1f-factor), 2f*M_PI);
	}

	private void HandleInput() {
		XAxis = 0f;
		YAxis = 0f;
		Turn = 0f;
		if(Input.GetKey(KeyCode.W)) {
			YAxis += 1f;
		}
		if(Input.GetKey(KeyCode.S)) {
			YAxis -= 1f;
		}
		if(Input.GetKey(KeyCode.A)) {
			XAxis -= 1f;
		}
		if(Input.GetKey(KeyCode.D)) {
			XAxis += 1f;
		}
		if(Input.GetKey(KeyCode.Q)) {
			Turn -= 1f;
		}
		if(Input.GetKey(KeyCode.E)) {
			Turn += 1f;
		}
	}

	private void HandleTrajectory() {
		int current = Trajectory.Length/2;
		int last = Trajectory.Length-1;
		Trajectory.Positions[current] = transform.position;
		Trajectory.Rotations[current] = transform.rotation;
		Trajectory.Directions[current] = transform.forward;

		//Predict Future Trajectory
		Vector3 targetDirection = UnityEngine.Camera.main.transform.rotation * new Vector3(XAxis, 0f, YAxis);
		Vector3 targetPosition = transform.position + targetDirection;
		Quaternion targetRotation = transform.rotation;
		Trajectory.SetTarget(Utility.Interpolate(Trajectory.Positions[last],targetPosition,0.5f), targetRotation, targetDirection);

		float futureWeight = 0.8f;
		for(int i=current; i<Trajectory.Length-1; i++) {
			int index = i-current-1;
			int points = Trajectory.Length-current-1;
			float factor = (float)(index)/(float)(points);

			Trajectory.Positions[i+1] = 
				Trajectory.Positions[i] + Utility.Interpolate(
					Trajectory.Positions[i+1] - Trajectory.Positions[i],
					factor * (Trajectory.Positions[last] - Trajectory.Positions[i]),
					futureWeight
				);

			/*
			Trajectory.Positions[i] = 
				Trajectory.Positions[i-1] + Utility.Interpolate(
					Trajectory.Positions[i] - Trajectory.Positions[i-1],
					factor * (Trajectory.Positions[last] - Trajectory.Positions[i-1]),
					futureWeight
				);
				*/
				

			/*
			Trajectory.Rotations[i] = 
				Trajectory.Rotations[i] * Utility.Interpolate(
					Trajectory.Rotations[i+1] * Quaternion.Inverse(Trajectory.Rotations[i]), 
					Utility.Interpolate(Quaternion.identity, (Trajectory.Rotations[last] * Quaternion.Inverse(Trajectory.Rotations[i])), factor),
					futureWeight
				);
	
			Trajectory.Directions[i] = 
				Trajectory.Directions[i] + Utility.Interpolate(
					Trajectory.Directions[i+1] - Trajectory.Directions[i], 
					factor * (Trajectory.Directions[last] - Trajectory.Directions[i]), 
					futureWeight
				);
			*/
		}

		//Update Previous Trajectory
		float pastWeight = 0.8f;
		for(int i=current-1; i>=0; i--) {
			int index = i+1;
			int points = current + 1;
			float factor = (float)(index)/(float)(points);

			Trajectory.Positions[i] = 
				Trajectory.Positions[i] + Utility.Interpolate(
					Trajectory.Positions[i+1] - Trajectory.Positions[i], 
					factor * (Trajectory.Positions[current] - Trajectory.Positions[i]), 
					pastWeight
				);

			Trajectory.Rotations[i] = 
				Trajectory.Rotations[i] * Utility.Interpolate(
					Trajectory.Rotations[i+1] * Quaternion.Inverse(Trajectory.Rotations[i]), 
					Utility.Interpolate(Quaternion.identity, (Trajectory.Rotations[current] * Quaternion.Inverse(Trajectory.Rotations[i])), factor),
					pastWeight
				);
	
			Trajectory.Directions[i] = 
				Trajectory.Directions[i] + Utility.Interpolate(
					Trajectory.Directions[i+1] - Trajectory.Directions[i], 
					factor * (Trajectory.Directions[current] - Trajectory.Directions[i]), 
					pastWeight
				);
		}
	}

	/*
	private void HandleTrajectory() {
		float velocity = 1f*Time.deltaTime;
		Vector3 targetDirection = UnityEngine.Camera.main.transform.forward;
		
		Quaternion targetRotation = UnityEngine.Camera.main.transform.rotation;
		Vector3 targetVelocity = velocity * (targetRotation * new Vector3(XAxis, 0f, YAxis));
		Trajectory.TargetVelocity = Utility.Interpolate(Trajectory.TargetVelocity, targetVelocity, 0.5f);

		Character.StrafeTarget = 0.5f;
		Character.StrafeAmount = Utility.Interpolate(Character.StrafeAmount, Character.StrafeTarget, 0.9f);

		targetDirection = Utility.Interpolate(Vector3.Normalize(Trajectory.TargetVelocity), targetDirection, 0.5f);
		Trajectory.TargetDirection = Utility.Interpolate(Trajectory.TargetDirection, targetDirection, Character.StrafeAmount);

		Vector3[] positionsBlend = new Vector3[Trajectory.Length];
		positionsBlend[Trajectory.Length/2] = Trajectory.Positions[Trajectory.Length/2];

		for(int i=Trajectory.Length/2+1; i<Trajectory.Length; i++) {
			float bias_pos = 2f;
			float bias_dir = 4f;
			float scale_pos = (1.0f - Mathf.Pow(1.0f - ((float)(i - Trajectory.Length/2) / (Trajectory.Length/2)), bias_pos));
			float scale_dir = (1.0f - Mathf.Pow(1.0f - ((float)(i - Trajectory.Length/2) / (Trajectory.Length/2)), bias_dir));
			positionsBlend[i] = positionsBlend[i-1] + Utility.Interpolate(Trajectory.Positions[i] - Trajectory.Positions[i-1], Trajectory.TargetVelocity, scale_pos);
			Trajectory.Directions[i] = Utility.Interpolate(Trajectory.Directions[i], Trajectory.TargetDirection, scale_dir);
			Trajectory.Heights[i] = Trajectory.Heights[Trajectory.Length/2];
		}

		for(int i=Trajectory.Length/2+1; i<Trajectory.Length; i++) {
			Trajectory.Positions[i] = positionsBlend[i];
		}

		for(int i=0; i<Trajectory.Length; i++) {
			Trajectory.Rotations[i] = Quaternion.Euler(0f, Vector3.Angle(Vector3.forward, Trajectory.Directions[i]), 0f);
		}

		for(int i=Trajectory.Length/2; i<Trajectory.Length; i++) {
			Trajectory.Positions[i].y = transform.position.y;
		}

		for(int i=Trajectory.Length/2; i>=0; i--) {
			Trajectory.Positions[i]  = Trajectory.Positions[i+1];
			Trajectory.Directions[i] = Trajectory.Directions[i+1];
			Trajectory.Rotations[i] = Trajectory.Rotations[i+1];
			Trajectory.Heights[i] = Trajectory.Heights[i+1];
		}
  	}
	*/

	/*
	private void HandlePostTrajectory() {
		for(int i=0; i<Trajectory.Length/2; i++) {
			Trajectory.Positions[i]  = Trajectory.Positions[i+1];
			Trajectory.Directions[i] = Trajectory.Directions[i+1];
			Trajectory.Rotations[i] = Trajectory.Rotations[i+1];
			Trajectory.Heights[i] = Trajectory.Heights[i+1];
		}

		float gaitStand = 0.5f;
		float stand_amount = Mathf.Pow(1.0f - gaitStand, 0.25f);
		Vector3 trajectoryUpdate = Trajectory.Rotations[Trajectory.Length/2] * new Vector3(Network.Yp[0,0], 0f, Network.Yp[1,0]);
		Trajectory.Positions[Trajectory.Length/2]  = Trajectory.Positions[Trajectory.Length/2] + stand_amount * trajectoryUpdate;
		Trajectory.Directions[Trajectory.Length/2] = 
		Quaternion.Euler(0f, stand_amount * -Network.Yp[2,0], 0f) * Trajectory.Directions[Trajectory.Length/2];
		Trajectory.Rotations[Trajectory.Length/2] = Quaternion.Euler(0f, Utility.GetSignedAngle(Trajectory.Directions[Trajectory.Length/2], Vector3.forward, Vector3.up), 0f);

		for(int i=Trajectory.Length/2+1; i<Trajectory.Length; i++) {
			int w = (Trajectory.Length/2) / 10;
			float m = Mathf.Repeat(((float)i - (Trajectory.Length/2)) / 10f, 1f);
			Trajectory.Positions[i].x = (1-m) * Network.Yp[8+(w*0)+(i/10)-w,0] + m * Network.Yp[8+(w*0)+(i/10)-w+1,0];
			Trajectory.Positions[i].z  = (1-m) * Network.Yp[8+(w*1)+(i/10)-w,0] + m * Network.Yp[8+(w*1)+(i/10)-w+1,0];
			Trajectory.Directions[i].x = (1-m) * Network.Yp[8+(w*2)+(i/10)-w,0] + m * Network.Yp[8+(w*2)+(i/10)-w+1,0];
			Trajectory.Directions[i].z = (1-m) * Network.Yp[8+(w*3)+(i/10)-w,0] + m * Network.Yp[8+(w*3)+(i/10)-w+1,0];
			Trajectory.Positions[i]    = (Trajectory.Rotations[Trajectory.Length/2] * Trajectory.Positions[i]) + Trajectory.Positions[Trajectory.Length/2];
			Trajectory.Directions[i]   = Vector3.Normalize((Trajectory.Rotations[Trajectory.Length/2] * Trajectory.Directions[i]));
			Trajectory.Rotations[i]    = Quaternion.Euler(0f, Utility.GetSignedAngle(Trajectory.Directions[i], Vector3.forward, Vector3.up), 0f);
		}
	}
	*/
	
	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			return;
		}
		Gizmos.color = Color.black;
		for(int i=0; i<Trajectory.Positions.Length-1; i++) {
			Gizmos.DrawLine(Trajectory.Positions[i], Trajectory.Positions[i+1]);
		}
		Gizmos.color = Color.cyan;
		for(int i=0; i<Trajectory.Positions.Length; i++) {
			Gizmos.DrawSphere(Trajectory.Positions[i], 0.025f);
		}
		Gizmos.color = Color.blue;
		for(int i=0; i<Trajectory.Positions.Length; i++) {
			Gizmos.DrawLine(Trajectory.Positions[i], Trajectory.Positions[i] + 0.25f * Trajectory.Directions[i]);
		}
	}

}
