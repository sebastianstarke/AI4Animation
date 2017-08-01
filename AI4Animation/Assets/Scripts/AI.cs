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
		//Character.Move(new Vector2(XAxis, YAxis));
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
		float acceleration = 20f;
		float damping = 5f;
		float decay = 2.5f;

		int current = Trajectory.Length/2;
		int last = Trajectory.Length-1;

		Trajectory.TargetDirection = /*transform.rotation **/ new Vector3(XAxis, 0f, YAxis).normalized;
		Trajectory.TargetDirection.y = 0f;
		Trajectory.TargetVelocity = Utility.Interpolate(Trajectory.TargetVelocity, Vector3.zero, damping * Time.deltaTime);
		Trajectory.TargetVelocity = Trajectory.TargetVelocity + acceleration * Time.deltaTime * Trajectory.TargetDirection;
		Trajectory.TargetPosition = Trajectory.TargetPosition + Time.deltaTime * Trajectory.TargetVelocity;

		if(Trajectory.TargetDirection.magnitude == 0f) {
			Trajectory.TargetPosition = Utility.Interpolate(Trajectory.TargetPosition, transform.position, decay * Time.deltaTime);
			Trajectory.TargetDirection = Utility.Interpolate(Trajectory.TargetDirection, transform.forward, decay * Time.deltaTime);
			Trajectory.TargetVelocity = Utility.Interpolate(Trajectory.TargetVelocity, Vector3.zero, decay * Time.deltaTime);
			for(int i=current+1; i<Trajectory.Length; i++) {
				Trajectory.Positions[i] = Utility.Interpolate(Trajectory.Positions[i], transform.position, decay * Time.deltaTime);
				Trajectory.Directions[i] = Utility.Interpolate(Trajectory.Directions[i], transform.forward, decay * Time.deltaTime);
			}
		} else {

		}

		//Update Trajectory
		float rate = 5f * Time.deltaTime;

		Trajectory.Positions[last] = Trajectory.TargetPosition;
		Trajectory.Directions[last] = Trajectory.TargetVelocity;

		for(int i=Trajectory.Length-2; i>=0; i--) {
			float factor = (float)(i+1)/(float)Trajectory.Length;
			factor = 2f * factor - 1f;
			factor = 1f - Mathf.Abs(factor);
			factor = Utility.Normalise(factor, 1f/(float)Trajectory.Length, ((float)Trajectory.Length-1f)/(float)Trajectory.Length, 1f - 60f / Trajectory.Length, 1f);

			Trajectory.Positions[i] = 
				Trajectory.Positions[i] + Utility.Interpolate(
					factor * (Trajectory.Positions[i+1] - Trajectory.Positions[i]), 
					Trajectory.Positions[i+1] - Trajectory.Positions[i],
					rate
				);

			Trajectory.Directions[i] = 
				Trajectory.Directions[i] + Utility.Interpolate(
					factor * (Trajectory.Directions[i+1] - Trajectory.Directions[i]), 
					Trajectory.Directions[i+1] - Trajectory.Directions[i],
					rate
				);
		}

		transform.position = Trajectory.Positions[current];
		transform.rotation = Quaternion.LookRotation(Trajectory.Directions[current], Vector3.up);
	}
	
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

		Gizmos.color = Color.grey;
		Gizmos.DrawSphere(Trajectory.TargetPosition, 0.05f);
		Gizmos.color = Color.red;
		Gizmos.DrawLine(Trajectory.TargetPosition, Trajectory.TargetPosition + Trajectory.TargetDirection);
		Gizmos.color = Color.green;
		Gizmos.DrawLine(Trajectory.TargetPosition, Trajectory.TargetPosition + Trajectory.TargetVelocity);
	}

}
