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
		HandleTrajecotry();
	}

	private void RegularUpdate() {
		Character.Move(new Vector2(XAxis, YAxis));
		Character.Turn(Turn);
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

	private void HandleTrajecotry() {
		/* Update Target Direction / Velocity */
		float velocity = 1f*Time.deltaTime;
		Vector3 targetDirection = UnityEngine.Camera.main.transform.forward;
		
		float angle = Utility.GetSignedAngle(Vector3.forward, targetDirection, Vector3.up);
		Debug.Log(angle);
		Quaternion targetRotation = Quaternion.Euler(0f, Utility.GetSignedAngle(Vector3.forward, targetDirection, Vector3.up), 0f);
		Vector3 targetVelocity = velocity * (targetRotation * new Vector3(XAxis, 0f, YAxis));
		Trajectory.TargetVelocity = Utility.Interpolate(Trajectory.TargetVelocity, targetVelocity, 0.5f);

		Character.StrafeTarget = 0.5f;
		Character.StrafeAmount = Utility.Interpolate(Character.StrafeAmount, Character.StrafeTarget, 0.9f);

		targetDirection = Utility.Interpolate(Vector3.Normalize(Trajectory.TargetVelocity), targetDirection, 0.5f);
		Trajectory.TargetDirection = Utility.Interpolate(Trajectory.TargetDirection, targetDirection, Character.StrafeAmount);

		/* Predict Future Trajectory */
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

		/* Positions */
		for(int i=Trajectory.Length/2+1; i<Trajectory.Length; i++) {
			Trajectory.Positions[i] = positionsBlend[i];
		}

		/* Rotations */
		for(int i=0; i<Trajectory.Length; i++) {
			Trajectory.Rotations[i] = Quaternion.Euler(0f, Vector3.Angle(Vector3.forward, Trajectory.Directions[i]), 0f);
		}

		/* Heights */
		for(int i=Trajectory.Length/2; i<Trajectory.Length; i++) {
			Trajectory.Positions[i].y = transform.position.y;
		}
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
		Gizmos.color = Color.black;
		for(int i=0; i<Trajectory.Positions.Length; i++) {
			Gizmos.DrawLine(Trajectory.Positions[i], Trajectory.Positions[i] + 0.25f * Trajectory.Directions[i]);
		}
	}

}
