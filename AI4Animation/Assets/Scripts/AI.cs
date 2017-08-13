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
		/*
		Network = new PFNN(PFNN.MODE.CONSTANT);
		Network.Load();

		Network.Predict(0f);
		for(int i=0; i<Network.Yp.RowCount; i++) {
			Debug.Log(Network.Yp[i, 0]);
		}
		*/
		Character = new Character(transform);
		Trajectory = new Trajectory(transform);
	}

	void Update() {
		PreUpdate();
		RegularUpdate();
		PostUpdate();
		
		Vector3 angles = Quaternion.LookRotation(Trajectory.Velocities[Trajectory.Length/2], Vector3.up).eulerAngles;
		angles.x = 0f;
		transform.rotation = Quaternion.Euler(angles);
	}

	private void PreUpdate() {
		HandleInput();

		//Update Trajectory Targets
		float acceleration = 15f;
		float damping = 5f;
		float decay = 2.5f;

		int current = Trajectory.Length/2;
		int last = Trajectory.Length-1;

		Trajectory.TargetDirection = /*transform.rotation **/ new Vector3(XAxis, 0f, YAxis).normalized;
		Trajectory.TargetDirection.y = 0f;
		Vector3 velocity = Utility.Interpolate(Trajectory.TargetVelocity, Vector3.zero, damping * Time.deltaTime);
		velocity = velocity + acceleration * Time.deltaTime * Trajectory.TargetDirection;
		//Trajectory.TargetVelocity = Utility.Interpolate(Trajectory.TargetVelocity, Vector3.zero, damping * Time.deltaTime);
		//Trajectory.TargetVelocity = Trajectory.TargetVelocity + acceleration * Time.deltaTime * Trajectory.TargetDirection;
		Vector3 target = Trajectory.TargetPosition + Time.deltaTime * Trajectory.TargetVelocity;
		//Trajectory.TargetPosition = Trajectory.TargetPosition + Time.deltaTime * Trajectory.TargetVelocity;
		if(!Physics.CheckSphere(target, 0.1f, LayerMask.GetMask("Obstacles"))) {
			Trajectory.TargetPosition = target;
			Trajectory.TargetVelocity = velocity;
		}

		if(Trajectory.TargetDirection.magnitude == 0f) {
			Trajectory.TargetPosition = Utility.Interpolate(Trajectory.TargetPosition, transform.position, decay * Time.deltaTime);
			Trajectory.TargetVelocity = Utility.Interpolate(Trajectory.TargetVelocity, Vector3.zero, decay * Time.deltaTime);
			for(int i=current+1; i<Trajectory.Length; i++) {
				Trajectory.Positions[i] = Utility.Interpolate(Trajectory.Positions[i], transform.position, decay * Time.deltaTime);
				Trajectory.Velocities[i] = Utility.Interpolate(Trajectory.Velocities[i], Vector3.zero, decay * Time.deltaTime);
			}
		}
		
		//Predict Trajectory
		//float rate = 10f * Time.deltaTime;
		float rate = 0.5f;

		Trajectory.Positions[last] = Trajectory.TargetPosition;
		Trajectory.Velocities[last] = Trajectory.TargetVelocity;

		float pastDamp = 1.5f;
		float futureDamp = 1.5f;
		for(int i=Trajectory.Length-2; i>=0; i--) {
			float factor = (float)(i+1)/(float)Trajectory.Length;
			factor = 2f * factor - 1f;
			factor = 1f - Mathf.Abs(factor);
			factor = Utility.Normalise(factor, 1f/(float)Trajectory.Length, ((float)Trajectory.Length-1f)/(float)Trajectory.Length, 1f - 60f / Trajectory.Length, 1f);

			if(i < current) {
				Trajectory.Positions[i] = 
					Trajectory.Positions[i] + Utility.Interpolate(
						Mathf.Pow(factor, pastDamp) * (Trajectory.Positions[i+1] - Trajectory.Positions[i]), 
						Trajectory.Positions[i+1] - Trajectory.Positions[i],
						rate
					);

				Trajectory.Velocities[i] = 
					Trajectory.Velocities[i] + Utility.Interpolate(
						Mathf.Pow(factor, pastDamp) * (Trajectory.Velocities[i+1] - Trajectory.Velocities[i]), 
						Trajectory.Velocities[i+1] - Trajectory.Velocities[i],
						rate
					);
			} else {
				Trajectory.Positions[i] = 
					Trajectory.Positions[i] + Utility.Interpolate(
						Mathf.Pow(factor, futureDamp) * (Trajectory.Positions[i+1] - Trajectory.Positions[i]), 
						Trajectory.Positions[i+1] - Trajectory.Positions[i],
						rate
					);

				Trajectory.Velocities[i] = 
					Trajectory.Velocities[i] + Utility.Interpolate(
						Mathf.Pow(factor, futureDamp) * (Trajectory.Velocities[i+1] - Trajectory.Velocities[i]), 
						Trajectory.Velocities[i+1] - Trajectory.Velocities[i],
						rate
					);
			}
		}
	}

	private void RegularUpdate() {
		Character.Move(new Vector2(XAxis, YAxis));
		//Character.Turn(Turn);

		//int current = Trajectory.Length/2;
		//int last = Trajectory.Length-1;
		//transform.position = Trajectory.Positions[current];
		//transform.rotation = Quaternion.LookRotation(Trajectory.Directions[current], Vector3.up);
		
		//Network.Predict(0.5f);
	}

	private void PostUpdate() {
		//Adjust Trajectory
		int current = Trajectory.Length/2;
		int last = Trajectory.Length-1;

		Vector3 error = (transform.position - Trajectory.Positions[current]);
		for(int i=0; i<Trajectory.Length; i++) {
			float factor = (float)i / (float)(Trajectory.Length-1);
			Trajectory.Positions[i] += factor * error;
			Trajectory.Velocities[i] = Trajectory.Velocities[i].magnitude * (Trajectory.Velocities[i] + factor * error).normalized;
		}

		for(int i=0; i<Trajectory.Length; i++) {
			Trajectory.Positions[i].y = GetHeight(Trajectory.Positions[i].x, Trajectory.Positions[i].z);
			Vector3 start = Trajectory.Positions[i];
			Vector3 end = Trajectory.Positions[i] + 0.1f * Trajectory.Velocities[i].normalized;
			end.y = (GetHeight(end.x, end.z) - start.y) / 0.1f;
			Trajectory.Velocities[i] = Trajectory.Velocities[i].magnitude * new Vector3(Trajectory.Velocities[i].x, end.y, Trajectory.Velocities[i].z).normalized;
		}

		Trajectory.TargetPosition = Trajectory.Positions[last];

		//Character.Phase = GetPhase();
	}

	private float GetHeight(float x, float y) {
		RaycastHit hit;
		bool intersection = Physics.Raycast(new Vector3(x,-10f,y), Vector3.up, out hit, LayerMask.GetMask("Ground"));
		if(!intersection) {
			intersection = Physics.Raycast(new Vector3(x,10f,y), Vector3.down, out hit, LayerMask.GetMask("Ground"));
		}
		if(intersection) {
			return hit.point.y;
		} else {
			return 0f;
		}
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

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			return;
		}
		Gizmos.color = Color.black;
		for(int i=0; i<Trajectory.Positions.Length-1; i++) {
			Gizmos.DrawLine(Trajectory.Positions[i], Trajectory.Positions[i+1]);
		}
		Gizmos.color = Color.blue;
		for(int i=0; i<Trajectory.Positions.Length; i++) {
			Vector3 ortho = Quaternion.Euler(0f, 90f, 0f) * Trajectory.Velocities[i];
			Vector3 left = Trajectory.Positions[i] - 0.15f * ortho.normalized;
			left.y = GetHeight(left.x, left.z);
			Vector3 right = Trajectory.Positions[i] + 0.15f * ortho.normalized;
			right.y = GetHeight(right.x, right.z);
			Gizmos.DrawLine(Trajectory.Positions[i], left);
			Gizmos.DrawLine(Trajectory.Positions[i], right);
			Gizmos.DrawSphere(left, 0.01f);
			Gizmos.DrawSphere(right, 0.01f);
		}
		Gizmos.color = Color.green;
		for(int i=0; i<Trajectory.Positions.Length; i++) {
			Gizmos.DrawLine(Trajectory.Positions[i], Trajectory.Positions[i] + Trajectory.Velocities[i]);
		}
		Gizmos.color = Color.red;
		for(int i=0; i<Trajectory.Positions.Length; i++) {
			Gizmos.DrawSphere(Trajectory.Positions[i], 0.015f);
		}

		Gizmos.color = Color.cyan;
		Gizmos.DrawSphere(Trajectory.TargetPosition, 0.03f);
		Gizmos.color = Color.red;
		Gizmos.DrawLine(Trajectory.TargetPosition, Trajectory.TargetPosition + Trajectory.TargetDirection);
		Gizmos.color = Color.green;
		Gizmos.DrawLine(Trajectory.TargetPosition, Trajectory.TargetPosition + Trajectory.TargetVelocity);
	}

}
