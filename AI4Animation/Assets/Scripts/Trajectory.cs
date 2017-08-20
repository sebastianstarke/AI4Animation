using UnityEngine;

public class Trajectory {

	public int Length = 120;

	public float Width = 25f;

	public Vector3 TargetPosition;
	public Vector3 TargetVelocity;
	public Vector3 TargetDirection;
	public Quaternion TargetRotation;

	public Vector3[] Positions;
	public Vector3[] Velocities;
	public Vector3[] Directions;
	public Quaternion[] Rotations;
	public float[] Heights;
	
	/*
	public float[] GaitStand;
	public float[] GaitWalk;
	public float[] GaitJog;
	public float[] GaitCrouch;
	public float[] GaitJump;
	public float[] GaitBump;
	*/

	private Transform Transform;

	public Trajectory(Transform t) {
		Transform = t;

		Positions = new Vector3[Length];
		Velocities = new Vector3[Length];
		Directions = new Vector3[Length];
		Rotations = new Quaternion[Length];
		Heights = new float[Length];

		/*
		GaitStand = new float[Length];
		GaitWalk = new float[Length];
		GaitJog = new float[Length];
		GaitCrouch = new float[Length];
		GaitJump = new float[Length];
		GaitBump = new float[Length];
		*/

		TargetPosition = Transform.position;
		TargetVelocity = Vector3.zero;
		TargetDirection = Transform.forward;
		TargetRotation = Quaternion.identity;
		
		for(int i=0; i<Length; i++) {
			Positions[i] = TargetPosition;
			Velocities[i] = TargetVelocity;
			Directions[i] = TargetDirection;
			Rotations[i] = TargetRotation;
			Heights[i] = Transform.position.y;
		}
	}

	public void Predict(Vector3 direction) {
		//Update Trajectory Targets
		float acceleration = 30f;
		float damping = 10f;
		float decay = 2.5f;

		int current = Length/2;
		int last = Length-1;

		TargetDirection = new Vector3(direction.x, 0f, direction.z);
		Vector3 velocity = Utility.Interpolate(TargetVelocity, Vector3.zero, damping * Time.deltaTime) + acceleration * Time.deltaTime * TargetDirection;
		Vector3 target = TargetPosition + Time.deltaTime * TargetVelocity;
		if(!Physics.Raycast(Transform.position, target-Transform.position, (target-Transform.position).magnitude, LayerMask.GetMask("Obstacles"))) {
			TargetPosition = target;
			TargetVelocity = velocity;
		} else {
			TargetVelocity = Vector3.zero;
		}
		
		if(TargetDirection.magnitude == 0f) {
			TargetPosition = Utility.Interpolate(TargetPosition, Transform.position, decay * Time.deltaTime);
			TargetVelocity = Utility.Interpolate(TargetVelocity, Vector3.zero, decay * Time.deltaTime);
			for(int i=current+1; i<Length; i++) {
				Positions[i] = Utility.Interpolate(Positions[i], Transform.position, decay * Time.deltaTime);
				Velocities[i] = Utility.Interpolate(Velocities[i], Vector3.zero, decay * Time.deltaTime);
			}
		}
		
		//Predict Trajectory
		//float rate = 10f * Time.deltaTime;
		float rate = 0.5f;

		Positions[last] = TargetPosition;
		Velocities[last] = TargetVelocity;

		float pastDamp = 1.5f;
		float futureDamp = 1.5f;
		for(int i=Length-2; i>=0; i--) {
			float factor = (float)(i+1)/(float)Length;
			factor = 2f * factor - 1f;
			factor = 1f - Mathf.Abs(factor);
			factor = Utility.Normalise(factor, 1f/(float)Length, ((float)Length-1f)/(float)Length, 1f - 60f / Length, 1f);

			if(i < current) {
				Positions[i] = 
					Positions[i] + Utility.Interpolate(
						Mathf.Pow(factor, pastDamp) * (Positions[i+1] - Positions[i]), 
						Positions[i+1] - Positions[i],
						rate
					);

				Velocities[i] = 
					Velocities[i] + Utility.Interpolate(
						Mathf.Pow(factor, pastDamp) * (Velocities[i+1] - Velocities[i]), 
						Velocities[i+1] - Velocities[i],
						rate
					);
			} else {
				Positions[i] = 
					Positions[i] + Utility.Interpolate(
						Mathf.Pow(factor, futureDamp) * (Positions[i+1] - Positions[i]), 
						Positions[i+1] - Positions[i],
						rate
					);

				Velocities[i] = 
					Velocities[i] + Utility.Interpolate(
						Mathf.Pow(factor, futureDamp) * (Velocities[i+1] - Velocities[i]), 
						Velocities[i+1] - Velocities[i],
						rate
					);
			}
		}
	}

	public void Correct() {
		//Adjust Trajectory
		int current = Length/2;
		int last = Length-1;

		Vector3 error = (Transform.position - Positions[current]);
		for(int i=0; i<Length; i++) {
			float factor = (float)i / (float)(Length-1);
			Positions[i] += factor * error;
			Velocities[i] = Velocities[i].magnitude * (Velocities[i] + factor * error).normalized;
		}

		for(int i=0; i<Length; i++) {
			Positions[i].y = Utility.GetHeight(Positions[i].x, Positions[i].z, LayerMask.GetMask("Ground"));
			Vector3 start = Positions[i];
			Vector3 end = Positions[i] + 0.1f * Velocities[i].normalized;
			end.y = (Utility.GetHeight(end.x, end.z, LayerMask.GetMask("Ground")) - start.y) / 0.1f;
			Velocities[i] = Velocities[i].magnitude * new Vector3(Velocities[i].x, end.y, Velocities[i].z).normalized;
		}

		TargetPosition = Positions[last];

		//Character.Phase = GetPhase();
	}

}
