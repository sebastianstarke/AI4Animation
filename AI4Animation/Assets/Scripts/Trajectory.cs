using UnityEngine;

public class Trajectory {

	public int Length = 120;

	public float Width = 25f;

	public Vector3[] Positions;
	public Vector3[] Directions;
	public Quaternion[] Rotations;
	public float[] Heights;

	public float[] GaitStand;
	public float[] GaitWalk;
	public float[] GaitJog;
	public float[] GaitCrouch;
	public float[] GaitJump;
	public float[] GaitBump;
  
  	public Vector3 TargetDirection;
	public Vector3 TargetVelocity;

	private Transform Transform;

	public Trajectory(Transform t) {
		Transform = t;

		Positions = new Vector3[Length];
		Directions = new Vector3[Length];
		Rotations = new Quaternion[Length];
		Heights = new float[Length];

		GaitStand = new float[Length];
		GaitWalk = new float[Length];
		GaitJog = new float[Length];
		GaitCrouch = new float[Length];
		GaitJump = new float[Length];
		GaitBump = new float[Length];

		TargetDirection = new Vector3(0f,0f,1f);
		TargetVelocity = new Vector3(0f,0f,0f);

		for(int i=0; i<Length; i++) {
			Positions[i] = Transform.position;
			Directions[i] = Transform.forward;
			Rotations[i] = Transform.rotation;
			Heights[i] = Transform.position.y;
		}
	}

}
