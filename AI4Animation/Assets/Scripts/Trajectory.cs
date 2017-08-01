using UnityEngine;

public class Trajectory {

	public int Length = 120;

	public float Width = 25f;

	public Vector3 TargetVelocity;
	public Vector3 TargetPosition;
	public Vector3 TargetDirection;

	public Vector3[] Positions;
	public Vector3[] Directions;
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
		Directions = new Vector3[Length];
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
		for(int i=0; i<Length; i++) {
			Positions[i] = Transform.position;
			Directions[i] = Transform.forward;
			Heights[i] = Transform.position.y;
		}
	}

	public void SetTarget(Vector3 position, Vector3 direction) {
		Positions[Length-1] = position;
		Directions[Length-1] = direction;
	}

}
