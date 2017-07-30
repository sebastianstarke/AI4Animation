using UnityEngine;

public class Trajectory {

	public int Length = 120;

	public float Width = 25f;

	public Vector3[] Positions;
	public Quaternion[] Rotations;
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
		Rotations = new Quaternion[Length];
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

		for(int i=0; i<Length; i++) {
			Positions[i] = Transform.position;
			Rotations[i] = Transform.rotation;
			Directions[i] = Transform.forward;
			Heights[i] = Transform.position.y;
		}
	}

	public void SetTarget(Vector3 position, Quaternion rotation, Vector3 direction) {
		Positions[Length-1] = position;
		Rotations[Length-1] = rotation;
		Directions[Length-1] = direction;
	}

}
