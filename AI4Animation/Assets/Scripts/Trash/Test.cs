using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Test : MonoBehaviour {
	
	public Vector3 Direction;

	private Trajectory Trajectory;

	void Awake() {
		Trajectory = GetComponent<Trajectory>();
	}

	void Update () {

		Direction = Vector3.zero;
		if(Input.GetKey(KeyCode.W)) {
			Direction.z += 1f;
		}
		if(Input.GetKey(KeyCode.S)) {
			Direction.z -= 1f;
		}
		if(Input.GetKey(KeyCode.A)) {
			Direction.x -= 1f;
		}
		if(Input.GetKey(KeyCode.D)) {
			Direction.x += 1f;
		}
		Direction = Direction.normalized;

		Trajectory.Target(Direction);
		Trajectory.Predict();
		float updateX = 0f;
		float updateZ = 0f;
		float angle = 0f;
		int size = 120;
		int capacity = 0;
		for(int i=size/2+1; i<size; i++) {
			capacity += 8;
		}
		float[] future = new float[capacity];
		Trajectory.Correct(updateX, updateZ, angle, future);
	}
}
