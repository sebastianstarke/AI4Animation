using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SerialIK : MonoBehaviour {

	public bool AutoUpdate = true;

	public int Iterations = 10;
	public Transform EndEffector;

	public Transform Target;
	public Vector3 TargetPosition;
	public Quaternion TargetRotation;

	public bool SolvePosition = true;
	public bool SolveRotation = true;

	private Transform[] Joints;

	void Start() {
		Initialise();
	}

	void LateUpdate() {
		if(AutoUpdate) {
			Solve();
		}
	}

	public void Initialise() {
		if(EndEffector == null) {
			Debug.Log("No end effector specified.");
		} else {
			Joints = null;
			List<Transform> chain = new List<Transform>();
			Transform joint = EndEffector;
			while(true) {
				joint = joint.parent;
				if(joint == null) {
					Debug.Log("No valid chain found.");
					return;
				}
				chain.Add(joint);
				if(joint == transform) {
					break;
				}
			}
			chain.Reverse();
			Joints = chain.ToArray();
		}
	}

	public void Solve() {
		if(Target != null) {
			TargetPosition = Target.position;
			TargetRotation = Target.rotation;
		}
		for(int k=0; k<Iterations; k++) {
			for(int i=0; i<Joints.Length; i++) {

				if(SolveRotation) {
					Joints[i].rotation = Quaternion.Slerp(
						Joints[i].rotation,
						Quaternion.Inverse(EndEffector.rotation) * TargetRotation * Joints[i].rotation,
						(float)(i+1)/(float)Joints.Length
					);
				}

				if(SolvePosition) {
					Joints[i].rotation = Quaternion.Slerp(
						Joints[i].rotation,
						Quaternion.FromToRotation(EndEffector.position - Joints[i].position, TargetPosition - Joints[i].position) * Joints[i].rotation,
						(float)(i+1)/(float)Joints.Length
					);
				}

			}

			if(SolveRotation) {
				EndEffector.rotation = TargetRotation;
			}
		}
	}

}
