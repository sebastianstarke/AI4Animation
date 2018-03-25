using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FootIK : MonoBehaviour {

	public bool AutoUpdate = true;

	public int Iterations = 10;
	public Transform FootBase;
	public Vector3 Normal = Vector3.down;
	public LayerMask Ground = 0;

	private Vector3 TargetPosition;
	private Quaternion TargetRotation;

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
		if(FootBase == null) {
			Debug.Log("No foot base specified.");
		} else {
			Joints = null;
			List<Transform> chain = new List<Transform>();
			Transform joint = FootBase;
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
		TargetPosition = Utility.ProjectGround(FootBase.position, Ground);
		Vector3 groundNormal = Utility.GetNormal(FootBase.position, Ground);
		Vector3 footNormal = FootBase.rotation * Normal;
		TargetRotation = Quaternion.FromToRotation(footNormal, groundNormal) * FootBase.rotation;

		for(int k=0; k<Iterations; k++) {
			for(int i=0; i<Joints.Length; i++) {
				/*
				Joints[i].rotation = Quaternion.Slerp(
					Joints[i].rotation,
					Quaternion.Inverse(FootBase.rotation) * TargetRotation * Joints[i].rotation,
					(float)(i+1)/(float)Joints.Length
				);
				*/
				Joints[i].rotation = Quaternion.Slerp(
					Joints[i].rotation,
					Quaternion.FromToRotation(FootBase.position - Joints[i].position, TargetPosition - Joints[i].position) * Joints[i].rotation,
					(float)(i+1)/(float)Joints.Length
				);
			}
			FootBase.rotation = TargetRotation;
		}
	}

	void OnDrawGizmosSelected() {
		if(FootBase == null || Normal == Vector3.zero) {
			return;
		}
		Gizmos.color = Color.cyan;
		Gizmos.DrawLine(FootBase.position, FootBase.position + 0.25f * (FootBase.rotation * Normal));
	}

}
