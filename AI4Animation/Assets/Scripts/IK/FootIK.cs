using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FootIK : MonoBehaviour {

	public bool AutoUpdate = true;

	public int Iterations = 10;
	public Transform Root;
	public Transform Base;
	public Vector3 Normal = Vector3.down;
	public LayerMask Ground = 0;

	private Vector3 TargetPosition;
	private Quaternion TargetRotation;

	private Transform[] Joints;

	private Vector3 LastPosition;
	private float Threshold = 1f;

	void Start() {
		Initialise();
	}

	void LateUpdate() {
		if(AutoUpdate) {
			Solve();
		}
	}

	public void Initialise() {
		if(Base == null) {
			Debug.Log("No ankle specified.");
		} else {
			Joints = null;
			List<Transform> chain = new List<Transform>();
			Transform joint = Base;
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
			LastPosition = Base.position;
		}
	}

	public void Solve() {
		float velocity = (Base.position - LastPosition).magnitude / Time.deltaTime;
		//Vector3 groundNormal = Utility.GetNormal(Base.position, Ground);
		//Vector3 footNormal = Base.rotation * Normal;
		Vector3 groundPosition = Utility.ProjectGround(Base.position, Ground);
		LastPosition = Base.position;
		//TargetRotation = Quaternion.Slerp(TargetRotation, Quaternion.FromToRotation(footNormal, -groundNormal) * Base.rotation, 0.1f);

		if(Base.position.y > groundPosition.y) {
			float weight = 1f - Mathf.Clamp(velocity / Threshold, 0f, 1f);
			TargetPosition = Utility.Interpolate(Base.position, groundPosition, weight);
			Debug.Log(weight);
		} else {
			TargetPosition = groundPosition;
		}

		for(int k=0; k<Iterations; k++) {

			for(int i=0; i<Joints.Length; i++) {
				//Joints[i].rotation = Quaternion.Slerp(
				//	Joints[i].rotation,
				//	Quaternion.Inverse(Base.rotation) * TargetRotation * Joints[i].rotation,
				//	(float)(i+1)/(float)Joints.Length
				//);
				Joints[i].rotation = Quaternion.Slerp(
					Joints[i].rotation,
					Quaternion.FromToRotation(Base.position - Joints[i].position, TargetPosition - Joints[i].position) * Joints[i].rotation,
					(float)(i+1)/(float)Joints.Length
				);
			}
			//Base.rotation = TargetRotation;

		}
	}

	void OnDrawGizmos() {
		if(Base == null || Normal == Vector3.zero) {
			return;
		}
		Gizmos.color = Color.cyan;
		Gizmos.DrawSphere(Base.position, 0.025f);
		Gizmos.DrawLine(Base.position, Base.position + 0.25f * (Base.rotation * Normal));
	}

}
