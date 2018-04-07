using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FootIK : MonoBehaviour {

	public bool AutoUpdate = true;

	public int Iterations = 10;
	public Transform Root;
	public Transform Ankle;
	public float Radius = 0.1f;
	public Vector3 Offset = Vector3.zero;
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
		if(Ankle == null) {
			Debug.Log("No ankle specified.");
		} else {
			Joints = null;
			List<Transform> chain = new List<Transform>();
			Transform joint = Ankle;
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
		Vector3 groundPosition = Utility.ProjectGround(GetPivotPosition(), Ground);
		Vector3 groundNormal = Utility.GetNormal(GetPivotPosition(), Ground);
		Vector3 footNormal = GetPivotRotation() * Normal;
		float stepHeight = Mathf.Max(0f, GetPivotPosition().y - Root.position.y);

		TargetPosition = groundPosition;
		TargetPosition.y = Mathf.Max(groundPosition.y + stepHeight, GetPivotPosition().y);
		if(TargetPosition.y <= groundPosition.y) {
			TargetRotation = Quaternion.FromToRotation(footNormal, -groundNormal) * GetPivotRotation();
		} else {
			float weight = 1f - Mathf.Clamp(Vector3.Distance(TargetPosition, groundPosition) / Radius, 0f, 1f);
			TargetRotation = Quaternion.Slerp(GetPivotRotation(), Quaternion.FromToRotation(footNormal, -groundNormal) * GetPivotRotation(), weight);
		}

		for(int k=0; k<Iterations; k++) {
			for(int i=0; i<Joints.Length; i++) {
				Joints[i].rotation = Quaternion.Slerp(
					Joints[i].rotation,
					Quaternion.FromToRotation(GetPivotPosition() - Joints[i].position, TargetPosition - Joints[i].position) * Joints[i].rotation,
					(float)(i+1)/(float)Joints.Length
				);
			}
			Ankle.rotation = TargetRotation;
		}
	}

	private Vector3 GetPivotPosition() {
		return Ankle.position + Ankle.rotation * Offset;
	}
	
	private Quaternion GetPivotRotation() {
		return Ankle.rotation;
	}

	void OnDrawGizmos() {
		if(Ankle == null || Normal == Vector3.zero) {
			return;
		}
		UltiDraw.Begin();
		UltiDraw.DrawSphere(GetPivotPosition(), Quaternion.identity, 0.025f, UltiDraw.Cyan.Transparent(0.5f));
		UltiDraw.DrawArrow(GetPivotPosition(), GetPivotPosition() + 0.25f * (GetPivotRotation() * Normal), 0.75f, 0.025f, 0.1f, UltiDraw.Cyan.Transparent(0.5f));
		UltiDraw.DrawSphere(GetPivotPosition(), Quaternion.identity, Radius, UltiDraw.Red.Transparent(0.5f));
		UltiDraw.End();
	}

}
