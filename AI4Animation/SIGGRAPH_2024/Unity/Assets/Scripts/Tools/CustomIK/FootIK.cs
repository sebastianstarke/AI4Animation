using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FootIK : MonoBehaviour {

	public bool AutoUpdate = true;

	public int Iterations = 10;
	public Transform Root;
	public Transform Ankle;
	public Transform Joint;
	public float Radius = 0.1f;
	public Vector3 Offset = Vector3.zero;
	public Vector3 WorldNormal = Vector3.down;
	public Vector3 Normal = Vector3.down;
	
	public LayerMask Ground = 0;

	public bool ImproveContact = false;

	public bool Visualise = false;

	public void ComputeNormal() {
		Normal = WorldNormal.DirectionTo(Ankle.GetGlobalMatrix());
	}

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
		Solve(GetPivotPosition(), GetPivotRotation());
	}

	public void Solve(Vector3 pivotPosition, Quaternion pivotRotation, float damping=0f) {
		float stepHeight = Mathf.Max(0f, pivotPosition.y - Root.position.y);

		if(ImproveContact) {
			float sliding = Mathf.Min(2f - Mathf.Pow(2f, Mathf.Clamp(stepHeight / Radius, 0f, 1f)));
			pivotPosition = Vector3.Lerp(TargetPosition, pivotPosition, 1f - Mathf.Max(damping, sliding));
		} else {
			pivotPosition = Vector3.Lerp(TargetPosition, pivotPosition, 1f - damping);
		}

		Vector3 groundPosition = Utility.ProjectGround(pivotPosition, Ground);
		Vector3 groundNormal = Utility.GetNormal(pivotPosition, Ground);
		Vector3 footNormal = pivotRotation * Normal;

		TargetPosition = groundPosition;
		TargetPosition.y = Mathf.Max(groundPosition.y + stepHeight, pivotPosition.y);
		if(TargetPosition.y <= groundPosition.y) {
			TargetRotation = Quaternion.FromToRotation(footNormal, -groundNormal) * pivotRotation;
		} else {
			float weight = 1f - Mathf.Clamp(Vector3.Distance(TargetPosition, groundPosition) / Radius, 0f, 1f);
			TargetRotation = Quaternion.Slerp(pivotRotation, Quaternion.FromToRotation(footNormal, -groundNormal) * pivotRotation, weight);
		}

		for(int k=0; k<Iterations; k++) {
			for(int i=0; i<Joints.Length; i++) {
				Joints[i].rotation = Quaternion.Slerp(
					Joints[i].rotation,
					Quaternion.FromToRotation(GetPivotPosition() - Joints[i].position, TargetPosition - Joints[i].position) * Joints[i].rotation,
					(float)(i+1)/(float)Joints.Length
				);
			}
			Joint.rotation = TargetRotation;
		}
	}

	public Vector3 GetPivotPosition() {
		return Ankle.position + Ankle.rotation * Offset;
	}
	
	public Quaternion GetPivotRotation() {
		return Joint.rotation;
	}

	void OnRenderObject() {
		
	}

	void OnDrawGizmos() {
		if(Visualise) {
			if(Ankle == null || Joint == null || Normal == Vector3.zero) {
				return;
			}
			if(!Application.isPlaying) {
				ComputeNormal();
			}
			UltiDraw.Begin();
			UltiDraw.DrawSphere(GetPivotPosition(), Quaternion.identity, 0.025f, UltiDraw.Cyan.Opacity(0.5f));
			UltiDraw.DrawArrow(GetPivotPosition(), GetPivotPosition() + 0.25f * (GetPivotRotation() * Normal.normalized), 0.8f, 0.02f, 0.1f, UltiDraw.Cyan.Opacity(0.5f));
			UltiDraw.DrawSphere(GetPivotPosition(), Quaternion.identity, Radius, UltiDraw.Mustard.Opacity(0.5f));
			UltiDraw.DrawSphere(TargetPosition, Quaternion.identity, 0.1f, UltiDraw.Red);
			UltiDraw.End();
		}
	}

}
