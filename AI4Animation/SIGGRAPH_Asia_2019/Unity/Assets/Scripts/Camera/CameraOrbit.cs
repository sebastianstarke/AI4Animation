using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraOrbit : MonoBehaviour {

	public Transform Target;
	public float Pivot = 1f;
	public float Height = 1f;
	public float Distance = 1f;
	public float Speed = 10f;

	private float Angle = 0f;

	void Update() {
		transform.position = Target.position + Quaternion.AngleAxis(Angle, Vector3.up) * new Vector3(0f, Height, Distance);
		Vector3 a = transform.position;
		Vector3 b = Target.position;
		a.y = 0f;
		b.y = 0f;
		transform.rotation = Quaternion.LookRotation(b-a, Vector3.up);
		Vector3 targetPosition = Target.position;
		targetPosition.y += Pivot;
		transform.LookAt(targetPosition, Vector3.up);
		Angle += Speed * Time.deltaTime;
	}

}
