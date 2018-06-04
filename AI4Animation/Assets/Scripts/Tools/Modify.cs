using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Modify : MonoBehaviour {

	public Vector3 Position;
	public Vector3 Rotation;

	private Vector3 LastPosition;
	private Quaternion LastRotation;

	void Awake() {
		LastPosition = transform.localPosition;
		LastRotation = transform.localRotation;
	}

	void LateUpdate() {
		if(LastPosition != transform.localPosition || LastRotation != transform.localRotation) {
			transform.localPosition += Position;
			transform.localRotation *= Quaternion.Euler(Rotation);
		}
		LastPosition = transform.localPosition;
		LastRotation = transform.localRotation;
	}
}
