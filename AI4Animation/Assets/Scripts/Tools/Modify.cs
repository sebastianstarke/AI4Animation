using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Modify : MonoBehaviour {

	public Vector3 Position;
	public Vector3 Rotation;

	void LateUpdate() {
		transform.localPosition += Position;
		transform.localRotation *= Quaternion.Euler(Rotation);
	}
}
