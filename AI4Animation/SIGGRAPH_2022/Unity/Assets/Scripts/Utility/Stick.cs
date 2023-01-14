using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Stick : MonoBehaviour {

	public Transform Target;

	public bool X, Y, Z;
	public bool Roll, Pitch, Yaw;

	void Update () {
		if(Target == null) {
			return;
		}
		Vector3 position = transform.position;
		Vector3 targetPosition = Target.position;
		if(X) {
			position.x = targetPosition.x;
		}
		if(Y) {
			position.y = targetPosition.y;
		}
		if(Z) {
			position.z = targetPosition.z;
		}
		transform.position = position;

		Vector3 rotation = transform.eulerAngles;
		Vector3 targetRotation = Target.eulerAngles;
		if(Roll) {
			rotation.z = targetRotation.z;
		}
		if(Pitch) {
			rotation.x = targetRotation.x;
		}
		if(Yaw) {
			rotation.y = targetRotation.y;
		}
		transform.rotation = Quaternion.Euler(rotation);
	}
}
