using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Stick : MonoBehaviour {

	public Transform Target;

	public bool X, Y, Z;

	void Update () {
		if(Target == null) {
			return;
		}
		Vector3 position = transform.position;
		Vector3 target = Target.position;
		if(X) {
			position.x = target.x;
		}
		if(Y) {
			position.y = target.y;
		}
		if(Z) {
			position.z = target.z;
		}
		transform.position = position;
	}
}
