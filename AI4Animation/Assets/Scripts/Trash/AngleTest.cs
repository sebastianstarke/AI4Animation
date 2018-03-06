using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AngleTest : MonoBehaviour {

	public Vector3 Angles;
	
	void Update() {
		Quaternion rotation = QuaternionEuler(Angles.z, Angles.x, Angles.y);
		transform.rotation = rotation;

		Debug.Log("Z: " + Vector3.SignedAngle(Vector3.right, Vector3.ProjectOnPlane(transform.right, Vector3.forward), Vector3.forward));
		Debug.Log("X: " + Vector3.SignedAngle(Vector3.up, Vector3.ProjectOnPlane(transform.up, Vector3.right), Vector3.right));
		Debug.Log("Y: " + Vector3.SignedAngle(Vector3.forward, Vector3.ProjectOnPlane(transform.forward, Vector3.up), Vector3.up));
	}

	public Quaternion QuaternionEuler(float roll, float pitch, float yaw) {
		roll *= Mathf.Deg2Rad / 2f;
		pitch *= Mathf.Deg2Rad / 2f;
		yaw *= Mathf.Deg2Rad / 2f;

		Vector3 Z = Vector3.forward;
		Vector3 X = Vector3.right;
		Vector3 Y = Vector3.up;

		float sin, cos;

		sin = (float)System.Math.Sin(roll);
		cos = (float)System.Math.Cos(roll);
		Quaternion q1 = new Quaternion(0f, 0f, Z.z * sin, cos);
		sin = (float)System.Math.Sin(pitch);
		cos = (float)System.Math.Cos(pitch);
		Quaternion q2 = new Quaternion(X.x * sin, 0f, 0f, cos);
		sin = (float)System.Math.Sin(yaw);
		cos = (float)System.Math.Cos(yaw);
		Quaternion q3 = new Quaternion(0f, Y.y * sin, 0f, cos);

		return mul(mul(q1, q2), q3);
	}

	public Quaternion mul(Quaternion q1, Quaternion q2) {
		float x =  q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x;
		float y = -q1.x * q2.z + q1.y * q2.w + q1.z * q2.x + q1.w * q2.y;
		float z =  q1.x * q2.y - q1.y * q2.x + q1.z * q2.w + q1.w * q2.z;
		float w = -q1.x * q2.x - q1.y * q2.y - q1.z * q2.z + q1.w * q2.w;
		return new Quaternion(x, y, z, w);
	}

}
