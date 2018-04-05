using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LogExpTest : MonoBehaviour {

	public Vector3 Angles;

	void Update () {
		Quaternion q = Quaternion.Euler(Angles).GetNormalised();
		Quaternion log = q.GetLog();
		Quaternion exp = log.GetExp();
		Debug.Log("Q: " + q);
		Debug.Log("Log: " + log);
		Debug.Log("Exp: " + exp);
	}

}
