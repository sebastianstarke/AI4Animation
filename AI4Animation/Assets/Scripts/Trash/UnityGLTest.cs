using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UnityGLTest : MonoBehaviour {

	public Material Material;

	public Vector3[] Points;

	void OnDrawGizmos() {
		UnityGL.Start();
		//UnityGL.DrawWireCircle(Vector3.one, 0.5f, 0.01f, Color.red);
		//UnityGL.DrawCube(Vector3.one, Quaternion.identity, 2f, Color.red);
		UnityGL.Finish();
	}

}
