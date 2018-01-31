using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UnityGLTest : MonoBehaviour {



	void OnDrawGizmos() {
		UnityGL.Start();
		//UnityGL.DrawWiredCube(Vector3.one, Quaternion.identity, 2f, Color.red.Transparent(0.5f), Color.black);
		UnityGL.DrawSphere(Vector3.one, 0.5f, Color.red.Transparent(0.25f));
		UnityGL.DrawWireSphere(Vector3.one, 0.5f, Color.black);
		UnityGL.Finish();
	}

}
