using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GizmosGLTest : MonoBehaviour {

	public int Count = 100;

	void OnRenderObject() {
		UnityGL.Start();
		Vector3 position = transform.position;
		for(int i=0; i<Count; i++) {
			UnityGL.DrawLine(position, new Vector3(position.x -1f + 2f*Random.value, position.y - 1f + 2f*Random.value, position.z - 1f + 2f*Random.value), new Color(Random.value, Random.value, Random.value, 1f));
		}
		UnityGL.Finish();
		//for(int i=0; i<Count; i++) {
		//	GizmosGL.DrawLine(position, new Vector3(position.x -1f + 2f*Random.value, position.y - 1f + 2f*Random.value, position.z - 1f + 2f*Random.value), new Color(Random.value, Random.value, Random.value, 1f));
		//}
	}

}
