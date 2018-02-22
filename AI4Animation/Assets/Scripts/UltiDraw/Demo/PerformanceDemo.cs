using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PerformanceDemo : MonoBehaviour {

	public int Objects = 1000;

	
	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			Draw();
		}
	}

	void OnRenderObject() {
		Draw();
	}

	void Draw() {
		Vector3 positionA = new Vector3(-1f, 1f, 0f);
		Vector3 positionB = new Vector3(1f, 1f, 0f);
		Quaternion rotation = Quaternion.identity;

		System.DateTime timestamp1 = System.DateTime.Now;
		UltiDraw.Begin();
		for(int i=0; i<Objects; i++) {
			UltiDraw.DrawSphere(positionA, rotation, 1f, UltiDraw.Cyan);
		}
		UltiDraw.End();
		Debug.Log("UltiDraw: " + (System.DateTime.Now - timestamp1).Duration().Milliseconds + "ms");

		System.DateTime timestamp2 = System.DateTime.Now;
		for(int i=0; i<Objects; i++) {
			Gizmos.color = UltiDraw.Cyan;
			Gizmos.DrawSphere(positionB, 0.5f);
		}
		Debug.Log("Unity: " + (System.DateTime.Now - timestamp2).Duration().Milliseconds + "ms");
	}

}
