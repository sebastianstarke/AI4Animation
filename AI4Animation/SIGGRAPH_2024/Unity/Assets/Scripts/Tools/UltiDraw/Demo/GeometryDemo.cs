using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GeometryDemo : MonoBehaviour {
	
	public int Lines = 100;
	[Range(0f, 1f)] public float StartWidth = 1f;
	[Range(0f, 1f)] public float EndWidth = 0f;

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			Draw();
		}
	}

	void OnRenderObject() {
		Draw();
	}

	void Draw() {
		UltiDraw.Begin();

		UltiDraw.SetCurvature(0f);

		UltiDraw.DrawQuad(new Vector3(0f, 1f, 3f), Quaternion.identity, 15f, 15f, UltiDraw.DarkGrey);

		UltiDraw.SetCurvature(0.25f);

		for(int i=0; i<Lines; i++) {
			Vector3 dir = new Vector3(2f*Random.value - 1f, 2f*Random.value - 1f, 2f*Random.value - 1f).normalized;
			UltiDraw.DrawLine(
				new Vector3(0f, 1f, 0f),
				new Vector3(0f, 1f, 0f) + dir,
				0.1f*StartWidth*Random.value,
				0.1f*EndWidth*Random.value,
				new Color(Random.value, Random.value, Random.value).Opacity(1f)
			);
		}

		UltiDraw.End();
	}

}
