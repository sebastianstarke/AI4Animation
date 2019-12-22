using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ComparisonDemo : MonoBehaviour {


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

		UltiDraw.DrawQuad(new Vector3(0f, 1f, 3f), Quaternion.identity, 10f, 10f, UltiDraw.DarkGrey);

		UltiDraw.SetCurvature(0.25f);

		UltiDraw.SetFilling(1f);
		UltiDraw.DrawCube(new Vector3(-2f, 1f, 2f), Quaternion.identity, 1f, UltiDraw.Cyan);

		UltiDraw.SetFilling(0f);
		UltiDraw.DrawCube(new Vector3(0f, 1f, 2f), Quaternion.identity, 1f, UltiDraw.Cyan);

		UltiDraw.SetFilling(1f);
		UltiDraw.DrawSphere(new Vector3(-2f, 1f, 0f), Quaternion.identity, 1f, UltiDraw.Cyan);

		UltiDraw.SetFilling(0f);
		UltiDraw.DrawSphere(new Vector3(0f, 1f, 0f), Quaternion.identity, 1f, UltiDraw.Cyan);

		UltiDraw.End();

		Gizmos.color = UltiDraw.Cyan;
		Gizmos.DrawCube(new Vector3(2f, 1f, 2f), Vector3.one);
		Gizmos.DrawSphere(new Vector3(2f, 1f, 0f), 0.5f);
	}

}
