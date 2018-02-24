using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GizmoDemo : MonoBehaviour {

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

		UltiDraw.DrawTranslateGizmo(new Vector3(-3f, 1f, 0f), Quaternion.Euler(0f, 100f*Time.time, 0f), 1f);
		UltiDraw.DrawRotateGizmo(new Vector3(0f, 1f, 0f), Quaternion.Euler(0f, 100f*Time.time, 0f), 1f);
		UltiDraw.DrawScaleGizmo(new Vector3(3f, 1f, 0f), Quaternion.Euler(0f, 100f*Time.time, 0f), 1f);

		UltiDraw.End();
	}

}
