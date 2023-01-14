using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LogoDemo : MonoBehaviour {

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
		UltiDraw.SetDepthRendering(false);
		UltiDraw.DrawQuad(new Vector3(0f, 1.5f, 1f), Quaternion.identity, 4.25f, 4.25f, UltiDraw.DarkGrey);
		UltiDraw.DrawWireQuad(new Vector3(0f, 1.5f, 1f), Quaternion.identity, 4.25f, 4.25f, UltiDraw.White);

		UltiDraw.SetCurvature(0.5f);
		for(int i=0; i<5; i++) {
			UltiDraw.DrawWireSphere(new Vector3(0f, 1.5f, 0f), Quaternion.identity, i*0.25f, UltiDraw.White);
		}
		
		UltiDraw.DrawSphere(new Vector3(0f, 1.5f, 0f), Quaternion.identity, 2.625f, UltiDraw.Grey.Opacity(0.5f));

		UltiDraw.DrawSphere(new Vector3(0f, 1.5f, 0f), Quaternion.identity, 1.95f, UltiDraw.Mustard.Opacity(0.5f));

		UltiDraw.DrawSphere(new Vector3(0f, 1.5f, 0f), Quaternion.identity, 1.125f, UltiDraw.Cyan);

		for(int i=0; i<10; i++) {
			UltiDraw.DrawWireCube(new Vector3(0f, 1.5f, 0f), Quaternion.Euler(0f, 0f, i*36f + 45f), 1.5f, UltiDraw.White);
		}

		UltiDraw.End();
	}

}
