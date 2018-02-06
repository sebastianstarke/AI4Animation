using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DrawingDemo : MonoBehaviour {

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			Draw();
		}
	}

	void OnRenderObject() {
		Draw();
	}

	void Draw() {
		//UnityGL.Start();

		float opacity = 1f;

		//UnityGL.DrawWiredCube(Vector3.one, Quaternion.identity, 0.5f, Color.cyan.Transparent(0.4f), Color.black);
		//UnityGL.DrawArrow(Vector3.one + new Vector3(1f, 0f, 0f), 2f*Vector3.one, 0.75f, 0.025f, 0.1f, Color.black, Color.magenta);

		//UnityGL.DrawWiredSphere(Vector3.one + Vector3.one, 1f, Color.cyan.Transparent(opacity), Color.white);
				
		//UnityGL.DrawWiredSphere(Vector3.one, 0.6f, Color.red.Transparent(opacity), Color.black);
		//UnityGL.DrawWiredCube(Vector3.one, Quaternion.identity, 0.8f, Color.cyan.Transparent(opacity), Color.black);

		//UnityGL.DrawWiredCone(Vector3.one + new Vector3(-1f, 0f, 0f), Quaternion.Euler(Angle, 0f, 0f), 0.5f, 1f, Color.red.Transparent(opacity), Color.black);

		//UnityGL.DrawWiredCircle(new Vector3(3f, 1f, 0f), 0.5f, Color.green.Transparent(0.5f), Color.black);

		//UnityGL.Finish();
		//DrawingLibrary.DrawWiredCircle(Vector3.one, Quaternion.Euler(0f, 45f, 0f), 1f, Color.cyan, Color.black);

		//DrawingLibrary.DrawWiredQuad(Vector3.one, Quaternion.Euler(0f, 45f, 0f), 1f, 2f, Color.cyan.Transparent(0.75f), Color.black);

		//DrawingLibrary.DrawWiredCuboid(Vector3.one, Quaternion.Euler(30f, 45f, 0f), new Vector3(2f, 0.5f, 1f), Color.cyan.Transparent(0.5f), Color.black);

		//DrawingLibrary.DrawWiredSphere(Vector3.one, 1f, Color.cyan.Transparent(0.5f), Color.black);
		//DrawingLibrary.DrawWiredCylinder(Vector3.one, Quaternion.Euler(30f, 45f, 0f), 1f, 2f, Color.cyan.Transparent(0.5f), Color.black);

		DrawingLibrary.DrawWiredCapsule(Vector3.one, Quaternion.Euler(30f, 45f, 0f), 1f, 2f, Color.cyan.Transparent(0.5f), Color.black);

		//DrawingLibrary.DrawArrow(Vector3.one, 2f*Vector3.one, 0.75f, 0.05f, 0.1f, Color.green);

		/*
		UnityGL.Start();
		for(int i=0; i<10000; i++) {
			UnityGL.DrawLine(Vector3.zero, Vector3.one, 0.25f, 0f, Color.blue.Transparent(0.5f));
		}
		UnityGL.Finish();
		*/

		//DrawingLibrary.DrawSphere(Vector3.one + Vector3.one + Vector3.one, 1f, Color.cyan.Transparent(opacity));
		//DrawingLibrary.DrawCube(Vector3.one, Quaternion.identity, 1f, Color.red.Transparent(opacity));
		//DrawingLibrary.DrawCapsule(Vector3.one + Vector3.one, Quaternion.Euler(90f, 0f, 0f), 1f, 2f, Color.green.Transparent(opacity));
		//DrawingLibrary.DrawQuad(new Vector3(0f, 1f, 0f), Quaternion.Euler(45f, 180f, 0f), 2f, 5f, Color.yellow.Transparent(opacity));
	}

}
