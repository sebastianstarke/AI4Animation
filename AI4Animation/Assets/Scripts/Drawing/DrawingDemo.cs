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
		DrawingLibrary.Begin();

		//DrawingLibrary.DrawWiredCircle(Vector3.one, Quaternion.Euler(0f, 45f, 0f), 1f, Color.cyan, Color.black);

		//DrawingLibrary.DrawWiredQuad(Vector3.one, Quaternion.Euler(0f, 45f, 0f), 1f, 2f, Color.cyan.Transparent(0.75f), Color.black);

		//DrawingLibrary.DrawWiredCuboid(Vector3.one, Quaternion.Euler(30f, 45f, 0f), new Vector3(2f, 0.5f, 1f), Color.cyan.Transparent(0.5f), Color.black);

		//DrawingLibrary.DrawWiredSphere(Vector3.one, 1f, Color.cyan.Transparent(0.5f), Color.black);
		//DrawingLibrary.DrawWiredCylinder(Vector3.one, Quaternion.Euler(30f, 45f, 0f), 1f, 2f, Color.cyan.Transparent(0.5f), Color.black);

		//DrawingLibrary.DrawWiredCapsule(Vector3.one, Quaternion.Euler(30f, 45f, 0f), 1f, 2f, Color.cyan.Transparent(0.5f), Color.black);

		//DrawingLibrary.DrawArrow(Vector3.one, 2f*Vector3.one, 0.75f, 0.05f, 0.1f, Color.green);
		
		//DrawingLibrary.DrawWiredBone(Vector3.one, Quaternion.identity, 1f, Color.cyan.Transparent(0.5f), Color.black);

		//DrawingLibrary.DrawGUILine(0.5f, 0.5f, 0.75f, 0.75f, 1f, Color.cyan);
		//DrawingLibrary.DrawGUICircle(0.5f, 0.5f, 0.1f, Color.cyan);
		//DrawingLibrary.DrawGUITriangle(0.25f, 0.25f, 0.5f, 0.25f, 0.375f, 0.5f, Color.red);
		//DrawingLibrary.DrawGUIRectangle(0.5f, 0.5f, 0.1f, 0.1f, Color.red);

		//UnityGL.Start();
		//for(int i=0; i<100000; i++) {
		//	UnityGL.DrawLine(Vector3.zero, Vector3.one, Color.blue);
		//}
		//UnityGL.Finish();

		//for(int i=0; i<10000; i++) {
		//	DrawingLibrary.DrawLine(Vector3.zero, Vector3.one, Color.blue);
		//}

		//DrawingLibrary.DrawLine(Vector3.one, 2f*Vector3.one, Color.cyan);
		//DrawingLibrary.DrawLine(Vector3.one, Vector3.zero, Color.red);

		DrawingLibrary.End();
	}

}
