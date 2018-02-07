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
		Drawing.Begin();

		//Drawing.DrawWiredCircle(Vector3.one, Quaternion.Euler(0f, 45f, 0f), 1f, Color.cyan, Color.black);

		//Drawing.DrawWiredQuad(Vector3.one, Quaternion.Euler(0f, 45f, 0f), 1f, 2f, Color.cyan.Transparent(0.75f), Color.black);

		//Drawing.DrawWiredCuboid(Vector3.one, Quaternion.Euler(30f, 45f, 0f), new Vector3(2f, 0.5f, 1f), Color.cyan.Transparent(0.5f), Color.black);

		//Drawing.DrawWiredSphere(Vector3.one, 1f, Color.cyan.Transparent(0.5f), Color.black);
		//Drawing.DrawWiredCylinder(Vector3.one, Quaternion.Euler(30f, 45f, 0f), 1f, 2f, Color.cyan.Transparent(0.5f), Color.black);

		//Drawing.DrawWiredCapsule(Vector3.one, Quaternion.Euler(30f, 45f, 0f), 1f, 2f, Color.cyan.Transparent(0.5f), Color.black);

		//Drawing.DrawArrow(Vector3.one, 2f*Vector3.one, 0.75f, 0.05f, 0.1f, Color.green);
		
		//Drawing.DrawWiredBone(Vector3.one, Quaternion.identity, 1f, Color.cyan.Transparent(0.5f), Color.black);

		//Drawing.DrawGUILine(0.5f, 0.5f, 0.75f, 0.75f, 1f, Color.cyan);
		//Drawing.DrawGUICircle(0.5f, 0.5f, 0.1f, Color.cyan);
		//Drawing.DrawGUITriangle(0.25f, 0.25f, 0.5f, 0.25f, 0.375f, 0.5f, Color.red);
		//Drawing.DrawGUIRectangle(0.5f, 0.5f, 0.1f, 0.1f, Color.red);

		//UnityGL.Start();
		//for(int i=0; i<100000; i++) {
		//	UnityGL.DrawLine(Vector3.zero, Vector3.one, Color.blue);
		//}
		//UnityGL.Finish();

		//for(int i=0; i<10000; i++) {
		//	Drawing.DrawLine(Vector3.zero, Vector3.one, Color.blue);
		//}

		//Drawing.DrawLine(Vector3.one, 2f*Vector3.one, Color.cyan);
		//Drawing.DrawLine(Vector3.one, Vector3.zero, Color.red);

		//Drawing.DrawGUIRectangle(new Vector2(0.5f, 0.5f), 0.5f, 0.1f, Color.cyan);

		//Drawing.DrawGUICircle(new Vector2(0.5f, 0.5f), 0.3f, Color.red);
		
		//Drawing.DrawGUILine(new Vector2(0.25f, 0.25f), new Vector2(0.75f, 0.75f), 0.05f, Color.cyan);

		//Drawing.DrawGUIRectangle(new Vector2(0.5f, 0.2f), 0.5f, 0.1f, Color.green);

		Drawing.End();
	}

}
