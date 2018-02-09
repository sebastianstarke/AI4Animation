using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DrawingDemo : MonoBehaviour {
	
	void OnDrawGizmos() {
		/*
		System.DateTime timestamp1 = Utility.GetTimestamp();
		Drawing.Begin();
		for(int i=0; i<1000; i++) {
			Drawing.DrawSphere(Vector3.one, 1f, Color.cyan);
		}
		Drawing.End();
		Debug.Log("Drawing Library: " + Utility.GetElapsedTime(timestamp1));

		System.DateTime timestamp2 = Utility.GetTimestamp();
		Gizmos.color = Color.cyan;
		for(int i=0; i<1000; i++) {
			Gizmos.DrawSphere(Vector3.one, 1f);
		}
		Debug.Log("Unity Gizmos: " + Utility.GetElapsedTime(timestamp2));
		*/
		
		//System.DateTime timestamp = Utility.GetTimestamp();
		Drawing.Begin();
		//Drawing.DrawWiredEllipse(Vector3.one, 1f, 2f, Color.cyan.Transparent(0.5f), Color.black);
		//Drawing.DrawWiredEllipsoid(Vector3.one, 1f, 2f, Color.cyan.Transparent(0.5f), Color.black);
		Drawing.DrawWiredPyramid(Vector3.one, Quaternion.identity, 1f, 1f, Color.cyan.Transparent(0.5f), Color.black);
		//for(int i=0; i<1000; i++) {
			//Drawing.DrawWiredSphere(Vector3.one, 1f, Color.cyan.Transparent(1f), Color.white);
			//Drawing.DrawGrid(Vector3.zero, Quaternion.identity, 1000f, 1000f, 100, 100, Drawing.LightGrey);
		//}
		Drawing.End();
		//Debug.Log("Elapsed Time: " + Utility.GetElapsedTime(timestamp));

		/*
		Gizmos.color = Color.cyan.Transparent(1f);
		Gizmos.DrawSphere(Vector3.one + new Vector3(2f, 0f, 0f), 0.5f);
		Gizmos.color = Color.black;
		Gizmos.DrawWireSphere(Vector3.one + new Vector3(2f, 0f, 0f), 0.5f);
		*/

		if(!Application.isPlaying) {
			Draw();
		}
	}

	void OnRenderObject() {
		Draw();
	}

	void Draw() {
		if(!Application.isPlaying) {
			return;
		}
		Drawing.Begin();

		Drawing.DrawWiredSphere(
			new Vector3(Mathf.Cos(Time.time), 1f, Mathf.Sin(Time.time)), 
			1f, 
			Drawing.Cyan.Transparent(0.75f), 
			Drawing.Black
		);

		Drawing.DrawWiredCapsule(
			new Vector3(Mathf.Cos(Time.time + Mathf.PI), 1f, Mathf.Sin(Time.time + Mathf.PI)),
			Quaternion.Euler(0f, Time.time, 0f),
			0.5f,
			1f,
			Drawing.Cyan.Transparent(0.75f), 
			Drawing.Black
		);

		Drawing.DrawWiredCylinder(
			new Vector3(Mathf.Cos(Time.time + Mathf.PI/2f), 1f, Mathf.Sin(Time.time + Mathf.PI/2f)),
			Quaternion.Euler(0f, Time.time, 0f),
			0.5f,
			1f,
			Drawing.Cyan.Transparent(0.75f), 
			Drawing.Black
		);

		Drawing.DrawWiredBone(
			new Vector3(Mathf.Cos(Time.time + 3f*Mathf.PI/2f), 0.5f, Mathf.Sin(Time.time + 3f*Mathf.PI/2f)),
			Quaternion.Euler(-90f, Time.time, 0f),
			1f,
			1.25f,
			Drawing.Cyan.Transparent(0.75f), 
			Drawing.Black
		);

		Drawing.DrawWiredCube(
			new Vector3(0f, 1f + 0.5f*Mathf.Sin(Time.time), 0f),
			Quaternion.Euler(10f*Time.time, 20f*Time.time, 30f*Time.time),
			0.5f,
			Drawing.Red.Transparent(0.5f),
			Drawing.Green
		);

		for(int i=0; i<100; i++) {
			Drawing.DrawLine(
				new Vector3(0f, 3f, 2f),
				new Vector3(2f*Random.value - 1f, 3f + (2f*Random.value - 1f), 2f + 2f*Random.value - 1f),
				0.1f*Random.value,
				new Color(Random.value, Random.value, Random.value).Transparent(0.5f)
			);
		}

		Drawing.End();
	}

}
