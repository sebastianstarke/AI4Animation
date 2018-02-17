using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DrawingDemo : MonoBehaviour {
	
	void OnDrawGizmos() {
		/*
		System.DateTime timestamp1 = Utility.GetTimestamp();
		UltiDraw.Begin();
		for(int i=0; i<1000; i++) {
			UltiDraw.DrawSphere(Vector3.one, 1f, Color.cyan);
		}
		UltiDraw.End();
		Debug.Log("Drawing Library: " + Utility.GetElapsedTime(timestamp1));

		System.DateTime timestamp2 = Utility.GetTimestamp();
		Gizmos.color = Color.cyan;
		for(int i=0; i<1000; i++) {
			Gizmos.DrawSphere(Vector3.one, 1f);
		}
		Debug.Log("Unity Gizmos: " + Utility.GetElapsedTime(timestamp2));
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

		UltiDraw.Begin();

		UltiDraw.DrawGrid(Vector3.zero, Quaternion.identity, 100, 100, 1f, 1f, UltiDraw.DarkGreen);

		UltiDraw.DrawWiredSphere(
			new Vector3(Mathf.Cos(Time.time), 1f, Mathf.Sin(Time.time)), 
			1f, 
			UltiDraw.Cyan.Transparent(1f), 
			UltiDraw.Black
		);

		UltiDraw.DrawWiredCapsule(
			new Vector3(Mathf.Cos(Time.time + Mathf.PI), 1f, Mathf.Sin(Time.time + Mathf.PI)),
			Quaternion.Euler(0f, Time.time, 0f),
			0.5f,
			1f,
			UltiDraw.Cyan.Transparent(1f), 
			UltiDraw.Black
		);

		UltiDraw.DrawWiredCylinder(
			new Vector3(Mathf.Cos(Time.time + Mathf.PI/2f), 1f, Mathf.Sin(Time.time + Mathf.PI/2f)),
			Quaternion.Euler(0f, Time.time, 0f),
			0.5f,
			1f,
			UltiDraw.Cyan.Transparent(1f), 
			UltiDraw.Black
		);

		UltiDraw.DrawWiredBone(
			new Vector3(Mathf.Cos(Time.time + 3f*Mathf.PI/2f), 0.5f, Mathf.Sin(Time.time + 3f*Mathf.PI/2f)),
			Quaternion.Euler(-90f, Time.time, 0f),
			1f,
			1.25f,
			UltiDraw.Cyan.Transparent(1f), 
			UltiDraw.Black
		);

		UltiDraw.DrawWiredCube(
			new Vector3(0f, 1f + 0.5f*Mathf.Sin(Time.time), 0f),
			Quaternion.Euler(10f*Time.time, 20f*Time.time, 30f*Time.time),
			0.5f,
			UltiDraw.Red.Transparent(1f),
			UltiDraw.Green
		);

		for(int i=0; i<100; i++) {
			UltiDraw.DrawLine(
				new Vector3(0f, 3f, 2f),
				new Vector3(2f*Random.value - 1f, 3f + (2f*Random.value - 1f), 2f + 2f*Random.value - 1f),
				0.1f*Random.value,
				new Color(Random.value, Random.value, Random.value).Transparent(1f)
			);
		}

		UltiDraw.End();
	}

}
