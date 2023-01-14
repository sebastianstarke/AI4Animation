using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PrimitivesDemo : MonoBehaviour {
	
	public bool DepthRendering = true;

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			Draw();
		}
	}

	void OnRenderObject() {
		Draw();
	}

	void Draw() {
		
		float speed = 100f;
		float spacing = 1.5f;
		float height = 1f;
		int index = 0;

		Color[] colors = UltiDraw.GetRainbowColors(9);

		UltiDraw.Begin();

		UltiDraw.SetDepthRendering(DepthRendering);

		UltiDraw.SetCurvature(0f);
		UltiDraw.DrawQuad(Vector3.zero, Quaternion.Euler(90f, 0f, 0f), 100f, 100f, UltiDraw.DarkGrey);
		UltiDraw.SetCurvature(0.25f);

		UltiDraw.DrawGrid(Vector3.zero, Quaternion.identity, 100, 100, 1f, 1f, UltiDraw.DarkGreen.Opacity(0.5f));

		UltiDraw.DrawWireCube(new Vector3(index*spacing, height, 0f*spacing), Quaternion.Euler(0f, speed*Time.time, 0f), 1f, colors[index]);
		UltiDraw.DrawCube(new Vector3(index*spacing, height, 1f*spacing), Quaternion.Euler(0f, speed*Time.time, 0f), 1f, colors[index]);

		index += 1;

		UltiDraw.DrawWireSphere(new Vector3(index*spacing, height, 0f*spacing), Quaternion.Euler(0f, speed*Time.time, 0f), 1f, colors[index]);
		UltiDraw.DrawSphere(new Vector3(index*spacing, height, 1f*spacing), Quaternion.Euler(0f, speed*Time.time, 0f), 1f, colors[index]);

		index += 1;

		UltiDraw.DrawWireCapsule(new Vector3(index*spacing, height, 0f*spacing), Quaternion.Euler(0f, speed*Time.time, 0f), 0.5f, 1f, colors[index]);
		UltiDraw.DrawCapsule(new Vector3(index*spacing, height, 1f*spacing), Quaternion.Euler(0f, speed*Time.time, 0f), 0.5f, 1f, colors[index]);

		index += 1;

		UltiDraw.DrawWireBone(new Vector3(index*spacing, height-0.5f, 0f*spacing), Quaternion.Euler(-90f, speed*Time.time, 0f), 1f, 1f, colors[index]);
		UltiDraw.DrawBone(new Vector3(index*spacing, height-0.5f, 1f*spacing), Quaternion.Euler(-90f, speed*Time.time, 0f), 1f, 1f, colors[index]);

		index += 1;

		UltiDraw.DrawWireCylinder(new Vector3(index*spacing, height, 0f*spacing), Quaternion.Euler(0f, speed*Time.time, 0f), 0.5f, 1f, colors[index]);
		UltiDraw.DrawCylinder(new Vector3(index*spacing, height, 1f*spacing), Quaternion.Euler(0f, speed*Time.time, 0f), 0.5f, 1f, colors[index]);
		
		index += 1;

		UltiDraw.DrawWirePyramid(new Vector3(index*spacing, height-0.5f, 0f*spacing), Quaternion.Euler(0f, speed*Time.time, 0f), 1f, 1f, colors[index]);
		UltiDraw.DrawPyramid(new Vector3(index*spacing, height-0.5f, 1f*spacing), Quaternion.Euler(0f, speed*Time.time, 0f), 1f, 1f, colors[index]);

		index += 1;

		UltiDraw.DrawWireCone(new Vector3(index*spacing, height-0.5f, 0f*spacing), Quaternion.Euler(0f, speed*Time.time, 0f), 0.75f, 1f, colors[index]);
		UltiDraw.DrawCone(new Vector3(index*spacing, height-0.5f, 1f*spacing), Quaternion.Euler(0f, speed*Time.time, 0f), 0.75f, 1f, colors[index]);

		index += 1;

		UltiDraw.DrawWireCuboid(new Vector3(index*spacing, height, 0f*spacing), Quaternion.Euler(0f, speed*Time.time, 0f), new Vector3(0.5f, 1f, 0.5f), colors[index]);
		UltiDraw.DrawCuboid(new Vector3(index*spacing, height, 1f*spacing), Quaternion.Euler(0f, speed*Time.time, 0f), new Vector3(0.5f, 1f, 0.5f), colors[index]);

		index += 1;

		UltiDraw.DrawWireEllipsoid(new Vector3(index*spacing, height, 0f*spacing), Quaternion.Euler(0f, speed*Time.time, 0f), 0.5f, 1f, colors[index]);
		UltiDraw.DrawEllipsoid(new Vector3(index*spacing, height, 1f*spacing), Quaternion.Euler(0f, speed*Time.time, 0f), 0.5f, 1f, colors[index]);

		index += 1;

		UltiDraw.End();
	}

}
