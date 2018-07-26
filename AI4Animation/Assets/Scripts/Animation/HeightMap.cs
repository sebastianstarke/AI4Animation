using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HeightMap {

	public Matrix4x4 Pivot = Matrix4x4.identity;
	public Vector3[] Points = new Vector3[0];

	private float Size = 1f;
	private const int Layer1 = 20;
	private const int Layer2 = 40;
	private const int Layer3 = 60;

	public HeightMap() {
		Size = 1f;
		Points = new Vector3[Layer1+Layer2+Layer3];
	}

	public HeightMap(float size) {
		Size = size;
		Points = new Vector3[Layer1+Layer2+Layer3];
	}

	public float GetSize() {
		return Size;
	}

	public void Sense(Matrix4x4 pivot, LayerMask mask) {
		Pivot = pivot;
		Vector3 position = Pivot.GetPosition();
		Quaternion rotation = Quaternion.AngleAxis(Pivot.GetRotation().eulerAngles.y, Vector3.up);
		for(int i=0; i<Layer1; i++) {
			int sample = i;
			float angle = 2f * Mathf.PI * (float)i / (float)Layer1;
			Points[sample] = Utility.ProjectGround(position + 1f/4f * Size * (rotation * new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle))), mask);
		}
		for(int i=0; i<Layer2; i++) {
			int sample = Layer1+i;
			float angle = 2f * Mathf.PI * (float)i / (float)Layer2;
			Points[sample] = Utility.ProjectGround(position + 1f/2f * Size * (rotation * new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle))), mask);
		}
		for(int i=0; i<Layer3; i++) {
			int sample = Layer1+Layer2+i;
			float angle = 2f * Mathf.PI * (float)i / (float)Layer3;
			Points[sample] = Utility.ProjectGround(position + 1f/1f * Size * (rotation * new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle))), mask);
		}
	}

	public float[] GetHeights() {
		float[] heights = new float[Points.Length];
		float reference = Pivot.GetPosition().y;
		for(int i=0; i<heights.Length; i++) {
			heights[i] = Points[i].y - reference;
		}
		return heights;
	}

	public void Draw() {
		UltiDraw.Begin();
		UltiDraw.DrawTranslateGizmo(Pivot.GetPosition(), Pivot.GetRotation(), 0.1f);
		for(int i=0; i<Points.Length; i++) {
			Vector3 bottom = new Vector3(Points[i].x, Pivot.GetPosition().y, Points[i].z);
			Vector3 top = Points[i];
			UltiDraw.DrawLine(bottom, top, UltiDraw.Green.Transparent(0.25f));
			UltiDraw.DrawCircle(bottom, 0.015f, UltiDraw.DarkGrey.Transparent(0.5f));
			UltiDraw.DrawCircle(top, 0.025f, UltiDraw.Mustard.Transparent(0.5f));
		}
		UltiDraw.End();
	}

}
