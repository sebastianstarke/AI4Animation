using UnityEngine;

[System.Serializable]
public class HeightField {

	public bool Inspect = false;

	public Vector3[] Circle = new Vector3[0];
	public Vector3[] Positions = new Vector3[0];
	public float[] Heights = new float[0];

	private static float Radius = 0.5f;

	public HeightField() {
		Inspect = false;
		Circle = new Vector3[60];
		Positions = new Vector3[60];
		Heights = new float[60];
		for(int i=0; i<10; i++) {
			float angle = 2f * Mathf.PI * (float)i / 10f;
			Circle[i] = 1f/3f*Radius * new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));
		}
		for(int i=0; i<20; i++) {
			float angle = 2f * Mathf.PI * (float)i / 20f;
			Circle[10+i] = 2f/3f*Radius * new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));
		}
		for(int i=0; i<30; i++) {
			float angle = 2f * Mathf.PI * (float)i / 30f;
			Circle[30+i] = 3f/3f*Radius * new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));
		}
	}

	public void Sense(Matrix4x4 transformation) {
		LayerMask mask = LayerMask.GetMask("Ground");
		for(int i=0; i<60; i++) {
			Positions[i] = transformation.GetPosition() + transformation.GetRotation() * Circle[i];
			Heights[i] = Utility.GetHeight(Positions[i], mask);
		}
	}

	public void Draw() {
		UltiDraw.Begin();
		for(int i=0; i<60; i++) {
			UltiDraw.DrawCircle(new Vector3(Positions[i].x, Heights[i], Positions[i].z), 0.025f, UltiDraw.Brown);
		}
		UltiDraw.End();
	}

}
