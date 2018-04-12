using UnityEngine;

public class HeightMap {

	public Matrix4x4 Pivot = Matrix4x4.identity;
	public Vector3[] Points = new Vector3[0];

	private const float Radius = 0.5f;
	private const int Layer1 = 20;
	private const int Layer2 = 40;
	private const int Layer3 = 60;

	public HeightMap() {
		
	}

	public void Sense(Matrix4x4 pivot, LayerMask mask) {
		Pivot = pivot;
		Points = new Vector3[Layer1+Layer2+Layer3];
		Vector3 position = Pivot.GetPosition();
		Quaternion rotation = Quaternion.AngleAxis(Pivot.GetRotation().eulerAngles.y, Vector3.up);
		for(int i=0; i<Layer1; i++) {
			int sample = i;
			float angle = 2f * Mathf.PI * (float)i / (float)Layer1;
			Points[sample] = 1f/3f*Radius * new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));
			Points[sample] = position + rotation * Points[sample];
			Points[sample] = Utility.ProjectGround(Points[sample], mask);
		}
		for(int i=0; i<Layer2; i++) {
			int sample = Layer1+i;
			float angle = 2f * Mathf.PI * (float)i / (float)Layer2;
			Points[sample] = 2f/3f*Radius * new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));
			Points[sample] = position + rotation * Points[sample];
			Points[sample] = Utility.ProjectGround(Points[sample], mask);
		}
		for(int i=0; i<Layer3; i++) {
			int sample = Layer1+Layer2+i;
			float angle = 2f * Mathf.PI * (float)i / (float)Layer3;
			Points[sample] = 3f/3f*Radius * new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));
			Points[sample] = position + rotation * Points[sample];
			Points[sample] = Utility.ProjectGround(Points[sample], mask);
		}
	}

	public void Draw() {
		UltiDraw.Begin();
		UltiDraw.DrawTranslateGizmo(Pivot.GetPosition(), Pivot.GetRotation(), 0.1f);
		for(int i=0; i<Points.Length; i++) {
			UltiDraw.DrawLine(Pivot.GetPosition(), Points[i], UltiDraw.DarkGreen.Transparent(0.1f));
			UltiDraw.DrawCircle(Points[i], 0.025f, UltiDraw.DarkGrey.Transparent(0.5f));
		}
		UltiDraw.End();
	}

}
