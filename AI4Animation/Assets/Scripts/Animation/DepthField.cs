using UnityEngine;

[System.Serializable]
public class DepthField {

	public bool Inspect = false;
	public Vector3[] Grid = new Vector3[0];
	public Vector3[] Positions = new Vector3[0];
	public float[] Depths = new float[0];
	Matrix4x4 Pivot = Matrix4x4.identity;

	private static int Resolution = 10;
	private static float Size = 0.5f;
	private static float Distance = 2f;

	public DepthField() {
		Inspect = false;
		Grid = new Vector3[Resolution * Resolution];
		Positions = new Vector3[Resolution * Resolution];
		Depths = new float[Resolution * Resolution];

		float xStart = -Size/2f;
		float yStart = -Size/2f;
		for(int x=0; x<Resolution; x++) {
			for(int y=0; y<Resolution; y++) {
				Grid[GridToArray(x, y)] = new Vector3(xStart + (float)x/(float)(Resolution-1) * Size, yStart + (float)y/(float)(Resolution-1) * Size, Distance);
			}
		}
	}

	public void Sense(Matrix4x4 transformation) {
		Pivot = transformation;
		LayerMask mask = LayerMask.GetMask("Ground");
		RaycastHit hit;
		for(int i=0; i<Grid.Length; i++) {
			Vector3 target = Grid[i].GetRelativePositionFrom(Pivot);
			if(Physics.Raycast(Pivot.GetPosition(), target-Pivot.GetPosition(), out hit, Distance, mask)) {
				Positions[i] = hit.point;
			} else {
				Positions[i] = target;
			}
		}
	}

	private int GridToArray(int x, int y) {
		return x + y*Resolution;
	}

	public void Draw() {
		UltiDraw.Begin();
		for(int i=0; i<Positions.Length; i++) {
			UltiDraw.DrawLine(Pivot.GetPosition(), Positions[i], UltiDraw.DarkGreen.Transparent(0.05f));
			UltiDraw.DrawCircle(Positions[i], 0.025f, UltiDraw.Black.Transparent(0.5f));
		}
		UltiDraw.End();
	}

}
