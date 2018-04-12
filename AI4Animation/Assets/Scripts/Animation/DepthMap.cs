using UnityEngine;

public class DepthMap {

	Matrix4x4 Pivot = Matrix4x4.identity;
	public Vector3[] Points = new Vector3[0];

	private const int Resolution = 20;
	private const float Size = 0.5f;
	private const float Distance = 2f;

	public DepthMap() {

	}

	public void Sense(Matrix4x4 pivot, LayerMask mask) {
		Pivot = pivot;
		Points = new Vector3[Resolution*Resolution];
		RaycastHit hit;
		for(int x=0; x<Resolution; x++) {
			for(int y=0; y<Resolution; y++) {
				Vector3 grid = new Vector3(
					-Size/2f + (float)x/(float)(Resolution-1) * Size, 
					-Size/2f + (float)y/(float)(Resolution-1) * Size,
					Distance).GetRelativePositionFrom(Pivot);
				if(Physics.Raycast(Pivot.GetPosition(), grid-Pivot.GetPosition(), out hit, Distance, mask)) {
					Points[GridToArray(x,y)] = hit.point;
				} else {
					Points[GridToArray(x,y)] = grid;
				}
			}
		}
	}

	private int GridToArray(int x, int y) {
		return x + y*Resolution;
	}

	public void Draw() {
		UltiDraw.Begin();
		UltiDraw.DrawTranslateGizmo(Pivot.GetPosition(), Pivot.GetRotation(), 0.1f);
		for(int i=0; i<Points.Length; i++) {
			UltiDraw.DrawLine(Pivot.GetPosition(), Points[i], UltiDraw.DarkGreen.Transparent(0.05f));
			UltiDraw.DrawCircle(Points[i], 0.025f, UltiDraw.DarkGrey.Transparent(0.5f));
		}
		UltiDraw.End();
	}

}
