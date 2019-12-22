using UnityEngine;

[System.Serializable]
public class DepthMap {

	public Matrix4x4 Pivot = Matrix4x4.identity;
	public Vector3[] Points = new Vector3[0];

	private int Resolution;
	private float Size;
	private float Distance;

	public DepthMap(int resolution, float size, float distance) {
		Resolution = resolution;
		Size = size;
		Distance = distance;
	}

	public int GetResolution() {
		return Resolution;
	}

	public float GetSize() {
		return Size;
	}

	public float GetDistance() {
		return Distance;
	}

	public void Sense(Matrix4x4 pivot, LayerMask mask) {
		Pivot = pivot;
		Points = new Vector3[Resolution*Resolution];
		RaycastHit hit;
		for(int x=0; x<Resolution; x++) {
			for(int y=0; y<Resolution; y++) {
				Vector3 direction = new Vector3(
					-Size/2f + (float)x/(float)(Resolution-1) * Size, 
					-Size/2f + (float)y/(float)(Resolution-1) * Size,
					Distance).GetRelativePositionFrom(Pivot) 
					- 
					Pivot.GetPosition();
				direction.Normalize();
				if(Physics.Raycast(Pivot.GetPosition(), direction, out hit, Distance, mask)) {
					Points[GridToArray(x,y)] = hit.point;
				} else {
					Points[GridToArray(x,y)] = Pivot.GetPosition() + Distance*direction;
				}
			}
		}
	}

	public int GridToArray(int x, int y) {
		return x + y*Resolution;
	}

	public float[] GetDistances() {
		float[] distances = new float[Points.Length];
		for(int i=0; i<Points.Length; i++) {
			distances[i] = Vector3.Distance(Pivot.GetPosition(), Points[i]);
		}
		return distances;
	}

	public void Draw() {
		UltiDraw.Begin();
		UltiDraw.DrawTranslateGizmo(Pivot.GetPosition(), Pivot.GetRotation(), 0.1f);
		float size = 0.5f * Size/Resolution;
		Quaternion rotation = Pivot.GetRotation();
		for(int i=0; i<Points.Length; i++) {
			UltiDraw.DrawLine(Pivot.GetPosition(), Points[i], UltiDraw.DarkGreen.Transparent(0.05f));
			UltiDraw.DrawQuad(Points[i], rotation, size, size, UltiDraw.Orange.Transparent(0.5f));
		}
		UltiDraw.End();
	}

}
