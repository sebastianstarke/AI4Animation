using UnityEngine;

public class HeightMap {

	public Matrix4x4 Pivot = Matrix4x4.identity;
	public Vector3[] Points = new Vector3[0];

	private float Size = 0.25f;
	private const int Layer1 = 20;
	private const int Layer2 = 40;
	private const int Layer3 = 60;

	public HeightMap() {
		Size = 0.25f;
	}

	public HeightMap(float size) {
		Size = size;
	}

	public float GetSize() {
		return Size;
	}

	public void Sense(Matrix4x4 pivot, LayerMask mask) {
		Pivot = pivot;
		Points = new Vector3[Layer1+Layer2+Layer3];
		Vector3 position = Pivot.GetPosition();
		Quaternion rotation = Quaternion.AngleAxis(Pivot.GetRotation().eulerAngles.y, Vector3.up);
		RaycastHit hit;
		for(int i=0; i<Layer1; i++) {
			int sample = i;
			float angle = 2f * Mathf.PI * (float)i / (float)Layer1;
			Vector3 target = position + rotation * new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));
			target = Vector3.Lerp(target, new Vector3(position.x, target.y-1f/1f, position.z), 1f-Size);
			if(Physics.Raycast(position, target - position, out hit, float.PositiveInfinity, mask)) {
				Points[sample] = hit.point;
			} else {
				Points[sample] = target;
			}
		}
		for(int i=0; i<Layer2; i++) {
			int sample = Layer1+i;
			float angle = 2f * Mathf.PI * (float)i / (float)Layer2;
			Vector3 target = position + rotation * new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));
			target = Vector3.Lerp(target, new Vector3(position.x, target.y-1f/2f, position.z), 1f-Size);
			if(Physics.Raycast(position, target - position, out hit, float.PositiveInfinity, mask)) {
				Points[sample] = hit.point;
			} else {
				Points[sample] = target;
			}
		}
		for(int i=0; i<Layer3; i++) {
			int sample = Layer1+Layer2+i;
			float angle = 2f * Mathf.PI * (float)i / (float)Layer3;
			Vector3 target = position + rotation * new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));
			target = Vector3.Lerp(target, new Vector3(position.x, target.y-1f/3f, position.z), 1f-Size);
			if(Physics.Raycast(position, target - position, out hit, float.PositiveInfinity, mask)) {
				Points[sample] = hit.point;
			} else {
				Points[sample] = target;
			}
		}
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
		for(int i=0; i<Points.Length; i++) {
			UltiDraw.DrawLine(Pivot.GetPosition(), Points[i], UltiDraw.DarkGreen.Transparent(0.125f));
			UltiDraw.DrawCircle(Points[i], 0.025f, UltiDraw.Mustard.Transparent(0.5f));
		}
		UltiDraw.End();
	}

}
