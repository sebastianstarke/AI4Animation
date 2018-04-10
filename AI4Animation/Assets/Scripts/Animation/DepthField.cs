using UnityEngine;

[System.Serializable]
public class DepthField {

	public bool Inspect = false;

	public DepthField() {
		Inspect = false;
	}

	public void Sense(Matrix4x4 transformation) {

	}

	public void Draw() {
		UltiDraw.Begin();

		UltiDraw.End();
	}

}
