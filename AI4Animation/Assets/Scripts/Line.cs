using UnityEngine;

[RequireComponent(typeof(LineRenderer))]
public class Line : MonoBehaviour {

	private LineRenderer Renderer;

	void Awake() {
		Renderer = GetComponent<LineRenderer>();
		Renderer.positionCount = 2;
	}

	public void Draw(Vector3 A, Vector3 B) {
		Renderer.SetPosition(0, A);
		Renderer.SetPosition(1, B);
	}

	public void SetWidth(float width) {
		GetComponent<LineRenderer>().startWidth = width;
		GetComponent<LineRenderer>().endWidth = width;
	}

	public void SetMaterial(Material material) {
		GetComponent<LineRenderer>().material = Resources.Load("Materials/Line", typeof(Material)) as Material;
	}

}
