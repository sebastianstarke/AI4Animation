using UnityEngine;
using UnityEngine.EventSystems;

public class MouseDrag : MonoBehaviour {

	public static bool Translate = true;
	public static bool Rotate = false;

	public float Sensitivity = 10f;

	private Vector2 LastMousePosition;

	void Awake() {

	}

	void Start() {
		LastMousePosition = GetNormalizedMousePosition();
	}

	void Update() {
		if(Input.GetKeyDown(KeyCode.W)) {
			Translate = true;
			Rotate = false;
		}
		if(Input.GetKeyDown(KeyCode.E)) {
			Translate = false;
			Rotate = true;
		}
		LastMousePosition = GetNormalizedMousePosition();
	}

	void OnMouseDrag() {
		if(EventSystem.current != null) {
			EventSystem.current.SetSelectedGameObject(gameObject);
		}

		if(Translate) {
			float screenDistance = Camera.main.WorldToScreenPoint(gameObject.transform.position).z;
			Vector3 newPosition = Camera.main.ScreenToWorldPoint(new Vector3(Input.mousePosition.x, Input.mousePosition.y, screenDistance));
			transform.position = newPosition;
		}

		if(Rotate) {
			Vector2 deltaMousePosition = GetNormalizedDeltaMousePosition();
			transform.Rotate(Camera.main.transform.right, 1000f*Sensitivity*Time.deltaTime*deltaMousePosition.y, Space.World);
			transform.Rotate(Camera.main.transform.up, -1000f*Sensitivity*Time.deltaTime*deltaMousePosition.x, Space.World);
		}
	}

	private Vector2 GetNormalizedMousePosition() {
		Vector2 ViewPortPosition = Camera.main.ScreenToViewportPoint(Input.mousePosition);
		return new Vector2(ViewPortPosition.x, ViewPortPosition.y);
	}

	private Vector2 GetNormalizedDeltaMousePosition() {
		return GetNormalizedMousePosition() - LastMousePosition;
	}
}
