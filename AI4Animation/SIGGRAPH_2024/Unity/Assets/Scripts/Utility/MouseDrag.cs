using UnityEngine;

[RequireComponent(typeof(BoxCollider))]
public class MouseDrag : MonoBehaviour {

	public static GameObject Selected = null;
	public static Collider Collider = null;

	public KeyCode Translate = KeyCode.T;
	public KeyCode Rotate = KeyCode.R;

	public float Sensitivity = 10f;
	public float Smoothing = 0.5f;
	public bool ProjectSurface = false;
	public bool LockX = false;
	public bool LockY = false;
	public bool LockZ = false;

	private Vector2 LastMousePosition;
	private Vector3 Offset;

	void Start() {
		LastMousePosition = Input.mousePosition;
	}

	void Update() {
		if(Selected == gameObject) {
			Move();
		}
		LastMousePosition = Input.mousePosition;
	}

	void OnMouseDown() {
		if(Selected != null) {
			return;
		}
		Collider collider = Utility.GetSelectedCollider();
		if(collider.gameObject == gameObject) {
			Selected = gameObject;
			Collider = collider;
			float screenDistance = Camera.main.WorldToScreenPoint(gameObject.transform.position).z;
			Offset = Camera.main.ScreenToWorldPoint(new Vector3(Input.mousePosition.x, Input.mousePosition.y, screenDistance)) - Selected.transform.position;
		}
	}

	void OnMouseUp() {
		Selected = null;
		Collider = null;
	}

	void OnRenderObject() {
		UltiDraw.Begin();
		if(Selected != null && Collider != null) {
			if(Collider is BoxCollider) {
				BoxCollider c = (BoxCollider)Collider;
				if(Input.GetKey(Translate)) {
					UltiDraw.DrawCuboid(c.bounds.center, c.transform.rotation, Vector3.Scale(c.transform.lossyScale, c.size), UltiDraw.Black.Opacity(0.25f));
					UltiDraw.DrawWireCuboid(c.bounds.center, c.transform.rotation, Vector3.Scale(c.transform.lossyScale, c.size), UltiDraw.Purple);
					UltiDraw.DrawTranslateGizmo(c.bounds.center, c.transform.rotation, c.size.magnitude);
				}
				if(Input.GetKey(Rotate)) {
					UltiDraw.DrawCuboid(c.bounds.center, c.transform.rotation, Vector3.Scale(c.transform.lossyScale, c.size), UltiDraw.Black.Opacity(0.25f));
					UltiDraw.DrawWireCuboid(c.bounds.center, c.transform.rotation, Vector3.Scale(c.transform.lossyScale, c.size), UltiDraw.Purple);
					UltiDraw.DrawRotateGizmo(c.bounds.center, c.transform.rotation, c.size.magnitude);
				}
			}
		} else {
			Collider collider = Utility.GetSelectedCollider();
			if(collider.gameObject == gameObject) {
				if(collider is BoxCollider) {
					BoxCollider c = (BoxCollider)collider;
					UltiDraw.DrawWireCuboid(c.bounds.center, c.transform.rotation, Vector3.Scale(c.transform.lossyScale, c.size), UltiDraw.Black.Opacity(0.5f));
				}
			}
		}
		UltiDraw.End();
	}

	private void Move() {
		if(Input.GetKey(Translate)) {
			float screenDistance = Camera.main.WorldToScreenPoint(gameObject.transform.position).z;
			Vector3 current = transform.position;
			Vector3 target = Camera.main.ScreenToWorldPoint(new Vector3(Input.mousePosition.x, Input.mousePosition.y, screenDistance)) - Offset;
			transform.position = Vector3.Lerp(current, target, 1f-Smoothing);
			if(ProjectSurface) {
				transform.position = Utility.ProjectGround(transform.position, LayerMask.GetMask("Default", "Ground"));
			}
		}
		if(Input.GetKey(Rotate)) {
			Vector2 deltaMousePosition = GetNormalizedMousePosition(Input.mousePosition) - GetNormalizedMousePosition(LastMousePosition);
			Vector3 prev = transform.eulerAngles;
			transform.Rotate(Camera.main.transform.right, Sensitivity/Time.deltaTime*deltaMousePosition.y, Space.World);
			transform.Rotate(Camera.main.transform.up, -Sensitivity/Time.deltaTime*deltaMousePosition.x, Space.World);
			Vector3 next = transform.eulerAngles;
			if(LockX) {
				next.x = prev.x;
			}
			if(LockY) {
				next.y = prev.y;
			}
			if(LockZ) {
				next.z = prev.z;
			}
			transform.rotation = Quaternion.Slerp(Quaternion.Euler(prev), Quaternion.Euler(next), 1f-Smoothing);
		}
	}

	private Vector2 GetNormalizedMousePosition(Vector2 mousePosition) {
		return Camera.main.ScreenToViewportPoint(mousePosition);
	}

}
