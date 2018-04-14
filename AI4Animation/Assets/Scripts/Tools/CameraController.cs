using UnityEngine;
using System.Collections;
using UnityEngine.EventSystems;

[ExecuteInEditMode]
public class CameraController : MonoBehaviour {

	public enum MODE {Follow, LookAt, FreeView}

	public bool ShowGUI = true;

	public MODE Mode = MODE.Follow;
	public Transform Target;
	public Vector3 SelfOffset = Vector3.zero;
	public Vector3 TargetOffset = Vector3.zero;
	[Range(0f, 1f)] public float Damping = 0.975f;
	[Range(-180f, 180f)] public float Yaw = 0f;
	[Range(-45f, 45f)] public float Pitch = 0f;
	[Range(0f, 10f)] public float FOV = 1f;
	public float MinHeight = 0.5f;

	private float Velocity = 5f;
	private float AngularVelocity = 5f;
	private float ZoomVelocity = 10;
	private float Sensitivity = 1f;
	private Vector2 MousePosition;
	private Vector2 LastMousePosition;
	private Vector3 DeltaRotation;
	private Quaternion ZeroRotation;

	private Vector3 TargetPosition;
	private Quaternion TargetRotation;

	void Awake() {
		TargetPosition = transform.position;
		TargetRotation = transform.rotation;
	}

	void Start() {
		SetMode(Mode);
	}

	void Update() {
		if(Target == null) {
			return;
		}

		if(Mode == MODE.Follow) {
			UpdateFollowCamera();
		}
		if(Mode == MODE.LookAt) {
			UpdateLookAtCamera();
		}
		if(Mode == MODE.FreeView) {
			MousePosition = GetNormalizedMousePosition();
			if(EventSystem.current != null) {
				if(!Input.GetKey(KeyCode.Mouse0)) {
					EventSystem.current.SetSelectedGameObject(null);
				}
				if(EventSystem.current.currentSelectedGameObject == null) {
					UpdateFreeCamera();
				}
			} else {
				UpdateFreeCamera();
			}

			LastMousePosition = MousePosition;
		}
	}

	void LateUpdate() {
		//Apply Transformation
		transform.position = Vector3.Lerp(transform.position, TargetPosition, 1f-GetDamping());
		transform.rotation = Quaternion.Lerp(transform.rotation, TargetRotation, 1f-GetDamping());

		//Correct Height
		float height = transform.position.y - Target.position.y;
		if(height < MinHeight) {
			transform.position += new Vector3(0f, MinHeight-height, 0f);
		}
	}
	
	private void UpdateFollowCamera() {
		Vector3 currentPosition = transform.position;
		Quaternion currentRotation = transform.rotation;

		//Determine Target
		Vector3 _selfOffset = FOV * SelfOffset;
		Vector3 _targetOffset = TargetOffset;
		transform.position = Target.position + Target.rotation * _selfOffset;
		transform.RotateAround(Target.position + Target.rotation * _targetOffset, Vector3.up, Yaw);
		transform.RotateAround(Target.position + Target.rotation * _targetOffset, transform.right, Pitch);
		transform.LookAt(Target.position + Target.rotation * _targetOffset);
		//

		TargetPosition = transform.position;
		TargetRotation = transform.rotation;
		transform.position = currentPosition;
		transform.rotation = currentRotation;
	}

	private void UpdateLookAtCamera() {
		Vector3 currentPosition = transform.position;
		Quaternion currentRotation = transform.rotation;

		//Translation
		Vector3 direction = Vector3.zero;
		if(Input.GetKey(KeyCode.LeftArrow)) {
			direction.x -= 1f;
		}
		if(Input.GetKey(KeyCode.RightArrow)) {
			direction.x += 1f;
		}
		if(Input.GetKey(KeyCode.UpArrow)) {
			direction.z += 1f;
		}
		if(Input.GetKey(KeyCode.DownArrow)) {
			direction.z -= 1f;
		}
		transform.position += Velocity*Sensitivity*Time.deltaTime*(transform.rotation*direction);

		//Zoom
		if(Input.mouseScrollDelta.y != 0) {
			transform.position += ZoomVelocity*Sensitivity*Time.deltaTime*Input.mouseScrollDelta.y*transform.forward;
		}

		//Rotation
		transform.LookAt(Target);

		TargetPosition = transform.position;
		TargetRotation = transform.rotation;
		transform.position = currentPosition;
		transform.rotation = currentRotation;
	}

	private void UpdateFreeCamera() {
		Vector3 currentPosition = transform.position;
		Quaternion currentRotation = transform.rotation;

		//Translation
		Vector3 direction = Vector3.zero;
		if(Input.GetKey(KeyCode.LeftArrow)) {
			direction.x -= 1f;
		}
		if(Input.GetKey(KeyCode.RightArrow)) {
			direction.x += 1f;
		}
		if(Input.GetKey(KeyCode.UpArrow)) {
			direction.z += 1f;
		}
		if(Input.GetKey(KeyCode.DownArrow)) {
			direction.z -= 1f;
		}
		transform.position += Velocity*Sensitivity*Time.deltaTime*(transform.rotation*direction);

		//Zoom
		if(Input.mouseScrollDelta.y != 0) {
			transform.position += ZoomVelocity*Sensitivity*Time.deltaTime*Input.mouseScrollDelta.y*transform.forward;
		}

		//Rotation
		MousePosition = GetNormalizedMousePosition();
		if(Input.GetMouseButton(0)) {
			DeltaRotation += 1000f*AngularVelocity*Sensitivity*Time.deltaTime*new Vector3(GetNormalizedDeltaMousePosition().x, GetNormalizedDeltaMousePosition().y, 0f);
			transform.rotation = ZeroRotation * Quaternion.Euler(-DeltaRotation.y, DeltaRotation.x, 0f);
		}

		TargetPosition = transform.position;
		TargetRotation = transform.rotation;
		transform.position = currentPosition;
		transform.rotation = currentRotation;
	}

	public void SetMode(MODE mode) {
		Mode = mode;
		switch(Mode) {
			case MODE.Follow:
			break;

			case MODE.LookAt:
			break;
			
			case MODE.FreeView:
			Vector3 euler = transform.rotation.eulerAngles;
			transform.rotation = Quaternion.Euler(0f, euler.y, 0f);
			ZeroRotation = transform.rotation;
			MousePosition = GetNormalizedMousePosition();
			LastMousePosition = GetNormalizedMousePosition();
			break;
		}
	}

	public void SetTargetPosition(Vector3 position) {
		TargetPosition = position;
	}

	public void SetTargetRotation(Quaternion rotation) {
		TargetRotation = rotation;
	}

	private float GetDamping() {
		return Application.isPlaying ? Damping : 0f;
	}

	private Vector2 GetNormalizedMousePosition() {
		Vector2 ViewPortPosition = Camera.main.ScreenToViewportPoint(Input.mousePosition);
		return new Vector2(ViewPortPosition.x, ViewPortPosition.y);
	}

	private Vector2 GetNormalizedDeltaMousePosition() {
		return MousePosition - LastMousePosition;
	}

	void OnGUI() {
		if(!ShowGUI) {
			return;
		}
		GUI.color = UltiDraw.Mustard;
		GUI.backgroundColor = UltiDraw.Black;
		if(GUI.Button(Utility.GetGUIRect(0.85f, 0.1f, 0.1f, 0.04f), "Follow")) {
			SetMode(MODE.Follow);
		}
		if(GUI.Button(Utility.GetGUIRect(0.85f, 0.15f, 0.1f, 0.04f), "Look At")) {
			SetMode(MODE.LookAt);
		}
		if(GUI.Button(Utility.GetGUIRect(0.85f, 0.2f, 0.1f, 0.04f), "Free View")) {
			SetMode(MODE.FreeView);
		}
		Yaw = GUI.HorizontalSlider(Utility.GetGUIRect(0.85f, 0.25f, 0.1f, 0.02f), Yaw, -180f, 180f);
		GUI.Label(Utility.GetGUIRect(0.96f, 0.25f, 0.04f, 0.02f), "Yaw");
		Pitch = GUI.HorizontalSlider(Utility.GetGUIRect(0.85f, 0.275f, 0.1f, 0.02f), Pitch, -45f, 45f);
		GUI.Label(Utility.GetGUIRect(0.96f, 0.275f, 0.04f, 0.02f), "Pitch");
		FOV = GUI.HorizontalSlider(Utility.GetGUIRect(0.85f, 0.3f, 0.1f, 0.02f), FOV, 0f, 10f);
		GUI.Label(Utility.GetGUIRect(0.96f, 0.3f, 0.04f, 0.02f), "FOV");
		Damping = GUI.HorizontalSlider(Utility.GetGUIRect(0.85f, 0.325f, 0.1f, 0.02f), Damping, 0f, 1f);
		GUI.Label(Utility.GetGUIRect(0.96f, 0.325f, 0.04f, 0.02f), "Damping");
	}

	/*
	public Vector3 MoveTo(Vector3 position, Quaternion rotation, float duration) {
	
	}

	private IEnumerator MoveToCoroutine(Vector3 position, Quaternion rotation, float duration) {
		float StartTime = Time.time;
		float EndTime = StartTime + TransitionTime;
	
		Vector3 startPosition = transform.position;
		Vector3 StartTargetOffset = TargetOffset;

		Vector3 EndSelfOffset = Vector3.zero;
		Vector3 EndTargetOffset = Vector3.zero;

		switch(Mode) {
			case MODE.Follow:
			EndSelfOffset = new Vector3(0f, 1f, -1.5f);
			EndTargetOffset = new Vector3(0f, 0.25f, 1f);
			break;

			case MODE.LookAt:
			break;
			
			case MODE.FreeView:
			Vector3 euler = transform.rotation.eulerAngles;
			transform.rotation = Quaternion.Euler(0f, euler.y, 0f);
			ZeroRotation = transform.rotation;
			MousePosition = GetNormalizedMousePosition();
			LastMousePosition = GetNormalizedMousePosition();
			break;
		}

		while(Time.time < EndTime) {
			float ratio = (Time.time - StartTime) / TransitionTime;
			SelfOffset = Vector3.Lerp(StartSelfOffset, EndSelfOffset, ratio);
			TargetOffset = Vector3.Lerp(StartTargetOffset, EndTargetOffset, ratio);
			yield return 0;
		}

	}
	*/

}