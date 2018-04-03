using UnityEngine;
using System.Collections;
using UnityEngine.EventSystems;

[ExecuteInEditMode]
public class CameraController : MonoBehaviour {

	public enum MODE {SmoothFollow, ConstantView, FreeView}

	public MODE Mode = MODE.SmoothFollow;
	public Transform Target;
	public Vector3 SelfOffset = Vector3.zero;
	public Vector3 TargetOffset = Vector3.zero;
	[Range(0f, 1f)] public float Damping = 0.95f;
	[Range(-180f, 180f)] public float Yaw = 0f;
	[Range(-45f, 45f)] public float Pitch = 0f;
	[Range(0f, 10f)] public float FOV = 1f;
	public float TransitionTime = 1f;
	public float MinHeight = 1f;

	private float Velocity = 5f;
	private float AngularVelocity = 5f;
	private float ZoomVelocity = 10;
	private float Sensitivity = 1f;
	private Vector2 MousePosition;
	private Vector2 LastMousePosition;
	private Vector3 DeltaRotation;
	private Quaternion ZeroRotation;

	void Start() {

	}

	void Update() {
		if(Target == null) {
			return;
		}

		if(Mode == MODE.FreeView) {
			return;
		}

		Vector3 currentPosition = transform.position;
		Quaternion currentRotation = transform.rotation;

		//Determine Target
		Vector3 _selfOffset = FOV * SelfOffset;
		Vector3 _targetOffset = TargetOffset;
		transform.position = Target.position + Target.rotation * _selfOffset;
		transform.RotateAround(Target.position + Target.rotation * _targetOffset, Vector3.up, Yaw);
		transform.RotateAround(Target.position + Target.rotation * _targetOffset, transform.right, Pitch);
		transform.LookAt(Target.position + Target.rotation * _targetOffset);

		//Lerp
		transform.position = Vector3.Lerp(currentPosition, transform.position, 1f-Damping);
		transform.rotation = Quaternion.Lerp(currentRotation, transform.rotation, 1f-Damping);

		//Correct Height
		float height = transform.position.y - Target.position.y;
		if(height < MinHeight) {
			transform.position += new Vector3(0f, MinHeight-height, 0f);
		}
	}

	void LateUpdate() {
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

	void OnGUI() {
		GUI.color = UltiDraw.Mustard;
		GUI.backgroundColor = UltiDraw.Black;
		if(GUI.Button(Utility.GetGUIRect(0.85f, 0.05f, 0.1f, 0.04f), "Smooth Follow")) {
			SetMode(MODE.SmoothFollow);
		}
		if(GUI.Button(Utility.GetGUIRect(0.85f, 0.1f, 0.1f, 0.04f), "Constant View")) {
			SetMode(MODE.ConstantView);
		}
		if(GUI.Button(Utility.GetGUIRect(0.85f, 0.15f, 0.1f, 0.04f), "Free View")) {
			SetMode(MODE.FreeView);
		}
		Yaw = GUI.HorizontalSlider(Utility.GetGUIRect(0.85f, 0.2f, 0.1f, 0.02f), Yaw, -180f, 180f);
		GUI.Label(Utility.GetGUIRect(0.96f, 0.2f, 0.04f, 0.02f), "Yaw");
		Pitch = GUI.HorizontalSlider(Utility.GetGUIRect(0.85f, 0.225f, 0.1f, 0.02f), Pitch, -45f, 45f);
		GUI.Label(Utility.GetGUIRect(0.96f, 0.225f, 0.04f, 0.02f), "Pitch");
		FOV = GUI.HorizontalSlider(Utility.GetGUIRect(0.85f, 0.25f, 0.1f, 0.02f), FOV, 0f, 10f);
		GUI.Label(Utility.GetGUIRect(0.96f, 0.25f, 0.04f, 0.02f), "FOV");
	}

	private void UpdateFreeCamera() {
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
	}


	public void SetMode(MODE mode) {
		Mode = mode;
		StopAllCoroutines();
		StartCoroutine(LerpMode());
	}

	IEnumerator LerpMode() {
		float StartTime = Time.time;
		float EndTime = StartTime + TransitionTime;
	
		Vector3 StartSelfOffset = SelfOffset;
		Vector3 StartTargetOffset = TargetOffset;
		float StartDamping = Damping;

		Vector3 EndSelfOffset = Vector3.zero;
		Vector3 EndTargetOffset = Vector3.zero;
		float EndDamping = 0f;

		switch(Mode) {
			case MODE.SmoothFollow:
			EndSelfOffset = SelfOffset; //new Vector3(0f, 1f, -1.5f);
			EndTargetOffset = TargetOffset; //new Vector3(0f, 0.25f, 1f);
			EndDamping = 0.975f;
			break;

			case MODE.ConstantView:
			EndSelfOffset = new Vector3(1.5f, 0.5f, 0.25f);
			EndTargetOffset = new Vector3(0f, 0.5f, 0.25f);
			EndDamping = 0.0f;
			break;
			
			case MODE.FreeView:
			Vector3 euler = transform.rotation.eulerAngles;
			transform.rotation = Quaternion.Euler(0f, euler.y, 0f);
			ZeroRotation = transform.rotation;
			MousePosition = GetNormalizedMousePosition();
			LastMousePosition = GetNormalizedMousePosition();
			EndDamping = 1f;
			break;
		}

		while(Time.time < EndTime) {
			float ratio = (Time.time - StartTime) / TransitionTime;
			SelfOffset = Vector3.Lerp(StartSelfOffset, EndSelfOffset, ratio);
			TargetOffset = Vector3.Lerp(StartTargetOffset, EndTargetOffset, ratio);
			Damping = Mathf.Lerp(StartDamping, EndDamping, ratio);
			yield return 0;
		}

		SelfOffset = EndSelfOffset;
		TargetOffset = EndTargetOffset;
		Damping = EndDamping;
	
	}

	private Vector2 GetNormalizedMousePosition() {
		Vector2 ViewPortPosition = Camera.main.ScreenToViewportPoint(Input.mousePosition);
		return new Vector2(ViewPortPosition.x, ViewPortPosition.y);
	}

	private Vector2 GetNormalizedDeltaMousePosition() {
		return MousePosition - LastMousePosition;
	}

}