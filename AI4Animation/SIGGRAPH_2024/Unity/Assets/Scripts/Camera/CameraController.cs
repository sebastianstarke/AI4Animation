using UnityEngine;
using System.Collections;

public class CameraController : MonoBehaviour {

	public enum MODE {FreeView, Follow, LookAt, FixedView}

	public bool ShowGUI = true;
	public MODE Mode = MODE.Follow;
	public Transform Target;
	public Transform[] Targets;
	public Vector3 SelfOffset = Vector3.zero;
	public Vector3 TargetOffset = Vector3.zero;
	public float MaxFollowAngle = 30f;
	[Range(0f, 1f)] public float Damping = 0.975f;
	[Range(-180f, 180f)] public float Yaw = 0f;
	[Range(-45f, 45f)] public float Pitch = 0f;
	[Range(0f, 10f)] public float FOV = 1.5f;
	public float MinHeight = 0.5f;
	public float ImmediateThreshold = 0f;

	private float Velocity = 5f;
	private float AngularVelocity = 5f;
	private float ZoomVelocity = 10;
	private float Sensitivity = 1f;
	private Vector2 MousePosition;
	private Vector2 LastMousePosition;
	private Vector3 DeltaRotation;
	private Quaternion ZeroRotation;

	private GUIStyle ButtonStyle;
	private GUIStyle SliderStyle;
	private GUIStyle ThumbStyle;
	private GUIStyle FontStyle;

	public float X = 0.85f;
	public float Y = 0.05f;
	private float YStep = 0.05f;

	void Start() {
		SetMode(Mode);
	}

	void Update() {
		if(Input.GetKeyDown(KeyCode.F1)) {
			ShowGUI = !ShowGUI;
		}
	}

	void LateUpdate() {
		// //Correct Height
		// if(Target != null) {
		// 	float height = transform.position.y - Target.position.y;
		// 	if(height < MinHeight) {
		// 		transform.position += new Vector3(0f, MinHeight-height, 0f);
		// 	}
		// }
	}

	private GUIStyle GetButtonStyle() {
		if(ButtonStyle == null) {
			ButtonStyle = new GUIStyle(GUI.skin.button);
			ButtonStyle.font = (Font)Resources.Load("Fonts/Coolvetica");
			ButtonStyle.normal.textColor = Color.white;
			ButtonStyle.alignment = TextAnchor.MiddleCenter;
		}
		return ButtonStyle;
	}

	private GUIStyle GetSliderStyle() {
		if(SliderStyle == null) {
			SliderStyle = new GUIStyle(GUI.skin.horizontalSlider);
		}
		return SliderStyle;
	}

	private GUIStyle GetThumbStyle() {
		if(ThumbStyle == null) {
			ThumbStyle = new GUIStyle(GUI.skin.horizontalSliderThumb);
		}
		return ThumbStyle;
	}

	private GUIStyle GetFontStyle() {
		if(FontStyle == null) {
			FontStyle = new GUIStyle();
			FontStyle.font = (Font)Resources.Load("Fonts/Coolvetica");
			FontStyle.normal.textColor = Color.white;
			FontStyle.alignment = TextAnchor.MiddleLeft;
		}
		return FontStyle;
	}

	private IEnumerator UpdateFreeCamera() {
		Vector3 euler = transform.rotation.eulerAngles;
		transform.rotation = Quaternion.Euler(0f, euler.y, 0f);
		ZeroRotation = transform.rotation;
		while(Mode == MODE.FreeView) {
			yield return new WaitForEndOfFrame();
			MousePosition = GetNormalizedMousePosition();

			Vector3 currentPosition = transform.position;
			Quaternion currentRotation = transform.rotation;

			//Translation
			Vector3 direction = Vector3.zero;
			if(Input.GetKey(KeyCode.A)) {
				direction.x -= 1f;
			}
			if(Input.GetKey(KeyCode.D)) {
				direction.x += 1f;
			}
			if(Input.GetKey(KeyCode.W)) {
				direction.z += 1f;
			}
			if(Input.GetKey(KeyCode.S)) {
				direction.z -= 1f;
			}
			transform.position += Velocity*Sensitivity*Time.smoothDeltaTime*(transform.rotation*direction);

			//Zoom
			if(Input.mouseScrollDelta.y != 0) {
				transform.position += ZoomVelocity*Sensitivity*Time.smoothDeltaTime*Input.mouseScrollDelta.y*transform.forward;
			}

			//Rotation
			MousePosition = GetNormalizedMousePosition();
			if(Input.GetMouseButton(0)) {
				DeltaRotation += 1000f*AngularVelocity*Sensitivity*Time.smoothDeltaTime*new Vector3(GetNormalizedDeltaMousePosition().x, GetNormalizedDeltaMousePosition().y, 0f);
				transform.rotation = ZeroRotation * Quaternion.Euler(-DeltaRotation.y, DeltaRotation.x, 0f);
			}

			LastMousePosition = MousePosition;
		}
	}

	private IEnumerator UpdateFollowCamera() {
		while(Mode == MODE.Follow) {
			yield return new WaitForEndOfFrame();
			Vector3 currentPosition = transform.position;
			Quaternion currentRotation = transform.rotation;

			//Determine Target
			Vector3 _selfOffset = FOV * SelfOffset;
			Vector3 _targetOffset = TargetOffset;
			Vector3 forward = Vector3.RotateTowards(
				(Target.position - transform.position).ZeroY().normalized,
				Target.forward,
				Mathf.Deg2Rad*MaxFollowAngle,
				0f
			);
			Quaternion targetRotation = Quaternion.LookRotation(forward, Vector3.up);
			transform.position = Target.position + targetRotation * _selfOffset;
			transform.RotateAround(Target.position + targetRotation * _targetOffset, Vector3.up, Yaw);
			transform.RotateAround(Target.position + targetRotation * _targetOffset, transform.right, Pitch);
			transform.LookAt(Target.position + targetRotation * _targetOffset);
			//

			transform.position = Vector3.Lerp(currentPosition, transform.position, 1f-GetDamping());
			transform.rotation = Quaternion.Lerp(currentRotation, transform.rotation, 1f-GetDamping());
		}
	}

	private IEnumerator UpdateLookAtCamera() {
		while(Mode == MODE.LookAt) {
			yield return new WaitForEndOfFrame();
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
			transform.position += Velocity*Sensitivity*Time.smoothDeltaTime*(transform.rotation*direction);

			//Zoom
			if(Input.mouseScrollDelta.y != 0) {
				transform.position += ZoomVelocity*Sensitivity*Time.smoothDeltaTime*Input.mouseScrollDelta.y*transform.forward;
			}

			//Rotation
			if(Targets.Length > 0) {
				Vector3[] positions = new Vector3[Targets.Length];
				for(int i=0; i<positions.Length; i++) {
					positions[i] = Targets[i].position;
				}
				transform.LookAt(positions.Mean());
			} else {
				transform.LookAt(Target);
			}

			transform.position = Vector3.Lerp(currentPosition, transform.position, 1f-GetDamping());
			transform.rotation = Quaternion.Lerp(currentRotation, transform.rotation, 1f-GetDamping());
		}
	}

	private IEnumerator UpdateFixedCamera() {
		while(Mode == MODE.FixedView) {
			yield return new WaitForEndOfFrame();
			Vector3 currentPosition = transform.position;
			Quaternion currentRotation = transform.rotation;

			Vector3 position = GetTarget().GetPosition();

			transform.position = position + FOV*SelfOffset;
			transform.LookAt(position + TargetOffset);

			transform.position = Vector3.Lerp(currentPosition, transform.position, 1f-GetDamping());
			transform.rotation = Quaternion.Lerp(currentRotation, transform.rotation, 1f-GetDamping());
		}
	}

	public Matrix4x4 GetTarget() {
		if(Target != null) {
			return Target.GetWorldMatrix();
		}
		if(Targets.Length > 0) {
			Matrix4x4[] targets = new Matrix4x4[Targets.Length];
			for(int i=0; i<Targets.Length; i++) {
				targets[i] = Targets[i].GetWorldMatrix();
			}
			return targets.Mean();
		}
		return Matrix4x4.identity;
	}

	public void SetMode(MODE mode) {
		StopAllCoroutines();
		Mode = mode;
		switch(Mode) {
			case MODE.FreeView:
			StartCoroutine(UpdateFreeCamera());
			break;

			case MODE.Follow:
			StartCoroutine(UpdateFollowCamera());
			break;

			case MODE.LookAt:
			StartCoroutine(UpdateLookAtCamera());
			break;

			case MODE.FixedView:
			StartCoroutine(UpdateFixedCamera());
			break;
		}
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

	Color BackgroundColor = UltiDraw.White;
	void OnGUI() {
		if(!ShowGUI) {
			return;
		}
		GetButtonStyle().fontSize = Mathf.RoundToInt(0.01f * Screen.width);
		GetSliderStyle().fixedHeight = Mathf.RoundToInt(0.01f * Screen.width);
		GetThumbStyle().fixedHeight = Mathf.RoundToInt(0.01f * Screen.width);
		GetThumbStyle().fixedWidth = Mathf.RoundToInt(0.01f * Screen.width);
		GetFontStyle().fixedHeight = Mathf.RoundToInt(0.01f * Screen.width);
		GetFontStyle().fontSize = Mathf.RoundToInt(0.01f * Screen.width);
		GUI.color = UltiDraw.White;
		GUI.backgroundColor = Mode == MODE.Follow ? UltiDraw.Mustard : BackgroundColor;
		if(GUI.Button(Utility.GetGUIRect(X, Y, 0.1f, 0.04f), "Follow", GetButtonStyle())) {
			SetMode(MODE.Follow);
		}
		GUI.backgroundColor = Mode == MODE.LookAt ? UltiDraw.Mustard : BackgroundColor;
		if(GUI.Button(Utility.GetGUIRect(X, Y+1*YStep, 0.1f, 0.04f), "Look At", GetButtonStyle())) {
			SetMode(MODE.LookAt);
		}
		GUI.backgroundColor = Mode == MODE.FreeView ? UltiDraw.Mustard : BackgroundColor;
		if(GUI.Button(Utility.GetGUIRect(X, Y+2*YStep, 0.1f, 0.04f), "Free View", GetButtonStyle())) {
			SetMode(MODE.FreeView);
		}
		GUI.backgroundColor = Mode == MODE.FixedView ? UltiDraw.Mustard : BackgroundColor;
		if(GUI.Button(Utility.GetGUIRect(X, Y+3*YStep, 0.1f, 0.04f), "Fixed View", GetButtonStyle())) {
			SetMode(MODE.FixedView);
		}
		GUI.color = BackgroundColor;
		FOV = GUI.HorizontalSlider(Utility.GetGUIRect(X, Y-0.5f*YStep, 0.1f, 0.025f), FOV, 0f, 10f, GetSliderStyle(), GetThumbStyle());
		GUI.Label(Utility.GetGUIRect(X-0.04f, Y-0.5f*YStep, 0.04f, 0.025f), "FOV", GetFontStyle());
		if(Mode == MODE.Follow) {
			Yaw = Mathf.RoundToInt(GUI.HorizontalSlider(Utility.GetGUIRect(X, Y-1.5f*YStep+0.025f, 0.1f, 0.025f), Yaw, -180f, 180f, GetSliderStyle(), GetThumbStyle()));
			GUI.Label(Utility.GetGUIRect(X-0.04f, Y-1.5f*YStep+0.025f, 0.04f, 0.025f), "Yaw", GetFontStyle());
			Pitch = Mathf.RoundToInt(GUI.HorizontalSlider(Utility.GetGUIRect(X, Y-2.5f*YStep+0.05f, 0.1f, 0.025f), Pitch, -45f, 45f, GetSliderStyle(), GetThumbStyle()));
			GUI.Label(Utility.GetGUIRect(X-0.04f, Y-2.5f*YStep+0.05f, 0.04f, 0.025f), "Pitch", GetFontStyle());
			Damping = GUI.HorizontalSlider(Utility.GetGUIRect(X, Y-3.5f*YStep+0.075f, 0.1f, 0.025f), Damping, 0f, 1f, GetSliderStyle(), GetThumbStyle());
			GUI.Label(Utility.GetGUIRect(X-0.04f, Y-3.5f*YStep+0.075f, 0.04f, 0.025f), "Damping", GetFontStyle());
		}
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