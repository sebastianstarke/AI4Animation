using UnityEngine;

public class FreeCamera : MonoBehaviour {
	public bool enableInputCapture = true;
	public bool holdRightMouseCapture = false;

	public float lookSpeed = 5f;
	public float moveSpeed = 5f;
	public float sprintSpeed = 50f;

	bool	m_inputCaptured;
	float	m_yaw;
	float	m_pitch;
	
	void Awake() {
		enabled = enableInputCapture;
	}

	void OnValidate() {
		if(Application.isPlaying)
			enabled = enableInputCapture;
	}

	void CaptureInput() {
		Cursor.lockState = CursorLockMode.Locked;

		Cursor.visible = false;
		m_inputCaptured = true;

		m_yaw = transform.eulerAngles.y;
		m_pitch = transform.eulerAngles.x;
	}

	void ReleaseInput() {
		Cursor.lockState = CursorLockMode.None;
		Cursor.visible = true;
		m_inputCaptured = false;
	}

	void OnApplicationFocus(bool focus) {
		if(m_inputCaptured && !focus)
			ReleaseInput();
	}

	void Update() {
		if(!m_inputCaptured) {
			if(!holdRightMouseCapture && Input.GetMouseButtonDown(0)) 
				CaptureInput();
			else if(holdRightMouseCapture && Input.GetMouseButtonDown(1))
				CaptureInput();
		}

		if(!m_inputCaptured)
			return;

		if(m_inputCaptured) {
			if(!holdRightMouseCapture && Input.GetKeyDown(KeyCode.Escape))
				ReleaseInput();
			else if(holdRightMouseCapture && Input.GetMouseButtonUp(1))
				ReleaseInput();
		}

		var rotStrafe = Input.GetAxis("Mouse X");
		var rotFwd = Input.GetAxis("Mouse Y");

		m_yaw = (m_yaw + lookSpeed * rotStrafe) % 360f;
		m_pitch = (m_pitch - lookSpeed * rotFwd) % 360f;
		transform.rotation = Quaternion.AngleAxis(m_yaw, Vector3.up) * Quaternion.AngleAxis(m_pitch, Vector3.right);

		var speed = Time.deltaTime * (Input.GetKey(KeyCode.LeftShift) ? sprintSpeed : moveSpeed);
		var forward = speed * Input.GetAxis("Vertical");
		var right = speed * Input.GetAxis("Horizontal");
		var up = speed * ((Input.GetKey(KeyCode.E) ? 1f : 0f) - (Input.GetKey(KeyCode.Q) ? 1f : 0f));
		transform.position += transform.forward * forward + transform.right * right + Vector3.up * up;
	}
}
