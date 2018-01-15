using UnityEngine;
using System.Collections;

[ExecuteInEditMode]
public class CameraController : MonoBehaviour {

	public enum MODE {Follow, Side, Diagonal, Static}

	public MODE Mode = MODE.Follow;
	public Transform Target;
	public Vector3 SelfOffset = Vector3.zero;
	public Vector3 TargetOffset = Vector3.zero;
	[Range(0f, 1f)] public float Damping = 0.95f;
	public float TransitionTime = 1f;
	public float MinHeight = 1f;

	void Update() {
		if(Input.GetKeyDown(KeyCode.F1)) {
			SetMode(MODE.Follow);
		}
		if(Input.GetKeyDown(KeyCode.F2)) {
			SetMode(MODE.Side);
		}
		if(Input.GetKeyDown(KeyCode.F3)) {
			SetMode(MODE.Diagonal);
		}
		if(Input.GetKeyDown(KeyCode.F4)) {
			SetMode(MODE.Static);
		}

		if(Target == null) {
			return;
		}

		Vector3 currentPosition = transform.position;
		Quaternion currentRotation = transform.rotation;

		//Determine final
		transform.position = Target.position + Target.rotation * SelfOffset;
		transform.LookAt(Target.position + Target.rotation * TargetOffset);

		//Lerp
		transform.position = Vector3.Lerp(currentPosition, transform.position, 1f-Damping);
		transform.rotation = Quaternion.Lerp(currentRotation, transform.rotation, 1f-Damping);

		//Correct height
		float height = transform.position.y - Target.position.y;
		if(height < MinHeight) {
			transform.position += new Vector3(0f, MinHeight-height, 0f);
		}
	}

	private void SetMode(MODE mode) {
		if(Mode == mode) {
			return;
		}
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
			case MODE.Follow:
			EndSelfOffset = new Vector3(0f, 1f, -1.5f);
			EndTargetOffset = new Vector3(0f, 0.5f, 0f);
			EndDamping = 0.975f;
			break;

			case MODE.Side:
			EndSelfOffset = new Vector3(1.5f, 0.5f, 0.25f);
			EndTargetOffset = new Vector3(0f, 0.5f, 0.25f);
			EndDamping = 0.0f;
			break;
			
			case MODE.Diagonal:
			EndSelfOffset = new Vector3(-1.5f, 1f, -0.75f);
			EndTargetOffset = new Vector3(0f, 0.5f, 1f);
			EndDamping = 0.975f;
			break;

			case MODE.Static:
			EndSelfOffset = new Vector3(0f, 0f, 0f);
			EndTargetOffset = new Vector3(0f, 0f, 0f);
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

}