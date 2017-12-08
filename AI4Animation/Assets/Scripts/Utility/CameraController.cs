using UnityEngine;

[ExecuteInEditMode]
public class CameraController : MonoBehaviour {

	public Transform Target;
	public Vector3 SelfOffset = Vector3.zero;
	public Vector3 TargetOffset = Vector3.zero;
	[Range(0f,1f)] public float MinSpeed = 0.1f;
	[Range(0f,1f)] public float MaxSpeed = 0.5f;
	public float MinHeight = 1f;

	void Update() {
		if(Target == null) {
			return;
		}

		Vector3 currentPosition = transform.position;
		Quaternion currentRotation = transform.rotation;

		//Determine final
		transform.position = Target.position + Target.rotation * SelfOffset;
		transform.LookAt(Target.position + TargetOffset);

		float speed = Vector3.Angle(currentRotation*Vector3.forward, (Target.position + TargetOffset) - transform.position) / 180f * MaxSpeed;
		transform.position = Vector3.Lerp(currentPosition, transform.position, Mathf.Max(MinSpeed, speed));
		transform.rotation = Quaternion.Lerp(currentRotation, transform.rotation, Mathf.Max(MinSpeed, speed));

		//Correct height
		float height = transform.position.y - Target.position.y;
		if(height < MinHeight) {
			transform.position += new Vector3(0f, MinHeight-height, 0f);
		}
	}

}