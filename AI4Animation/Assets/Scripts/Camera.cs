using UnityEngine;

public class Camera : MonoBehaviour {

	public Transform Target;
	public Vector3 OffsetPosition = Vector3.zero;
	public Vector3 OffsetOrientation = Vector3.zero;
	[Range(0f,1f)] public float Speed = 1f/3f;
	public float MinHeight = 1f;

	void Update() {
		Vector3 currentPosition = transform.position;
		Quaternion currentRotation = transform.rotation;

		//Determine Position
		Vector3 newPosition = Target.transform.position + Target.transform.rotation * OffsetPosition;
		float height = transform.position.y - Target.position.y;
		if(height < MinHeight) {
			newPosition.y += MinHeight-height;
		}

		//Determine Rotation
		Quaternion newRotation = Target.transform.rotation * Quaternion.Euler(OffsetOrientation);

		transform.position = Vector3.Lerp(currentPosition, newPosition, Speed);
		transform.rotation = Quaternion.Lerp(currentRotation, newRotation, Speed);
	}

}