using UnityEngine;

public class CameraOrbit : MonoBehaviour {

	public Transform[] Targets;
	public float Pivot = 1f;
	public float Height = 1f;
	public float Distance = 1f;
	public float Speed = 10f;
	public float Smoothing = 0.95f;

	private float Angle = 0f;
	private Vector3 TargetPosition;

	void Start() {
		TargetPosition = GetFocus();;
	}

	private Vector3 GetFocus() {
		Vector3 center = Vector3.zero;
		for(int i=0; i<Targets.Length; i++) {
			center += Targets[i].position;
		}
		return center / Targets.Length;
	}

	void Update() {
		TargetPosition = Vector3.Lerp(TargetPosition, GetFocus(), Smoothing);
		transform.position = TargetPosition + Quaternion.AngleAxis(Angle, Vector3.up) * new Vector3(0f, Height, Distance);
		Vector3 a = transform.position;
		Vector3 b = TargetPosition;
		a.y = 0f;
		b.y = 0f;
		transform.rotation = Quaternion.LookRotation(b-a, Vector3.up);
		Vector3 targetPosition = TargetPosition;
		targetPosition.y += Pivot;
		transform.LookAt(targetPosition, Vector3.up);
		Angle += Speed * Time.deltaTime;
	}

}
