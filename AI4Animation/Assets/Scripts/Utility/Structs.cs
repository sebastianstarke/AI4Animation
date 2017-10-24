using UnityEngine;

public struct Transformation {
	public Vector3 Position;
	public Quaternion Rotation;
	public Transformation(Transform t) {
		Position = t.position;
		Rotation = t.rotation;
	}
	public Transformation(Vector3 position, Quaternion rotation) {
		Position = position;
		Rotation = rotation;
	}
	public Transformation(Vector3 position) {
		Position = position;
		Rotation = Quaternion.identity;
	}
	public Transformation(Quaternion rotation) {
		Position = Vector3.zero;
		Rotation = rotation;
	}
}
