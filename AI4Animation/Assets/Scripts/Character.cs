using UnityEngine;

public class Character {

	public Transform Transform;

	public Vector3 Velocity = Vector3.zero;
	public float Phase = 0.0f;

	public Character(Transform t) {
		Transform = t;
	}

	public void Move(Vector2 direction) {
		float acceleration = 5f;
		float damping = 2f;

		Velocity = Utility.Interpolate(Velocity, Vector3.zero, damping * Time.deltaTime);
		Velocity += acceleration * Time.deltaTime * (new Vector3(direction.x, 0f, direction.y).normalized);
		Transform.position += Velocity * Time.deltaTime;
	}

	public void Turn(float direction) {
		Transform.Rotate(0f, 100f*direction*Time.deltaTime, 0f);
	}

}
