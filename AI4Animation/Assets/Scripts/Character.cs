using UnityEngine;

public class Character {

	public Transform Transform;

	public float Speed = 3f;
	public float StrafeAmount = 1f;
	public float StrafeTarget = 1f;
	public float Phase = 0.0f;

	public Character(Transform t) {
		Transform = t;
	}

	public void Move(Vector2 direction) {
		Transform.position += Speed * Time.deltaTime * (Transform.rotation * new Vector3(direction.x,0,direction.y));
	}

	public void Turn(float direction) {
		Transform.rotation *= Quaternion.Euler(0f, Mathf.Rad2Deg * Speed * direction * Time.deltaTime, 0f);
	}

}
