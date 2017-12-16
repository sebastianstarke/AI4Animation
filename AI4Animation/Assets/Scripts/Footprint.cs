using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Footprint : MonoBehaviour {

	public GameObject Decal;
	public Vector3 Forward = Vector3.forward;
	public Vector3 Up = Vector3.up;
	public bool Mirror = false;
	public float Radius = 0.025f;
	public float TimeToLive = 10f;
	public bool Collision = false;

	void Update() {
		if(Decal == null) {
			return;
		}
		bool collision = Physics.CheckSphere(transform.position, Radius, LayerMask.GetMask("Ground"));
		if(collision != Collision) {
			Collision = collision;
			if(Collision) {
				Vector3 position = transform.position;
				Quaternion rotation = Quaternion.LookRotation(transform.rotation * Forward, transform.rotation * Up);
				GameObject instance = Instantiate(Decal, position, rotation);
				if(Mirror) {
					Vector3 scale = instance.transform.localScale;
					scale.x *= -1f;
					instance.transform.localScale = scale;
				}
				Destroy(instance, TimeToLive);
			}
		}
	}

	void OnDrawGizmosSelected() {
		Gizmos.color = Color.red;
		Gizmos.DrawWireSphere(transform.position, Radius);

		Gizmos.color = Color.blue;
		Gizmos.DrawLine(transform.position, transform.position + transform.rotation * (0.25f * Forward));

		Gizmos.color = Color.green;
		Gizmos.DrawLine(transform.position, transform.position + transform.rotation * (0.25f * Up));
	}

}
