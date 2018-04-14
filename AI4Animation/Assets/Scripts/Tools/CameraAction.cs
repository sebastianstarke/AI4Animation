using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class CameraAction : MonoBehaviour {

	public CameraController Camera;
	public Vector3 Size = Vector3.one;

	public CameraController.MODE Mode = CameraController.MODE.Follow;
	[Range(0f, 1f)] public float Damping = 0.975f;
	[Range(-180f, 180f)] public float Yaw = 0f;
	[Range(-45f, 45f)] public float Pitch = 0f;
	[Range(0f, 10f)] public float FOV = 1f;
	
	[Range(0f, 1f)] public float Speed = 0.01f;

	void LateUpdate() {
		if(IsActive()) {
			Camera.Mode = Mode;
			Camera.Damping = Utility.Interpolate(Camera.Damping, Damping, Speed);
			Camera.Yaw = Utility.Interpolate(Camera.Yaw, Yaw, Speed);
			Camera.Pitch = Utility.Interpolate(Camera.Pitch, Pitch, Speed);
			Camera.FOV = Utility.Interpolate(Camera.FOV, FOV, Speed);
		}
	}

	private bool IsActive() {
		if(Camera.Target == null) {
			return false;
		}
		Vector3 position = Camera.Target.position;
		Vector3 minimum = GetMinimum();
		Vector3 maximum = GetMaximum();
		return	position.x >= minimum.x && position.x <= maximum.x
				&& position.y >= minimum.y && position.y <= maximum.y
				&& position.z >= minimum.z && position.z <= maximum.z;
	}

	private Vector3 GetMinimum() {
		return transform.position - Size/2f;
	}

	private Vector3 GetMaximum() {
		return transform.position + Size/2f;
	}

	void OnDrawGizmos() {
		if(IsActive()) {
			Gizmos.color = UltiDraw.Orange;
		} else {
			Gizmos.color = UltiDraw.Cyan;
		}
		Gizmos.DrawWireCube(transform.position, Size);
	}

}
