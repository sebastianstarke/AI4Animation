using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class GamePad {
	
	public static float GetX() {
		return Input.GetAxis("Horizontal");
	}

	public static float GetY() {
		return Input.GetAxis("Vertical");
	}

	public static bool GetButtonA() {
		return Input.GetButton("Fire1");
	}

	public static bool GetButtonB() {
		return Input.GetButton("Fire2");
	}

	public static bool GetButtonX() {
		return Input.GetButton("Fire3");
	}

	public static bool GetButtonY() {
		return Input.GetButton("Fire4");
	}

}
