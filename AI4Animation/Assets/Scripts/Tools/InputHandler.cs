using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InputHandler : MonoBehaviour {

	private static List<List<KeyCode>> Keys = new List<List<KeyCode>>();
	private static int Capacity = 2;
	private static int Clients = 0;

	void OnEnable() {
		Clients += 1;
	}

	void OnDisable() {
		Clients -= 1;
	}

	public static bool anyKey {
		get{
			for(int i=0; i<Keys.Count; i++) {
				if(Keys[i].Count > 0) {
					return true;
				}
			}
			return false;
		}
	}

	void Update () {
		while(Keys.Count >= Capacity) {
			Keys.RemoveAt(0);
		}
		List<KeyCode> state = new List<KeyCode>();
		foreach(KeyCode k in Enum.GetValues(typeof(KeyCode))) {
			if(Input.GetKey(k)) {
				state.Add(k);
			}
		}
		Keys.Add(state);
	}

	public static bool GetKey(KeyCode k) {
		if(Clients == 0) {
			return Input.GetKey(k);
		} else {
			for(int i=0; i<Keys.Count; i++) {
				if(Keys[i].Contains(k)) {
					return true;
				}
			}
			return false;
		}
	}
	
}
