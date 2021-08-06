using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class ReloadScene : MonoBehaviour {

	public KeyCode Key = KeyCode.Escape;

	void Update () {
		if(Input.GetKeyDown(Key)) {
			SceneManager.LoadScene(SceneManager.GetActiveScene().name);
		}
	}

}
