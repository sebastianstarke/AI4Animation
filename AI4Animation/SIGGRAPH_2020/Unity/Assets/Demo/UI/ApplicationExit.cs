using UnityEngine;

public class ApplicationExit : MonoBehaviour {

    public KeyCode Key;

    void Update() {
        if(Input.GetKeyDown(Key)) {
            Application.Quit();
        }
    }

}
