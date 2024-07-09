using System;
using UnityEngine;

public class SceneLoader : MonoBehaviour {

    [Serializable]
    public class Option {
        public KeyCode Key = KeyCode.Escape;
        public string Scene = string.Empty;
    }

    public Option[] Options;

    void Update() {
        foreach(Option option in Options) {
            if(Input.GetKeyDown(option.Key)) {
                LoadScene(option.Scene);
                break;
            }
        }
    }

    public void LoadScene(string scene) {
        UnityEngine.SceneManagement.SceneManager.LoadScene(scene);
    }
}