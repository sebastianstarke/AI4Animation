using UnityEngine;
using UnityEngine.SceneManagement;

public class SceneLoader : MonoBehaviour {

    [System.Serializable]
    public class Trigger {
        public KeyCode Code;
        public string Name;
    }

    public Trigger[] Triggers = new Trigger[0];

    void Update() {
        for(int i=0; i<Triggers.Length; i++) {
            if(Input.GetKeyDown(Triggers[i].Code)) {
                LoadByName(Triggers[i].Name);
                break;
            }
        }
    }

    public void LoadByName(string name) {
        SceneManager.LoadScene(name);
    }

}