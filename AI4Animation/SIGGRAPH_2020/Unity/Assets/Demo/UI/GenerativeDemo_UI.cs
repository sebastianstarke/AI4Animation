using UnityEngine;
using UnityEngine.UI;
using DeepLearning;

public class GenerativeDemo_UI : MonoBehaviour {

    public BinaryButton ShowUIElements;
    public BinaryButton ShowDebugLines;
    public BinaryButton ShowGenerativeSpace;
    public GenerativeController Player;

    private Color InactiveColor = new Color(150f/255f, 150f/255f, 150f/255f);
    private Color ActiveColor = new Color(250f/255f, 180f/255f, 0f);
    private Color InactiveTextColor = new Color(0.8f, 0.8f, 0.8f);
    private Color ActiveTextColor = Color.white;

    #if UNITY_EDITOR
    public bool EnablePausing = true;
    #endif

    [System.Serializable]
    public class BinaryButton {
        public bool State;
        public Button Button;
    }

    void Start() {
        SetState(ShowUIElements, Player.DrawGUI);
        SetState(ShowDebugLines, Player.DrawDebug);
        SetState(ShowGenerativeSpace, Player.GetComponent<GenerativeControl>().Draw);
    }

    #if UNITY_EDITOR
    void Update() {
        if(EnablePausing && Input.GetButtonDown("1ButtonX")) {
            UnityEditor.EditorApplication.isPaused = true;
        }
    }
    #endif

    public void Callback(Button button) {
        if(button == ShowUIElements.Button) {
            ToggleState(ShowUIElements);
        }
        if(button == ShowDebugLines.Button) {
            ToggleState(ShowDebugLines);
        }
        if(button == ShowGenerativeSpace.Button) {
            ToggleState(ShowGenerativeSpace);
        }
    }

    private void ToggleState(BinaryButton button) {
        SetState(button, !button.State);
    }

    private void SetState(BinaryButton button, bool state) {
        Image image = button.Button.GetComponent<Image>();
        image.color = state ? ActiveColor : InactiveColor;
        Text text = button.Button.GetComponentInChildren<Text>();
        text.color = state ? ActiveTextColor : InactiveTextColor;
        button.State = state;
        if(button == ShowUIElements) {
            Player.DrawGUI = button.State;
        }
        if(button == ShowDebugLines) {
            Player.DrawDebug = button.State;
        }
        if(button == ShowGenerativeSpace) {
            Player.GetComponent<GenerativeControl>().Draw = button.State;
        }
    }

}
