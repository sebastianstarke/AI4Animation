using UnityEngine;
using UnityEngine.UI;

public class QuadrupedDemo_UI : MonoBehaviour {

    public BinaryButton ShowUIElements;
    public BinaryButton ShowDebugLines;
    public BinaryButton UseKeyboard;
    public BinaryButton UseGamepad;
    public GameObject KeyboardInfo;
    public GameObject GamepadInfo;
    public QuadrupedController Player;

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
        SetState(UseKeyboard, Player.ControlType == Controller.TYPE.Keyboard);
        SetState(UseGamepad, Player.ControlType == Controller.TYPE.Gamepad);
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
        if(button == UseKeyboard.Button) {
            SetState(UseKeyboard, true);
            SetState(UseGamepad, false);
        }
        if(button == UseGamepad.Button) {
            SetState(UseKeyboard, false);
            SetState(UseGamepad, true);
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
        if(button == UseKeyboard && state) {
            Player.ControlType = Controller.TYPE.Keyboard;
            KeyboardInfo.SetActive(true);
            GamepadInfo.SetActive(false);
        }
        if(button == UseGamepad && state) {
            Player.ControlType = Controller.TYPE.Gamepad;
            KeyboardInfo.SetActive(false);
            GamepadInfo.SetActive(true);
        }
    }

}
