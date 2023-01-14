using System.Collections;
using UnityEngine;

public class ControlResponse : MonoBehaviour {

	public UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.5f, 0.5f, 0.6f, 0.2f);
	public AnimationController Animation = null;

    public float FontSize = 1f;

    public float Framerate = 30f;
    
    private bool Measuring = false;
    private float Time = 0f;

    void Start() {

    }

	void LateUpdate () {
        if(Animation != null) {
            if(Condition() && !Measuring) {
                StartCoroutine(Measure());
            }
        }
    }
    
    void OnGUI() {
        UltiDraw.Begin();
        UltiDraw.OnGUILabel(Rect.GetCenter(), Rect.GetSize(), FontSize, "Time: " + Time.Round(3), UltiDraw.White, UltiDraw.Black.Opacity(0.1f));
        UltiDraw.End();
    }

    private bool Condition() {
        // return Animation.GetController().GetLogic("Move").Query();
        return Input.GetButton("1ButtonX");
    }

    private IEnumerator Measure() {
        Measuring = true;
        Time = 0f;
        while(Condition()) {
            Time += 1f/Framerate;
            yield return 0;
        }
        Measuring = false;
    }

}
