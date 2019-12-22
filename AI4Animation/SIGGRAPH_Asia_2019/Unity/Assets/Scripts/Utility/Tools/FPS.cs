using UnityEngine;
 
public class FPS : MonoBehaviour {

	public bool Visualise = true;

	public float Size = 0.01f;
	public bool ShowTime = true;
	public bool ShowRate = true;

	private float Padding = 10;
	private float DeltaTime = 0.0f;
 
	void Update() {
		DeltaTime += (Time.deltaTime - DeltaTime) * 0.1f;
		if(Input.GetKeyDown(KeyCode.F1)) {
			Visualise = !Visualise;
		}
	}
 
	void OnGUI() {
		if(Visualise) {
			int size = Mathf.RoundToInt(0.01f*Screen.width);
			GUIStyle style = new GUIStyle();
			Rect rect = new Rect(Padding, Screen.height-Padding-size, Screen.width-2f*Padding, size);
			style.alignment = TextAnchor.MiddleRight;
			style.fontSize = size;
			style.normal.textColor = Color.black;
			float msec = DeltaTime * 1000.0f;
			float fps = 1.0f / DeltaTime;
			string text = string.Empty;
			if(ShowTime) {
				text += "[" + msec.ToString("F1") + " msec]";
			}
			text += " ";
			if(ShowRate) {
				text += "[" + Mathf.RoundToInt(fps) + " FPS]";
			}
			GUI.Label(rect, text, style);
		}
	}
}