using UnityEngine;
 
public class FPS : MonoBehaviour {
	public int Size = 15;

	private float Padding = 10;
	private float DeltaTime = 0.0f;
 
	void Update() {
		DeltaTime += (Time.deltaTime - DeltaTime) * 0.1f;
	}
 
	void OnGUI() {
		GUIStyle style = new GUIStyle();
		Rect rect = new Rect(Padding, Screen.height-Padding-Size, Screen.width-2f*Padding, Size);
		style.alignment = TextAnchor.MiddleRight;
		style.fontSize = Size;
		style.normal.textColor = Color.black;
		float msec = DeltaTime * 1000.0f;
		float fps = 1.0f / DeltaTime;
		string text = string.Format("{0:0.0} ms ({1:0.} fps)", msec, fps);
		GUI.Label(rect, text, style);
	}
}