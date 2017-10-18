using UnityEngine;
 
public class FPS : MonoBehaviour {
	public int Size = 10;
	float deltaTime = 0.0f;
 
	void Update()
	{
		deltaTime += (Time.deltaTime - deltaTime) * 0.1f;
	}
 
	void OnGUI()
	{
		int w = Screen.width, h = Screen.height;
 
		GUIStyle style = new GUIStyle();
 
		Rect rect = new Rect(0, 0, w, Size);
		style.alignment = TextAnchor.UpperCenter;
		style.fontSize = Size;
		style.normal.textColor = new Color (1.0f, 1.0f, 1.0f, 1.0f);
		float msec = deltaTime * 1000.0f;
		float fps = 1.0f / deltaTime;
		string text = string.Format("{0:0.0} ms ({1:0.} fps)", msec, fps);
		GUI.Label(rect, text, style);
	}
}