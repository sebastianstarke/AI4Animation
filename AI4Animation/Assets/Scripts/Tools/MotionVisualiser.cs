using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

[RequireComponent(typeof(Actor))]
public class MotionVisualiser : MonoBehaviour {

	public string Path = "Motion.txt";
	public int Framerate = 60;

	private Actor Actor;
	private List<float[]> Motions = new List<float[]>();

	private GUIStyle FontStyle;
	private GUIStyle TextFieldStyle;

	void Awake() {
		Actor = GetComponent<Actor>();
	}

	void Start() {
		Application.targetFrameRate = 60;
	}

	private GUIStyle GetFontStyle() {
		if(FontStyle == null) {
			FontStyle = new GUIStyle();
			FontStyle.font = (Font)Resources.Load("Fonts/Coolvetica");
			FontStyle.normal.textColor = Color.white;
			FontStyle.alignment = TextAnchor.MiddleLeft;
		}
		return FontStyle;
	}

	private GUIStyle GetTextFieldStyle() {
		if(TextFieldStyle == null) {
			TextFieldStyle = new GUIStyle(GUI.skin.textField);
			TextFieldStyle.alignment = TextAnchor.MiddleLeft;
		}
		return TextFieldStyle;
	}

	void Update() {
		if(Motions.Count == 0) {
			Initialise();
		}
		if(Motions.Count > 0) {
			float[] motion = Motions[0];
			ApplyMotion(motion);
			Motions.RemoveAt(0);
		}
	}

	void OnGUI() {
		GetFontStyle().fontSize = Mathf.RoundToInt(0.0075f * Screen.width);
		GetTextFieldStyle().fontSize = Mathf.RoundToInt(0.0075f * Screen.width);
		GUI.Label(Utility.GetGUIRect(0.025f, 0.025f, 0.025f, 0.025f), "Path", GetFontStyle());
		Path = GUI.TextField(Utility.GetGUIRect(0.05f, 0.025f, 0.9f, 0.025f), Path, GetTextFieldStyle());
		if(!IsPathValid()) {
			GUI.Label(Utility.GetGUIRect(0.05f, 0.05f, 0.9f, 0.025f), "File at path " + GetFilePath() + " does not exist.", GetFontStyle());
		}
	}

	private void Initialise() {
		if(LoadMotions()) {
			Actor.transform.position = Vector3.zero;
			Actor.transform.rotation = Quaternion.identity;
		}
	}

	private void ApplyMotion(float[] motion) {
		Vector3[] positions = new Vector3[Actor.Bones.Length];
		for(int i=0; i<positions.Length; i++) {
			positions[i] = new Vector3(motion[i*3 + 0], motion[i*3 + 1], motion[i*3 + 2]);
		}
		Vector3 rootMotion = new Vector3(motion[motion.Length-3], motion[motion.Length-2], motion[motion.Length-1]) / (float)Framerate;
		
		Matrix4x4 delta = Matrix4x4.TRS(new Vector3(rootMotion.x, 0f, rootMotion.z), Quaternion.AngleAxis(rootMotion.y, Vector3.up), Vector3.one);
		Matrix4x4 root = transform.GetWorldMatrix() * delta;

		transform.position = root.GetPosition();
		transform.rotation = root.GetRotation();
		for(int i=0; i<Actor.Bones.Length; i++) {
			Actor.Bones[i].Transform.position = positions[i].GetRelativePositionFrom(root);
		}
	}

	private bool LoadMotions() {
		if(IsPathValid()) {
			string[] lines = File.ReadAllLines(GetFilePath());
			for(int i=0; i<lines.Length; i++) {
				string[] entries = lines[i].Split(' ');
				float[] values = new float[entries.Length];
				for(int j=0; j<entries.Length; j++) {
					values[j] = Utility.ParseFloat(entries[j]);
				}
				Motions.Add(values);
			}
			return true;
		}
		return false;
	}

	private bool IsPathValid() {
		return File.Exists(GetFilePath());
	}

	private string GetFilePath() {
		return Path;
	}

}
