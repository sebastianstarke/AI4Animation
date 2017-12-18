using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;

public class BVHRecorder : EditorWindow {

	public static EditorWindow Window;

	public BioAnimation Animation;

	private bool Recording = false;

	[MenuItem ("Addons/BVH Recorder")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(BVHRecorder));
	}

	void LateUpdate() {
		Debug.Log("LATE UPDATE");
	}

	private IEnumerator Record() {
		while(Recording) {
			yield return new WaitForEndOfFrame();
			Debug.Log("RECORD");
		}
	}

	void OnGUI() {
		Utility.SetGUIColor(Utility.Black);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(Utility.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(Utility.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Recorder");
				}

				if(!Application.isPlaying) {
					EditorGUILayout.LabelField("Change into play mode to start recording.");
					return;
				}

				Animation = (BioAnimation)EditorGUILayout.ObjectField("Animation", Animation, typeof(BioAnimation), true);

				if(Utility.GUIButton(Recording ? "Stop" : "Start", Recording ? Utility.DarkRed : Utility.DarkGreen, Utility.White)) {
					Recording = !Recording;
					if(Recording) {
						Animation.StartCoroutine(Record());
					}
				}

			}
		}
	}

}
