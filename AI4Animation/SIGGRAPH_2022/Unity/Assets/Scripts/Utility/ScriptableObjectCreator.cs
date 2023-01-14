using UnityEngine;
using UnityEditor;

#if UNITY_EDITOR
public class ScriptableObjectCreator : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;

    public string Type = string.Empty;

	[MenuItem ("AI4Animation/Tools/Scriptable Object Creator")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(ScriptableObjectCreator));
		Scroll = Vector3.zero;
	}

	public void OnInspectorUpdate() {
		Repaint();
	}

	void OnGUI() {
		Scroll = EditorGUILayout.BeginScrollView(Scroll);

		Utility.SetGUIColor(UltiDraw.Black);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Mustard);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField(this.GetType().ToString());
				}

				Utility.SetGUIColor(UltiDraw.LightGrey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Type = EditorGUILayout.TextField("Type", Type);

                    if(Utility.GUIButton("Create", UltiDraw.DarkGrey, UltiDraw.White)) {
						ScriptableObjectExtensions.Create(Type, string.Empty, Type);
                    }
				}

			}
		}

		EditorGUILayout.EndScrollView();
	}

}
#endif