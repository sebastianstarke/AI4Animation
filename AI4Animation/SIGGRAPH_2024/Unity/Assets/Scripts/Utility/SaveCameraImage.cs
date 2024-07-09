using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class SaveCameraImage : MonoBehaviour
{

    private Camera Camera;

    public RenderTexture Texture;

    public string Path = string.Empty;

    public int ImageNumber = 1;

    void Start() {

    }

    void Update() {

    }

    public void Save() {
        Texture.width = Screen.width;
        Texture.height = Screen.height;
        byte[] bytes = toTexture2D(GetCamera().activeTexture).EncodeToPNG();
        System.IO.File.WriteAllBytes(Path + "/" + ImageNumber + ".png", bytes);
        ImageNumber += 1;
    }

    public Texture2D toTexture2D(RenderTexture rTex) {
        Texture2D tex = new Texture2D(1920, 1080, TextureFormat.RGB24, false);
        RenderTexture.active = rTex;
        tex.ReadPixels(new Rect(0, 0, rTex.width, rTex.height), 0, 0);
        tex.Apply();
        return tex;
    }

    public Camera GetCamera() {
        if(Camera == null) {
            Camera = GetComponent<Camera>();
        }
        return Camera;
    }

	#if UNITY_EDITOR
	[CustomEditor(typeof(SaveCameraImage), true)]
	public class SaveCameraImage_Editor : Editor {

		public SaveCameraImage Target;

		void Awake() {
			Target = (SaveCameraImage)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

            DrawDefaultInspector();

            if(GUILayout.Button("Save Image")) {
                Target.Save();                
            }

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}
        
	}
	#endif
}
