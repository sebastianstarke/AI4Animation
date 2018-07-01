#if UNITY_EDITOR
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditorInternal;

public class EditorExtensions : MonoBehaviour {

	[MenuItem("Assets/Create/Prototype Scene")]
	public static string CreatePrototypeScene() {
		string source = Application.dataPath + "/Resources/PrototypeScene.unity";
		string destination = AssetDatabase.GetAssetPath(Selection.activeObject) + "/Scene.unity";
		int index = 0;
		while(File.Exists(destination)) {
			index += 1;
			destination = AssetDatabase.GetAssetPath(Selection.activeObject) + "/Scene (" + index +").unity";
		}
		if(!File.Exists(source)) {
			Debug.Log("Source file at path " + source + " does not exist.");
		} else {
			FileUtil.CopyFileOrDirectory(source, destination);
		}
		return destination;
	}

}

#endif