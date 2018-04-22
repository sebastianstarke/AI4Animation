using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using System.IO;

public class Extensions {

	[MenuItem("Assets/Create/Motion Capture Biped")]
	private static void CreateMotionCaptureBiped() {
		CreateMotionCapture("Biped");
	}

	[MenuItem("Assets/Create/Motion Capture Quadruped")]
	private static void CreateMotionCaptureQuadruped() {
		CreateMotionCapture("Quadruped");
	}

	private static void CreateMotionCapture(string name) {
		string source = Application.dataPath + "/Project/MotionCapture/"+name+".unity";
		string destination = AssetDatabase.GetAssetPath(Selection.activeObject) + "/MotionCapture.unity";
		int index = 0;
		while(File.Exists(destination)) {
			index += 1;
			destination = AssetDatabase.GetAssetPath(Selection.activeObject) + "/MotionCapture (" + index +").unity";
		}
		if(!File.Exists(source)) {
			Debug.Log("Source file at path " + source + " does not exist.");
		} else {
			FileUtil.CopyFileOrDirectory(source, destination);
		}
	}

}
