using UnityEngine;
using UnityEditor;
using System.IO;

public class Extensions {

	[MenuItem("Assets/Create/Motion Capture")]
	private static void CreateMotionCapture() {
		string source = Application.dataPath + "/Project/MotionCapture/Setup.unity";
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
