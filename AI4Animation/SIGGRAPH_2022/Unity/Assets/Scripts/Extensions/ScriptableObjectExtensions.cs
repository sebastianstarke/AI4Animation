using UnityEngine;
using UnityEditor;

public static class ScriptableObjectExtensions {
    public static T Create<T>(string directory, string name, bool assetPath=false) {
		string destination = (directory == string.Empty ? string.Empty : (directory + "/")) + name + ".asset";
		if(!assetPath) {
			destination = "Assets" + "/" + destination;
		}
		destination = AssetDatabase.GenerateUniqueAssetPath(destination);
		ScriptableObject asset = ScriptableObject.CreateInstance(typeof(T).ToString());
		AssetDatabase.CreateAsset(asset, destination);
		return asset.ToType<T>();
	}

    public static void Create(string type, string directory, string name, bool assetPath=false) {
		string destination = (directory == string.Empty ? string.Empty : (directory + "/")) + name + ".asset";
		if(!assetPath) {
			destination = "Assets" + "/" + destination;
		}
		destination = AssetDatabase.GenerateUniqueAssetPath(destination);
		ScriptableObject asset = ScriptableObject.CreateInstance(type);
		AssetDatabase.CreateAsset(asset, destination);
	}

    public static T Create<T>(ScriptableObject parent) {
		ScriptableObject asset = ScriptableObject.CreateInstance(typeof(T).ToString());
		AssetDatabase.AddObjectToAsset(asset, parent);
		return asset.ToType<T>();
	}

    public static void Create(string type, ScriptableObject parent) {
		ScriptableObject asset = ScriptableObject.CreateInstance(type);
		AssetDatabase.AddObjectToAsset(asset, parent);
	}

    public static void Save(this ScriptableObject asset) {
		EditorUtility.SetDirty(asset);
		AssetDatabase.SaveAssets();
		AssetDatabase.Refresh();
    }
}
