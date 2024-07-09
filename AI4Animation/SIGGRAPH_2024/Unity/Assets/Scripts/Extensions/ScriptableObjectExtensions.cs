#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.IO;

public static class ScriptableObjectExtensions {

	[MenuItem("Assets/Create Scriptable Object")]
	private static void CreateInstance() {
		MonoScript script = AssetDatabase.LoadAssetAtPath(EditorExtensions.GetSelectedAsset(), typeof(MonoScript)) as MonoScript;
		Create(script.GetClass().ToString());
	}

	public static void Create(string type) {
		Create(type, EditorExtensions.GetSelectedDirectory(), EditorExtensions.GetSelectedAssetName(), true);
	}

	public static ScriptableObject Create(string type, string name) {
		return Create(type, EditorExtensions.GetSelectedDirectory(), name, true);
	}

    public static T Create<T>(T instance, string name) where T : UnityEngine.Object {
        if(instance != null) {
            return Create<T>(Path.GetDirectoryName(AssetDatabase.GetAssetPath(instance)), name, true);
        } else {
            return Create<T>(EditorExtensions.GetSelectedDirectory(), name, true);
        }
    }

    public static T Create<T>(string directory, string name, bool assetPath=false) {
		string destination = (directory == string.Empty ? string.Empty : (directory + "/")) + (name == string.Empty ? "ScriptableObject" : name) + ".asset";
		if(!assetPath) {
			destination = "Assets" + "/" + destination;
		}
		destination = AssetDatabase.GenerateUniqueAssetPath(destination);
		ScriptableObject asset = ScriptableObject.CreateInstance(typeof(T).ToString());
		AssetDatabase.CreateAsset(asset, destination);
		return asset.ToType<T>();
	}

    public static ScriptableObject Create(string type, string directory, string name, bool assetPath=false) {
		string destination = (directory == string.Empty ? string.Empty : (directory + "/")) + (name == string.Empty ? "ScriptableObject" : name) + ".asset";
		if(!assetPath) {
			destination = "Assets" + "/" + destination;
		}
		destination = AssetDatabase.GenerateUniqueAssetPath(destination);
		ScriptableObject asset = ScriptableObject.CreateInstance(type);
		AssetDatabase.CreateAsset(asset, destination);
		return asset;
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

	public static void MarkDirty(this ScriptableObject asset) {
		EditorUtility.SetDirty(asset);
	}

    public static void Save(this ScriptableObject asset) {
		EditorUtility.SetDirty(asset);
		AssetDatabase.SaveAssets();
		AssetDatabase.Refresh();
    }

	public static void Save() {
		AssetDatabase.SaveAssets();
		AssetDatabase.Refresh();
	}
}
#endif