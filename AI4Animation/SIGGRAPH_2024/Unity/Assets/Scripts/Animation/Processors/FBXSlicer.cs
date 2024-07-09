#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;

namespace AI4Animation {
	public class FBXSlicer : BatchProcessor {

		public string Source = string.Empty;
		public string Destination = string.Empty;

        public float SlicingWindow = 2.0f;
		private List<string> Imported;
		private List<string> Skipped;

		[MenuItem ("AI4Animation/Importer/FBX Slicer")]
		static void Init() {
			Window = EditorWindow.GetWindow(typeof(FBXSlicer));
			Scroll = Vector3.zero;
		}

		public override string GetID(Item item) {
			return item.ID;
		}

		public override void DerivedRefresh() {
			
		}

		public override void DerivedInspector() {
			EditorGUILayout.LabelField("Source");
			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.LabelField("<Path>", GUILayout.Width(50));
			Source = EditorGUILayout.TextField(Source);
			GUI.skin.button.alignment = TextAnchor.MiddleCenter;
			if(GUILayout.Button("O", GUILayout.Width(20))) {
				Source = EditorUtility.OpenFolderPanel("FBX Importer", Source == string.Empty ? Application.dataPath : Source, "");
				GUIUtility.ExitGUI();
			}
			EditorGUILayout.EndHorizontal();

			EditorGUILayout.LabelField("Destination");
			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.LabelField("<Path>", GUILayout.Width(50));
			Destination = EditorGUILayout.TextField(Destination);
			GUI.skin.button.alignment = TextAnchor.MiddleCenter;
			if(GUILayout.Button("O", GUILayout.Width(20))) {
				Destination = EditorUtility.OpenFolderPanel("FBX Slice Exporter", Destination == string.Empty ? Application.dataPath : Destination, "");
				GUIUtility.ExitGUI();
			}
			EditorGUILayout.EndHorizontal();
            SlicingWindow = EditorGUILayout.FloatField("Slicing Window", SlicingWindow);

			if(Utility.GUIButton("Load Source Directory", UltiDraw.DarkGrey, UltiDraw.White)) {
				LoadDirectory(Source);
			}
		}

		public override void DerivedInspector(Item item) {
		
		}

		private void SetSource(string source) {
			if(Source != source) {
				Source = source;
				LoadDirectory(null);
			}
		}

		private void LoadDirectory(string directory) {
			if(directory == null) {
				LoadItems(new string[0]);
			} else {
				if(Directory.Exists(directory)) {
					List<string> paths = new List<string>();
					Iterate(directory);
					LoadItems(paths.ToArray());
					void Iterate(string folder) {
						DirectoryInfo info = new DirectoryInfo(folder);
						foreach(FileInfo i in info.GetFiles()) {
							if(i.Name.Contains(".fbx")) {
								paths.Add(i.FullName);
							}
						}
						foreach(DirectoryInfo i in info.GetDirectories()) {
							Iterate(i.FullName);
						}
					}
				} else {
					LoadItems(new string[0]);
				}
			}
		}

        public override bool CanProcess() {
            return true;
        }

		public override void DerivedStart() {
			Imported = new List<string>();
			Skipped = new List<string>();
		}

		public override IEnumerator DerivedProcess(Item item) {
			string source = Source;
			string destination = Destination;
			string target = (destination + item.ID.Remove(0, source.Length)).Replace(".fbx", "");
			if(!Directory.Exists(target)) {
				Directory.CreateDirectory(target);
			}
            
            string tmpFileName = "tmp.fbx";
            File.Copy(item.ID,  Application.dataPath + "/" + tmpFileName);
            AssetDatabase.ImportAsset("Assets/" + tmpFileName);
			AnimationClip sourceClip = (AnimationClip)AssetDatabase.LoadAssetAtPath("Assets/" + tmpFileName, typeof(AnimationClip));
            GameObject go = (GameObject)AssetDatabase.LoadAssetAtPath("Assets/" + tmpFileName, typeof(GameObject));
            //Create Model
            GameObject instance = Instantiate(go) as GameObject;
            List<Transform> transforms = new List<Transform>(instance.GetComponentsInChildren<Transform>());
            transforms.RemoveAt(0);

            float totalFrames = Mathf.RoundToInt(sourceClip.frameRate * sourceClip.length);
            // EditorCurveBinding[] bindings = AnimationUtility.GetCurveBindings(sourceClip);
            // AnimationCurve[] curves = new AnimationCurve[bindings.Length];
            // for (int i = 0; i < bindings.Length; i++)
            // {
            //     sourceClip.
            //     curves[i] = AnimationUtility.GetEditorCurve(sourceClip, bindings[i]);
            //     curves[i].keys[0].
            // }
            // AnimationClip processedClip = new AnimationClip();
            // AnimationUtility.SetEditorCurves(processedClip, bindings, curves);
            AssetDatabase.DeleteAsset("Assets/" + tmpFileName);
			yield return new WaitForSeconds(0f);
		}

		public override void BatchCallback() {
			AssetDatabase.SaveAssets();
			Resources.UnloadUnusedAssets();
		}

		public override void DerivedFinish() {
			if(Imported.Count > 0) {
				AssetDatabase.Refresh();
			}

			Debug.Log("Imported " + Imported.Count + " assets.");
			Imported.ToArray().Print();

			Debug.Log("Skipped " + Skipped.Count + " assets.");
			Skipped.ToArray().Print();
		}

	}
}
#endif