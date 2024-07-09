#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;

namespace AI4Animation {
	public class LatentImporter : EditorWindow {

		public static EditorWindow Window;
		public static Vector2 Scroll;

        public string SequencePath = string.Empty;
        public string FeaturesPath = string.Empty;
        public string Tag = string.Empty;
        public bool RemoveAll = false;

		private MotionEditor Editor = null;
        private EditorCoroutines.EditorCoroutine Coroutine = null;
        private int Count = 0;
        private int BatchSize = 1000;

		[MenuItem ("AI4Animation/Importer/Latent Importer")]
		static void Init() {
			Window = EditorWindow.GetWindow(typeof(LatentImporter));
			Scroll = Vector3.zero;
		}

		public MotionEditor GetEditor() {
			if(Editor == null) {
				Editor = GameObjectExtensions.Find<MotionEditor>(true);
			}
			return Editor;
		}


        public void OnInspectorUpdate() {
            Repaint();
        }

		void OnGUI() {
			Scroll = EditorGUILayout.BeginScrollView(Scroll);

			if(GetEditor() == null) {
				EditorGUILayout.LabelField("No editor available in scene.");
				return;
			}

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

                    EditorGUILayout.ObjectField("Editor", Editor, typeof(MotionEditor), true);

                    SequencePath = EditorGUILayout.TextField("Sequence Path", SequencePath);
                    FeaturesPath = EditorGUILayout.TextField("Features Path", FeaturesPath);
                    Tag = EditorGUILayout.TextField("Tag", Tag);
                    RemoveAll = EditorGUILayout.Toggle("Remove All", RemoveAll);

                    if(Coroutine == null) {
                        if(Utility.GUIButton("Process", UltiDraw.DarkGrey, UltiDraw.White)) {
                            Coroutine = EditorCoroutines.StartCoroutine(Process(), this);
                        }
                    } else {
                        EditorGUILayout.LabelField("Read " + Count + " lines...");
                        if(Utility.GUIButton("Stop", UltiDraw.DarkGrey, UltiDraw.White)) {
                            Coroutine = null;
                        }
                    }
				}
			}

			EditorGUILayout.EndScrollView();
		}

        public IEnumerator Process() {
            Editor.AutoSave = false;

            StreamReader featuresFile = new StreamReader(FeaturesPath);
            StreamReader sequenceFile = new StreamReader(SequencePath);
            yield return new WaitForSeconds(0f);

            LatentModule module = null;

            Count = 0;
            while(Coroutine != null && !sequenceFile.EndOfStream) {
                string sLine = sequenceFile.ReadLine();
                string fLine = featuresFile.ReadLine();

                string[] tags = FileUtility.LineToArray(sLine, ' ');
                float[] features = FileUtility.LineToFloat(fLine, ' ');
                string fileGUID = tags[4];
                bool fileMirrored = tags[2] == "Standard" ? false : true;
                float fileTimestamp = tags[1].ToFloat();

                {
                    if(Editor.Asset != fileGUID || module == null) {
                        Editor.LoadSession(fileGUID);
                        MotionAsset asset = Editor.GetSession().Asset;
                        asset.MarkDirty(true, false);
                        if(RemoveAll) {
                            asset.RemoveAllModules<LatentModule>();
                        } else {
                            if(asset.HasModule<LatentModule>(Tag)) {
                                asset.RemoveModule<LatentModule>(Tag);
                            }
                        }
                        module = asset.AddModule<LatentModule>(Tag);
                    }

                    module.SetValues(fileTimestamp, fileMirrored, features);
                    module.Dimensions = features.Length;
                    module.Min = Mathf.Min(module.Min, features.Min());
                    module.Max = Mathf.Max(module.Max, features.Max());
                }

                Count += 1;
                if(Count % BatchSize == 0) {
                    yield return new WaitForSeconds(0f);
                }
            }

            featuresFile.Close();
            sequenceFile.Close();

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Resources.UnloadUnusedAssets();
            
            Editor.AutoSave = true;

            yield return new WaitForSeconds(0f);
        }
	}
}
#endif