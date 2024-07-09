#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.Collections;
using System.Collections.Generic;

namespace AI4Animation {
    public class ExportCountAnalyzer : BatchProcessor {

        private MotionEditor Editor = null;

        private int ExportCount = 0;
        private int FrameCount = 0;

        [MenuItem ("AI4Animation/Tools/Export Count Analyzer")]
        static void Init() {
            Window = EditorWindow.GetWindow(typeof(ExportCountAnalyzer));
            Scroll = Vector3.zero;
        }

		public MotionEditor GetEditor() {
			if(Editor == null) {
				Editor = GameObjectExtensions.Find<MotionEditor>(true);
			}
			return Editor;
		}

        public override string GetID(Item item) {
            return Utility.GetAssetName(item.ID);
        }

        public override void DerivedRefresh() {
            
        }

        public override void DerivedInspector() {
			if(GetEditor() == null) {
				EditorGUILayout.LabelField("No editor available in scene.");
				return;
			}

            EditorGUILayout.LabelField("Export Count: " + ExportCount);
            EditorGUILayout.LabelField("Frame Count: " + FrameCount);
            EditorGUILayout.LabelField("Total Time: " + FrameCount/(Editor.TargetFramerate * Editor.TargetFramerate) + "min");

            if(Utility.GUIButton("Refresh", UltiDraw.DarkGrey, UltiDraw.White)) {
                LoadItems(GetEditor().Assets.ToArray());
            }
        }

        public override void DerivedInspector(Item item) {

        }

        public override bool CanProcess() {
            return true;
        }

        public override void DerivedStart() {
            ExportCount = 0;
            FrameCount = 0;
        }

        public override IEnumerator DerivedProcess(Item item) {
            MotionAsset asset = MotionAsset.Retrieve(item.ID);
            {
                ExportCount += asset.Export ? 1 : 0;
                if(asset.Export) {
                    foreach(Frame frame in asset.Frames) {
                        if(asset.InSequences(frame)) {
                            FrameCount += 1;
                        }
                    }
                }
            }
            yield return new WaitForSeconds(0f);
        }

        public override void BatchCallback() {
            Resources.UnloadUnusedAssets();
        }

        public override void DerivedFinish() {

        }

    }
}
#endif