#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.Collections;

namespace AI4Animation {
    public class ClipLengthAnalyzer : BatchProcessor {

        public float Resolution = 0.1f;
        private RunningStatistics Distribution;

		private MotionEditor Editor = null;

        [MenuItem ("AI4Animation/Tools/Clip Length Analyzer")]
        static void Init() {
            Window = EditorWindow.GetWindow(typeof(ClipLengthAnalyzer));
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

            if(Utility.GUIButton("Refresh", UltiDraw.DarkGrey, UltiDraw.White)) {
                LoadItems(GetEditor().Assets.ToArray());
            }

            Resolution = EditorGUILayout.FloatField("Resolution", Resolution);

            if(Distribution != null) {
                float sum = Distribution.Sum();
                EditorGUILayout.HelpBox("Total sec: " + sum.ToString("F3"), MessageType.None);
                EditorGUILayout.HelpBox("Total min: " + (sum/60f).ToString("F3"), MessageType.None);
                EditorGUILayout.HelpBox("Total hrs: " + (sum/3600f).ToString("F3"), MessageType.None);
                EditorGUILayout.HelpBox("Mean sec: " + Distribution.Mean().ToString("F3"), MessageType.None);
            }
        }

        public override void DerivedInspector(Item item) {
        
        }

        public override bool CanProcess() {
            return true;
        }

        public override void DerivedStart() {
            Distribution = new RunningStatistics();
        }

        public override IEnumerator DerivedProcess(Item item) {
            GetEditor().LoadSession(item.ID);
            MotionAsset asset = GetEditor().GetSession().Asset;
            Distribution.Add(asset.GetTotalTime());
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
