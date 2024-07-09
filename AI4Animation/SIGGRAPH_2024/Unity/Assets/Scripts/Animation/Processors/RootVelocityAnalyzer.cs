#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.Collections;
using System.Collections.Generic;

namespace AI4Animation {
    public class RootVelocityAnalyzer : BatchProcessor {

        public float Resolution = 0.1f;
        private float Slider = 0f;
        private List<int> RootVelocities = new List<int>();

        private MotionEditor Editor = null;

        [MenuItem ("AI4Animation/Tools/Root Velocity Analyzer")]
        static void Init() {
            Window = EditorWindow.GetWindow(typeof(RootVelocityAnalyzer));
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

            if(RootVelocities.Count > 0) {
                Slider = EditorGUILayout.Slider(Slider, 0f, RootVelocities.Count * Resolution);
                EditorGUILayout.IntField("Samples", RootVelocities[Mathf.Clamp(Mathf.RoundToInt(Slider / Resolution), 0, RootVelocities.Count-1)]);
                EditorGUILayout.BeginVertical(GUILayout.Height(50f));
                Rect ctrl = EditorGUILayout.GetControlRect();
                Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
                EditorGUI.DrawRect(rect, UltiDraw.Black);
                UltiDraw.Begin();
                float[] values = new float[RootVelocities.Count];
                for(int i=0; i<values.Length; i++) {
                    values[i] = RootVelocities[i];
                }
                float min = RootVelocities.Min();
                float max = RootVelocities.Max();
                for(int i=0; i<values.Length; i++) {
                    float ratio = i.Ratio(0, values.Length-1);
                    Vector3 p1 = new Vector3(rect.x + ratio * rect.width, rect.y + 1f * rect.height, 0f);
                    Vector3 p2 = new Vector3(rect.x + ratio * rect.width, rect.y + values[i].Normalize(min, max, 1f, 0f) * rect.height, 0f);
                    UltiDraw.DrawLine(p1, p2, UltiDraw.White);
                }
                {
                    float ratio = Slider.Normalize(0f, RootVelocities.Count * Resolution, 0f, 1f);
                    Vector3 p1 = new Vector3(rect.x + ratio * rect.width, rect.y + 1f * rect.height, 0f);
                    Vector3 p2 = new Vector3(rect.x + ratio * rect.width, rect.y + 0f * rect.height, 0f);
                    UltiDraw.DrawLine(p1, p2, UltiDraw.Green);
                }
                UltiDraw.End();
                EditorGUILayout.EndVertical();
            }
        }

        public override void DerivedInspector(Item item) {
        
        }

        public override bool CanProcess() {
            return true;
        }

        public override void DerivedStart() {
            RootVelocities.Clear();
        }

        public override IEnumerator DerivedProcess(Item item) {
            GetEditor().LoadSession(item.ID);
            MotionAsset asset = GetEditor().GetSession().Asset;
            {
                RootModule root = asset.GetModule<RootModule>();
                if(root != null) {
                    for(int i=0; i<asset.Frames.Length; i++) {
                        if(asset.InSequences(asset.Frames[i])) {
                            // float velocity = root.GetRootVelocity(asset.Frames[i].Timestamp, false, asset.GetDeltaTime()).magnitude;
                            // int pivot = Mathf.RoundToInt(velocity / Resolution);
                            // while(pivot >= RootVelocities.Count) {
                            //     RootVelocities.Add(0);
                            // }
                            // RootVelocities[pivot] += 1;
                        }
                    }
                } else {
                    Debug.Log("No root found for asset: " + asset.name);
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