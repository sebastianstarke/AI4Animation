#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.Collections;
using System.Collections.Generic;

namespace AI4Animation {
    public class JointHeightAnalyzer : BatchProcessor {

        public string[] Bones = new string[0];

        private float[] Heights = new float[0];

        private MotionEditor Editor = null;

        [MenuItem ("AI4Animation/Tools/Joint Height Analyzer")]
        static void Init() {
            Window = EditorWindow.GetWindow(typeof(JointHeightAnalyzer));
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

            for(int i=0; i<Bones.Length; i++) {
                Bones[i] = EditorGUILayout.TextField("Bone " + (i+1), Bones[i]);
            }
            if(Utility.GUIButton("Add Bone", UltiDraw.DarkGrey, UltiDraw.White)) {
                ArrayExtensions.Append(ref Bones, string.Empty);
                Heights = new float[0];
            }
            if(Utility.GUIButton("Remove Bone", UltiDraw.DarkGrey, UltiDraw.White)) {
                ArrayExtensions.Shrink(ref Bones);
                Heights = new float[0];
            }

            for(int i=0; i<Heights.Length; i++) {
                EditorGUILayout.LabelField(Bones[i] + ": " + Heights[i]);
            }
        }

        public override void DerivedInspector(Item item) {
        
        }

        public override bool CanProcess() {
            return true;
        }

        public override void DerivedStart() {
            Heights = new float[Bones.Length];
            Heights.SetAll(float.MaxValue);
        }

        public override IEnumerator DerivedProcess(Item item) {
            MotionAsset asset = MotionAsset.Retrieve(item.ID);
            for(int i=0; i<asset.Frames.Length; i++) {
                for(int j=0; j<Bones.Length; j++) {
                    Heights[j] = Mathf.Min(Heights[j], asset.Frames[i].GetBoneTransformation(Bones[j], false).GetPosition().y);
                    Heights[j] = Mathf.Min(Heights[j], asset.Frames[i].GetBoneTransformation(Bones[j], true).GetPosition().y);
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