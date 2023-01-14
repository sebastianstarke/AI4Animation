#if UNITY_EDITOR
using UnityEditor;

namespace AI4Animation {
	public class MultiAssetModule : Module {
		
		public string[] Assets = new string[0];

		public override void DerivedResetPrecomputation() {

		}

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			return null;
		}

		protected override void DerivedInitialize() {
			
		}

		protected override void DerivedLoad(MotionEditor editor) {
			// for(int i=0; i<Assets.Length; i++) {
			// 	editor.OpenAsset(Assets[i], true);
			// }
		} 

		protected override void DerivedUnload(MotionEditor editor) {

		}

		protected override void DerivedCallback(MotionEditor editor) {
		}

		protected override void DerivedGUI(MotionEditor editor) {
		
		}

		protected override void DerivedDraw(MotionEditor editor) {
			
		}

		protected override void DerivedInspector(MotionEditor editor) {
			if(Assets.Length == 0) {
				EditorGUILayout.HelpBox("No interaction assets specified.", MessageType.None);
				return;
			}
			EditorGUI.BeginDisabledGroup(true);
			for(int i=0; i<Assets.Length; i++) {
				EditorGUILayout.ObjectField(MotionAsset.Retrieve(Assets[i]), typeof(MotionAsset), true);
			}
			EditorGUI.EndDisabledGroup();
		}

		public void AddInteractionGUID(string guid) {
			if(guid != Utility.GetAssetGUID(Asset)) {
				ArrayExtensions.Append(ref Assets, guid);
			}
		}

		public void Clear() {
			Assets = new string[0];
		}

	}
}
#endif