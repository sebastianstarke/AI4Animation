#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

namespace AI4Animation {
	public class TailModule : Module {

		public override void DerivedResetPrecomputation() {

		}

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			return null;
		}

		protected override void DerivedInitialize() {

		}

		protected override void DerivedLoad(MotionEditor editor) {
			
		}

		protected override void DerivedUnload(MotionEditor editor) {
            
		}
		

		protected override void DerivedCallback(MotionEditor editor) {
			Actor.Bone bone = editor.GetSession().GetActor().FindBone("Tail");
            Matrix4x4 child = bone.GetChild(0).GetTransformation();
            bone.SetRotation(QuaternionExtensions.LookRotationXY(child.GetPosition()-bone.GetPosition(), Vector3.up));
            bone.GetChild(0).SetTransformation(child);
		}

		protected override void DerivedGUI(MotionEditor editor) {
            
		}

		protected override void DerivedDraw(MotionEditor editor) {

		}

		protected override void DerivedInspector(MotionEditor editor) {

		}

	}
}
#endif
