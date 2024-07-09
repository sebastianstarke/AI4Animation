using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditorInternal;
#endif
namespace AI4Animation {
    public class SmoothingModule : Module {

        public float[] Windows;

        public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
            return null;
        }

#if UNITY_EDITOR
        protected override void DerivedInitialize() {
            Windows = new float[Asset.Source.Bones.Length];
        }

        protected override void DerivedLoad(MotionEditor editor) {
            
        }

        protected override void DerivedUnload(MotionEditor editor) {
            
        }
        protected override void DerivedCallback(MotionEditor editor) {
            // for(int i=0; i<Asset.Source.Bones.Length; i++) {
            //     if(Windows[i] != 0f) {
            //         editor.GetSession().GetActor().SetBoneTransformation(GetBoneTransformation(editor.GetTimestamp(), editor.Mirror, i), Asset.Source.Bones[i].GetName());
            //     }
            // }
        }

        protected override void DerivedGUI(MotionEditor editor) {
            
        }

        protected override void DerivedDraw(MotionEditor editor) {

        }

        protected override void DerivedInspector(MotionEditor editor) {
            for(int i=0; i<Windows.Length; i++) {
                Windows[i] = EditorGUILayout.Slider(Asset.Source.Bones[i].GetName(), Windows[i], 0f, 1f);
            }
        }		
#endif
        public Matrix4x4 GetBoneTransformation(float timestamp, bool mirrored, int bone) {
            float[] timestamps = Asset.SimulateTimestamps(Windows[bone]);
            Matrix4x4[] transformations = new Matrix4x4[timestamps.Length];
            for(int t=0; t<timestamps.Length; t++) {
                transformations[t] = Asset.GetFrame(timestamps[t]).GetBoneTransformation(bone, mirrored);
            }
            return transformations.Gaussian();
        }
    }
}
