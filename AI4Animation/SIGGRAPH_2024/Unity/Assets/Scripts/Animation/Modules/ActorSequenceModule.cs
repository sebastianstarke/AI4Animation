using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
    public class ActorSequenceModule : Module {

        public int SequenceSamples = 30;
        public float SequenceWindow = 5f;

        private GameObject Container;
        private Actor[] ActorReferences;
        private int[] BoneIndices;

        public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
            return null;
        }
#if UNITY_EDITOR
        protected override void DerivedInitialize() {
        
        }

        protected override void DerivedLoad(MotionEditor editor) {
            // LoadReferences(editor);
        }

        protected override void DerivedUnload(MotionEditor editor) {
            UnloadReferences();
        }
        
        protected override void DerivedCallback(MotionEditor editor) {
            // if(SequenceLength != ActorReferences.Length) {
            // 	LoadReferences(editor);
            // }
            // AssignReferences(editor, editor.GetTimestamp(), editor.Mirror, SequenceSamples, SequenceWindow);
        }

        protected override void DerivedGUI(MotionEditor editor) {

        }

        protected override void DerivedDraw(MotionEditor editor) {

        }

        protected override void DerivedInspector(MotionEditor editor) {
            SequenceSamples = EditorGUILayout.IntField("Sequence Samples", SequenceSamples);
            SequenceWindow = EditorGUILayout.FloatField("Sequence Window", SequenceWindow);
        }
        public void LoadReferences(MotionEditor editor, int samples) {
            UnloadReferences();
            Container = GameObject.Find("ActorSequences");
            if(Container == null) {
                Container = new GameObject("ActorSequences");
            }
            ActorReferences = new Actor[samples];
            for(int i=0; i<ActorReferences.Length; i++) {
                GameObject instance = Instantiate(editor.GetSession().GetActor().gameObject);
                instance.transform.SetParent(Container.transform);
                ActorReferences[i] = instance.GetComponent<Actor>();
            }
            BoneIndices = Asset.Source.GetBoneIndices(editor.GetSession().GetActor().GetBoneNames());
        }

        public void AssignReferences(MotionEditor editor, float timestamp, bool mirrored, int samples, float window) {
            if(Container == null || ActorReferences.Length != samples) {
                LoadReferences(editor, samples);
            }
            float delta = window/samples;
            for(int i=0; i<ActorReferences.Length; i++) {
                float t = timestamp + (i+1)*delta;
                Frame frame = Asset.GetFrame(t);
                for(int j=0; j<BoneIndices.Length; j++) {
                    ActorReferences[i].Bones[j].SetTransformation(frame.GetBoneTransformation(BoneIndices[j], mirrored));
                }
            }
        }
#endif

        private void UnloadReferences(){
            if(Container != null) DestroyImmediate(Container);
            ActorReferences = null;
        }
    }
}
