using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
    public class MotionSearchModule : Module {
        public string RootTag = "Hips";

        [NonSerialized] public int States = 6;
        [NonSerialized] public float Anticipation = 0.5f;
        [NonSerialized] public Vector3 Deviation = Vector3.zero;
        [NonSerialized] public int SequenceLength = 15;
        // [NonSerialized] public bool AlignSequences = true;
        [NonSerialized] [Range(0f,1f)] public float Opacity = 0.5f;

        [NonSerialized] public bool Randomize = false;
        [NonSerialized] public float SamplingRadius = 1f;
        [NonSerialized] public float SamplingAngle = 180f;

        public Actor Skeleton;
        public MotionSearchDatabase Database;
        
#if UNITY_EDITOR
        private EditorCoroutines.EditorCoroutine Coroutine = null;
#endif
        // private float Timestamp = 0f;

        public class State {
            public Matrix4x4 Root;
            public Matrix4x4[] Transformations;
            public Vector3[] Velocities;
            public State(Matrix4x4 root, Matrix4x4[] transformations, Vector3[] velocities) {
                Root = root;
                Transformations = transformations;
                Velocities = velocities;
            }
            public void Draw(Actor actor, Color color, float opacity, string[] bones) {
                // UltiDraw.Begin();
                // Draw Root
                // UltiDraw.End();
                actor.Draw(Transformations, bones, color.Opacity(opacity), UltiDraw.Black.Opacity(opacity), Actor.DRAW.Sketch);
            }
        }

        public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
            return null;
        }
#if UNITY_EDITOR
        protected override void DerivedInitialize() {

        }

        protected override void DerivedLoad(MotionEditor editor) {

        }

        protected override void DerivedUnload(MotionEditor editor) {

        }

        protected override void DerivedCallback(MotionEditor editor) {

        }

        protected override void DerivedGUI(MotionEditor editor) {

        }

        protected override void DerivedDraw(MotionEditor editor) {
            // float length = Anticipation - Divergence;
            // float scale = length / Anticipation;
            // Timestamp = Timestamp + scale/editor.TargetFramerate;
            // if(float.IsNaN(Timestamp) || Timestamp >= length) {
            //     Timestamp = 0f;
            // }

            Vector3 deviation = Randomize ? new Vector3(
                UnityEngine.Random.Range(-SamplingRadius, SamplingRadius),
                UnityEngine.Random.Range(-SamplingAngle, SamplingAngle),
                UnityEngine.Random.Range(-SamplingRadius, SamplingRadius)
            ) : Deviation;

            State[] states = SampleStates(editor.GetTimestamp(), editor.Mirror, Anticipation, deviation, States, Asset, GetSkeleton(editor).GetBoneNames());
            for(int i=0; i<states.Length; i++) {
                GetSkeleton(editor).Draw(states[i].Transformations, UltiDraw.Red.Opacity(Opacity), UltiDraw.Red.Opacity(Opacity), Actor.DRAW.Sketch);
            }
            // GetSkeleton(editor).Draw(
            //     Asset.GetFrame(editor.GetTimestamp() + Divergence + Timestamp).GetBoneTransformations(GetSkeleton(editor).GetBoneNames(), editor.Mirror),
            //     UltiDraw.Red,
            //     UltiDraw.Red,
            //     Actor.DRAW.Skeleton
            // );

            Value[] search = Search(editor.GetTimestamp(), editor.Mirror, Anticipation, deviation, SequenceLength);
            for(int i=0; i<search.Length; i++) {
                GetSkeleton(editor).Draw(search[i].GetTransformations(GetSkeleton(editor)).TransformationsFromTo(search.First().GetRoot(), states.First().Root, false), UltiDraw.Green.Opacity(Opacity), UltiDraw.Green.Opacity(Opacity), Actor.DRAW.Sketch);
            }
            // if(search.Length > 0) {
            //     int index = Mathf.RoundToInt(Timestamp.Ratio(0f, length) * (search.Length-1));
            //     GetSkeleton(editor).Draw(
            //         search[index].Asset.GetFrame(search[index].Timestamp).GetBoneTransformations(GetSkeleton(editor).GetBoneNames(), search[index].Mirrored).TransformationsFromTo(search.First().GetRoot(), (AlignSequences ? states : search).First().GetRoot(), false),
            //         UltiDraw.Green,
            //         UltiDraw.Green,
            //         Actor.DRAW.Skeleton
            //     );
            // }

            UltiDraw.Begin();
            UltiDraw.DrawWireCircle(editor.GetSession().GetActor().transform.position, Quaternion.Euler(90f, 0f, 0f), SamplingRadius*2f, UltiDraw.Red.Opacity(Opacity));
            UltiDraw.End();
        }

        protected override void DerivedInspector(MotionEditor editor) {
            RootTag = EditorGUILayout.TextField("Root Tag", RootTag);

            States = Mathf.Max(EditorGUILayout.IntField("States", States), 0);
            Anticipation = Mathf.Max(EditorGUILayout.FloatField("Anticipation", Anticipation));
            // Divergence = Mathf.Max(EditorGUILayout.FloatField("Divergence", Divergence));
            Deviation = EditorGUILayout.Vector3Field("Deviation", Deviation);
            SequenceLength = Mathf.Max(EditorGUILayout.IntField("Sequence Length", SequenceLength), 0);
            // AlignSequences = EditorGUILayout.Toggle("Align Sequences", AlignSequences);
            Opacity = EditorGUILayout.Slider("Opacity", Opacity, 0f, 1f);

            Randomize = EditorGUILayout.Toggle("Randomize", Randomize);
            SamplingRadius = EditorGUILayout.FloatField("Sampling Radius", SamplingRadius);
            SamplingAngle = EditorGUILayout.FloatField("Sampling Angle", SamplingAngle);

            Skeleton = EditorGUILayout.ObjectField("Skeleton", Skeleton, typeof(Actor), true) as Actor;
            Database = EditorGUILayout.ObjectField("Database", Database, typeof(MotionSearchDatabase), true) as MotionSearchDatabase;
            
            if(Coroutine == null) {
                if(Utility.GUIButton("Generate Database", UltiDraw.DarkGrey, UltiDraw.White)) {
                    Coroutine = EditorCoroutines.StartCoroutine(BuildDatabase(editor), this);
                }
            } else {
                if(Utility.GUIButton("Stop", UltiDraw.DarkGrey, UltiDraw.White)) {
                    EditorCoroutines.StopCoroutine(BuildDatabase(editor), this);
                    Coroutine = null;
                }
            }

            if(Database != null) {
                EditorGUILayout.LabelField("Samples: " + Database.Samples.Count);
                EditorGUILayout.LabelField("Bones: " + Database.Bones.Length);
                EditorGUILayout.LabelField("Labels: " + Database.Labels.Length);
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField("Initialized: " + Database.IsInitialized());
                if(!Database.IsInitialized()) {
                    EditorGUI.BeginDisabledGroup(Coroutine != null);
                    if(Utility.GUIButton("Initialize", UltiDraw.DarkGrey, UltiDraw.White)) {
                        Initialize();
                    }
                    EditorGUI.EndDisabledGroup();
                }
                EditorGUILayout.EndHorizontal();
            }
        }

        public Actor GetSkeleton(MotionEditor editor) {
            return Skeleton == null ? editor.GetSession().GetActor() : Skeleton;
        }
#endif
        public void Initialize() {
            if(Database == null) {
                Debug.Log("No database available.");
                return;
            }
            Database.Build(normalize:true, groups:Database.Labels, bucketCapacity:24, discretization:null);
        }

        public Value[] Search(float timestamp, bool mirrored, float anticipation, Vector3 deviation, int sequenceLength) {
            if(Database == null) {
                Debug.Log("Searching sequence failed since no database is available.");
                return new Value[0];
            }
            if(!Database.IsInitialized()) {
                Debug.Log("Searching sequence failed since database is not initialized.");
                return new Value[0];
            }

            State[] states = SampleStates(timestamp, mirrored, anticipation, deviation, States, Asset, Database.Bones);
            Value match = Database.Query(Value.GenerateKey(states), 1, 0, null).First();
            
            Value[] values = new Value[sequenceLength+1];
            for(int i=0; i<values.Length; i++) {
                values[i] = new Value(null, match.Asset, match.Timestamp, match.Mirrored);
                match = match.Next != null ? match.Next : match;
            }
            return values;
        }

        public State[] SampleStates(float timestamp, bool mirrored, float anticipation, Vector3 deviation, int states, MotionAsset asset, string[] bones) {
            State[] values = new State[states];
            int[] indices = asset.Source.GetBoneIndices(bones);
            for(int i=0; i<values.Length; i++) {
                float ratio = i.Ratio(0, values.Length-1);
                float t = ratio.Normalize(0f, 1f, timestamp, timestamp + anticipation);
                Matrix4x4 delta = Matrix4x4.TRS(
                    new Vector3(deviation.x, 0f, deviation.z),
                    Quaternion.AngleAxis(deviation.y, Vector3.up),
                    Vector3.one
                );
                delta = Utility.Interpolate(delta, Matrix4x4.identity, ratio);
                Matrix4x4 source = asset.GetModule<RootModule>("Hips").GetRootTransformation(t, mirrored);
                Matrix4x4 target = source * delta;
                Matrix4x4[] transformations = asset.GetFrame(t).GetBoneTransformations(indices, mirrored);
                Vector3[] velocities = asset.GetFrame(t).GetBoneVelocities(indices, mirrored);
                values[i] = new State(
                    target,
                    transformations.TransformationsFromTo(source, target, false),
                    velocities.DirectionsFromTo(source, target, false)
                );
            }
            return values;
        }

#if UNITY_EDITOR
        public IEnumerator BuildDatabase(MotionEditor editor) {
            float step = 0.1f;
            float framerate = editor.TargetFramerate;
            DateTime timer = Utility.GetTimestamp();

            string[] bones = GetSkeleton(editor).GetBoneNames();
            int[] indices = Asset.Source.GetBoneIndices(bones);
            Debug.Log("Building Database");
            Debug.Log("Bones: " + bones.Format());
            Debug.Log("Indices: " + indices.Format());

            MotionSearchDatabase database = ScriptableObjectExtensions.Create<MotionSearchDatabase>(Database, "MotionSearch" + "_" + Mathf.RoundToInt(framerate) + "Hz" + "_" + Tag);
            database.Bones = bones;
            database.Labels = Value.GenerateLabels(States, bones.Length);
            for(int i=0; i<editor.Assets.Count; i++) {
                MotionAsset asset = MotionAsset.Retrieve(editor.Assets[i]);
                Debug.Log("Processing " + (i+1) + " / " + editor.Assets.Count + ": " + asset.name);
                
                MotionSearchModule searchModule = asset.AddOrGetModule<MotionSearchModule>(Tag);
                searchModule.Database = database;

                //Standard
                {
                    int index = 0;
                    Value previous = null;
                    while(index / framerate < asset.GetTotalTime()) {
                        float timestamp = index / framerate;
                        bool mirrored = false;
                        Value value = new Value(previous, asset, timestamp, mirrored);
                        float[] key = Value.GenerateKey(SampleStates(timestamp, mirrored, Anticipation, Vector3.zero, States, asset, bones));
                        previous = value;
                        database.AddSample(key, value);
                        index += 1;
                        if(Utility.GetElapsedTime(timer) > step) {
                            timer = Utility.GetTimestamp();
                            yield return new WaitForSeconds(0f);
                        }
                    }
                }

                //Mirrored
                {
                    int index = 0;
                    Value previous = null;
                    while(index / framerate < asset.GetTotalTime()) {
                        float timestamp = index / framerate;
                        bool mirrored = true;
                        Value value = new Value(previous, asset, timestamp, mirrored);
                        float[] key = Value.GenerateKey(SampleStates(timestamp, mirrored, Anticipation, Vector3.zero, States, asset, bones));
                        previous = value;
                        database.AddSample(key, value);
                        index += 1;
                        if(Utility.GetElapsedTime(timer) > step) {
                            timer = Utility.GetTimestamp();
                            yield return new WaitForSeconds(0f);
                        }
                    }
                }

                searchModule.MarkDirty();
            }
            database.MarkDirty();
            ScriptableObjectExtensions.Save();
            Coroutine = null;
        }
#endif
    }
}
