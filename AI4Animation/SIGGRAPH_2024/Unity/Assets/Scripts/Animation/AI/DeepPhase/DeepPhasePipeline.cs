#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.Collections;
using AI4Animation;

namespace DeepPhase {
    public class DeepPhasePipeline : AssetPipelineSetup {

        public float Framerate = 60f;
        public bool WriteMirror = true;
        public bool[] Mask = new bool[0];

        private DateTime Timestamp;
        private float Progress = 0f;
        private float SamplesPerSecond = 0f;
        private int Samples = 0;
        private int Sequence = 0;

        private AssetPipeline.Data.File S;
        private AssetPipeline.Data X;

        public override void Inspector() {
            EditorGUI.BeginDisabledGroup(true);
            EditorGUILayout.TextField("Export Path", AssetPipeline.Data.GetExportPath());
            EditorGUI.EndDisabledGroup();
            Framerate = EditorGUILayout.FloatField("Framerate", Framerate);
            WriteMirror = EditorGUILayout.Toggle("Write Mirror", WriteMirror);

            Utility.SetGUIColor(UltiDraw.White);
            using(new EditorGUILayout.VerticalScope ("Box")) {
                Utility.ResetGUIColor();
                Actor actor = Pipeline.GetEditor().GetSession().GetActor();
                if(Mask.Length == 0 || Mask.Length != actor.Bones.Length) {
                    Mask = new bool[actor.Bones.Length];
                    Mask.SetAll(true);
                }
                for(int i=0; i<Mask.Length; i++) {
                    Mask[i] = EditorGUILayout.Toggle(actor.Bones[i].GetName(), Mask[i]);
                }
            }

            if(Pipeline.IsProcessing() || Pipeline.IsAborting()) {
                EditorGUI.BeginDisabledGroup(true);
                EditorGUILayout.FloatField("Samples Per Second", SamplesPerSecond);
                EditorGUI.EndDisabledGroup();
                EditorGUI.DrawRect(
                    new Rect(
                        EditorGUILayout.GetControlRect().x,
                        EditorGUILayout.GetControlRect().y,
                        Progress * EditorGUILayout.GetControlRect().width, 25f
                    ),
                    UltiDraw.Green.Opacity(0.75f)
                );
            }
        }

        public override void Inspector(AssetPipeline.Item item) {

        }

        public override bool CanProcess() {
            return true;
        }

        public override void Begin() {
            Samples = 0;
            Sequence = 0;
            S = AssetPipeline.Data.CreateFile("Sequences", AssetPipeline.Data.TYPE.Text);
            X = new AssetPipeline.Data("Data", true, true, true);
        }

        private void WriteSequenceInfo(int sequence, int frame, bool mirrored, MotionAsset asset) {
            //Sequence - Frame - Mirroring - Name - GUID
            S.WriteLine(
                sequence.ToString() + AssetPipeline.Data.Separator +
                frame.ToString() + AssetPipeline.Data.Separator +
                (mirrored ? "Mirrored" : "Standard") + AssetPipeline.Data.Separator +
                asset.name + AssetPipeline.Data.Separator +
                Utility.GetAssetGUID(asset));
        }

        public override IEnumerator Iterate(MotionAsset asset) {
            Pipeline.GetEditor().LoadSession(Utility.GetAssetGUID(asset));
            if(asset.Export) {
                TimeSeries timeSeries = Pipeline.GetEditor().GetTimeSeries();
                Actor actor = Pipeline.GetEditor().GetSession().GetActor();
                foreach(bool value in new bool[]{false,true}) {
                    Pipeline.GetEditor().SetMirror(value);
                    if(value && !WriteMirror) {
                        continue;
                    }
                    Sequence += 1;
                    int step = Mathf.RoundToInt(Pipeline.GetEditor().GetSession().Asset.Framerate / Framerate);
                    for(int f=0; f<asset.Frames.Length; f+=step) {
                        Frame frame = asset.Frames[f];
                        float timestamp = frame.Timestamp;
                        bool mirrored = Pipeline.GetEditor().Mirror;
                        PAESetup.Export(this, X, asset, timestamp, mirrored, timeSeries, actor);
                        X.Store();
                        WriteSequenceInfo(Sequence, f, mirrored, asset);
                        Samples += 1;
                        if(Utility.GetElapsedTime(Timestamp) >= 0.1f) {
                            Progress = frame.Timestamp / asset.GetTotalTime();
                            SamplesPerSecond = Samples / (float)Utility.GetElapsedTime(Timestamp);
                            Samples = 0;
                            Timestamp = Utility.GetTimestamp();
                            yield return new WaitForSeconds(0f);
                        }
                    }
                }
            }
        }

        public override void Callback() {
            Resources.UnloadUnusedAssets();
        }

        public override void Finish() {
            S.Close();
            X.Finish();
        }

        private class PAESetup {
            public static void Export(DeepPhasePipeline setup, AssetPipeline.Data X, MotionAsset asset, float timestamp, bool mirrored, TimeSeries timeSeries, Actor actor) {
                if(DeepPhaseModule.Curves == null || DeepPhaseModule.Curves.Asset != asset) {
                    Debug.Log("Computing curves in asset: " + asset.name);
                    DeepPhaseModule.ComputeCurves(
                        asset,
                        actor,
                        timeSeries
                    );
                }
                X.Feed(DeepPhaseModule.Curves.Collect(timestamp, mirrored, setup.Mask), "MotionFeature-");
            }
        }

    }
}
#endif
