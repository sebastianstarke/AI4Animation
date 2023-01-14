#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.Collections;
using AI4Animation;

namespace DeepPhase {
    public class DeepPhasePipeline : AssetPipelineSetup {

        public SocketNetwork Network;

        public int Channels = 5;
        public bool WriteMirror = true;

        public string TagSuffix = string.Empty;
        public string PhasePath = string.Empty;
        public string SequencesPath = string.Empty;

        private DateTime Timestamp;
        private float Progress = 0f;
        private float SamplesPerSecond = 0f;
        private int Samples = 0;
        private int Sequence = 0;

        private AssetPipeline.Data.File S;
        private AssetPipeline.Data X, Y;

        public override void Inspector() {
            EditorGUI.BeginDisabledGroup(true);
            EditorGUILayout.TextField("Export Path", AssetPipeline.Data.GetExportPath());
            EditorGUI.EndDisabledGroup();
            WriteMirror = EditorGUILayout.Toggle("Write Mirror", WriteMirror);
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
            X = new AssetPipeline.Data("Data", false, false, true);
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
                for(int i=1; i<=2; i++) {
                    if(i==1) {
                        Pipeline.GetEditor().SetMirror(false);
                    } else if(i==2 && WriteMirror) {
                        Pipeline.GetEditor().SetMirror(true);
                    } else {
                        break;
                    }
                    foreach(Interval seq in asset.Sequences) {
                        Sequence += 1;
                        for(int frame=seq.Start; frame<=seq.End; frame++) {
                            float timestamp = asset.GetFrame(frame).Timestamp;
                            bool mirrored = Pipeline.GetEditor().Mirror;
                            TrainingSetup.Export(this, X, asset, timestamp, mirrored, timeSeries, actor);
                            X.Store();
                            WriteSequenceInfo(Sequence, frame, mirrored, asset);
                            Samples += 1;
                            if(Utility.GetElapsedTime(Timestamp) >= 0.1f) {
                                Progress = frame.Ratio(seq.Start, seq.End-1);
                                SamplesPerSecond = Samples / (float)Utility.GetElapsedTime(Timestamp);
                                Samples = 0;
                                Timestamp = Utility.GetTimestamp();
                                yield return new WaitForSeconds(0f);
                            }
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

        private class TrainingSetup {
            public static void Export(DeepPhasePipeline setup, AssetPipeline.Data X, MotionAsset asset, float timestamp, bool mirrored, TimeSeries timeSeries, Actor actor) {
                if(DeepPhaseModule.Curves == null || DeepPhaseModule.Curves.Asset != asset) {
                    Debug.Log("Computing curves in asset: " + asset.name);
                    DeepPhaseModule.ComputeCurves(
                        asset,
                        actor,
                        timeSeries
                    );
                }
                X.Feed(DeepPhaseModule.Curves.Collect(timestamp, mirrored), "MotionFeature-");
            }
        }

    }
}
#endif
