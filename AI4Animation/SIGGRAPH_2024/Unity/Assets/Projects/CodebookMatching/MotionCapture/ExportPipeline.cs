#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.Collections;
using AI4Animation;

namespace SIGGRAPH_2024 {
    public class ExportPipeline : AssetPipelineSetup {

        public enum MODE {
            TrackerBodyPredictor,
            FutureBodyPredictor,
            LowerBodyPredictor,
            TrackedUpperBodyPredictor,
            UntrackedUpperBodyPredictor
        };
        public MODE Mode = MODE.TrackerBodyPredictor;

        public bool WriteMirror = true;
        public bool SubsampleTargetFramerate = true;
        public int SequenceLength = 15;

        private DateTime Timestamp;
        private float Progress = 0f;
        private float SamplesPerSecond = 0f;
        private int Samples = 0;
        private int Sequence = 0;

        private AssetPipeline.Data.File S;
        private AssetPipeline.Data X, Y;

        public override void Inspector() {
            Mode = (MODE)EditorGUILayout.EnumPopup("Mode", Mode);

            WriteMirror = EditorGUILayout.Toggle("Write Mirror", WriteMirror);
            SubsampleTargetFramerate = EditorGUILayout.Toggle("Subsample Target Framerate", SubsampleTargetFramerate);
            SequenceLength = EditorGUILayout.IntField("Sequence Length", SequenceLength);
            EditorGUI.BeginDisabledGroup(true);
            EditorGUILayout.FloatField("Export Framerate", Pipeline.GetEditor().TargetFramerate);
            EditorGUILayout.TextField("Export Path", AssetPipeline.Data.GetExportPath());
            EditorGUI.EndDisabledGroup();
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
            X = new AssetPipeline.Data("Input");
            Y = new AssetPipeline.Data("Output");
        }

        private void WriteSequenceInfo(int sequence, float timestamp, bool mirrored, MotionAsset asset) {
            //Sequence - Timestamp - Mirroring - Name - GUID
            S.WriteLine(
                sequence.ToString() + AssetPipeline.Data.Separator + 
                timestamp + AssetPipeline.Data.Separator +
                (mirrored ? "Mirrored" : "Standard") + AssetPipeline.Data.Separator +
                asset.name + AssetPipeline.Data.Separator +
                Utility.GetAssetGUID(asset)
            );
        }

        public override IEnumerator Iterate(MotionAsset asset) {
            Debug.Log("Asset: " + asset.name);
            Pipeline.GetEditor().AutoSave = false;
            Pipeline.GetEditor().LoadSession(Utility.GetAssetGUID(asset));
            Container.ResetStatic();
            if(asset.Export) {
                foreach(bool value in new bool[]{false,true}) {
                    Pipeline.GetEditor().SetMirror(value);
                    if(value && !WriteMirror) {
                        continue;
                    }
                    Sequence += 1;
                    int step = SubsampleTargetFramerate ? Mathf.RoundToInt(Pipeline.GetEditor().GetSession().Asset.Framerate / Pipeline.GetEditor().TargetFramerate) : 1;
                    for(int f=0; f<asset.Frames.Length; f+=step) {
                        Frame frame = asset.Frames[f];

                        if(!asset.InSequences(frame)) {
                            continue;
                        }

                        if(Mode == MODE.TrackerBodyPredictor) {
                            TrackerBodyPredictor.Export(this, X, Y, frame.Timestamp);
                        }
                        if(Mode == MODE.FutureBodyPredictor) {
                            FutureBodyPredictor.Export(this, X, Y, frame.Timestamp);
                        }
                        if(Mode == MODE.LowerBodyPredictor) {
                            LowerBodyPredictor.Export(this, X, Y, frame.Timestamp);
                        }
                        if(Mode == MODE.TrackedUpperBodyPredictor) {
                            TrackedUpperBodyPredictor.Export(this, X, Y, frame.Timestamp);
                        }
                        if(Mode == MODE.UntrackedUpperBodyPredictor) {
                            UntrackedUpperBodyPredictor.Export(this, X, Y, frame.Timestamp);
                        }

                        X.Store();
                        Y.Store();
                        
                        WriteSequenceInfo(Sequence, frame.Timestamp, Pipeline.GetEditor().Mirror, asset);

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
            Pipeline.GetEditor().AutoSave = true;
        }

        public override void Callback() {
            Resources.UnloadUnusedAssets();
        }

        public override void Finish() {
            S.Close();
            X.Finish();
            Y.Finish();
        }

        private class TrackerBodyPredictor {
            public static void Export(ExportPipeline setup, AssetPipeline.Data X, AssetPipeline.Data Y, float timestamp) {
                Container current = new Container(setup, timestamp);
                float then = UnityEngine.Random.Range(timestamp-current.TimeSeries.PastWindow, timestamp+current.TimeSeries.FutureWindow);
                Matrix4x4 from = current.GetRootModule().GetRootTransformation(then, current.Mirror);
                Matrix4x4 to = current.GetRootModule().GetRootTransformation(timestamp, current.Mirror);

                {
                    foreach(MotionModule.Trajectory trajectory in (current.GetMotionModule().ExtractSeries(current.TimeSeries, current.Timestamp, current.Mirror, Blueman.TrackerNames) as MotionModule.Series).Trajectories) {
                        for(int i=0; i<=current.TimeSeries.Pivot; i++) {
                            string id = current.TimeSeries.Samples[i].Timestamp.ToString();
                            int index = current.TimeSeries.Samples[i].Index;
                            Matrix4x4 m = trajectory.Transformations[index].TransformationTo(from);
                            Vector3 v = trajectory.Velocities[index].DirectionTo(from);
                            X.Feed(m.GetPosition(), trajectory.Name+"Position"+id);
                            X.Feed(m.GetForward(), trajectory.Name+"Forward"+id);
                            X.Feed(m.GetUp(), trajectory.Name+"Up"+id);
                            X.Feed(v, trajectory.Name+"Velocity"+id);
                        }
                    }
                }

                {
                    Matrix4x4 delta = to.TransformationTo(from);
                    Y.Feed(new Vector3(delta.GetPosition().x, Vector3.SignedAngle(Vector3.forward, delta.GetForward(), Vector3.up), delta.GetPosition().z), "RootUpdate");
                    setup.OutputPose(current, timestamp, to, Blueman.UpperBodyIndices);
                }
            }
        }

        private class FutureBodyPredictor {
            public static void Export(ExportPipeline setup, AssetPipeline.Data X, AssetPipeline.Data Y, float timestamp) {
                Container container = new Container(setup, timestamp);

                //Input
                {
                    for(int i=0; i<=container.TimeSeries.PivotKey; i++) {
                        string id = container.TimeSeries.GetKey(i).Timestamp.ToString();
                        float t = timestamp + container.TimeSeries.GetKey(i).Timestamp;

                        {
                            Matrix4x4 root = container.Root;
                            X.FeedXZ(container.GetRootModule().GetRootPosition(t, container.Mirror).PositionTo(root), "RootPosition"+id, sigmaGroup:"RootPosition");
                            X.FeedXZ(container.GetRootModule().GetRootRotation(t, container.Mirror).GetForward().DirectionTo(root), "RootDirection"+id, sigmaGroup:"RootDirection");
                            X.FeedXZ(container.GetRootModule().GetRootVelocity(t, container.Mirror).DirectionTo(root), "RootVelocity"+id, sigmaGroup:"RootVelocity");
                            X.Feed(container.GetRootModule().GetAngularVelocity(t, container.Mirror), "RootAngularVelocity"+id, sigmaGroup:"RootAngularVelocity");
                        }
                        
                        foreach(int bone in Blueman.UpperBodyIndices) {
                            string name = container.Asset.Source.Bones[bone].GetName();
                            Matrix4x4 reference = container.GetMotionModule().GetBoneTransformation(container.Timestamp, container.Mirror, bone);
                            Matrix4x4 m = container.GetMotionModule().GetBoneTransformation(t, container.Mirror, bone).TransformationTo(reference);
                            Vector3 v = container.GetMotionModule().GetBoneVelocity(t, container.Mirror, bone).DirectionTo(reference);
                            X.Feed(m.GetPosition(), name+"Position"+id, sigmaGroup:name+"Position");
                            X.Feed(v, name+"Velocity"+id, sigmaGroup:name+"Velocity");
                        }
                    }
                }
                //Output
                {
                    for(int i=container.TimeSeries.PivotKey+1; i<container.TimeSeries.KeyCount; i++) {
                        string id = container.TimeSeries.GetKey(i).Timestamp.ToString();
                        float t = timestamp + container.TimeSeries.GetKey(i).Timestamp;

                        {
                            Matrix4x4 root = container.Root;
                            Y.FeedXZ(container.GetRootModule().GetRootPosition(t, container.Mirror).PositionTo(root), "RootPosition"+id);
                            Y.FeedXZ(container.GetRootModule().GetRootRotation(t, container.Mirror).GetForward().DirectionTo(root), "RootDirection"+id);
                            Y.FeedXZ(container.GetRootModule().GetRootVelocity(t, container.Mirror).DirectionTo(root), "RootVelocity"+id);
                            Y.Feed(container.GetRootModule().GetAngularVelocity(t, container.Mirror), "RootAngularVelocity"+id);
                        }

                        foreach(int bone in Blueman.UpperBodyIndices) {
                            string name = container.Asset.Source.Bones[bone].GetName();
                            Matrix4x4 reference = container.GetMotionModule().GetBoneTransformation(container.Timestamp, container.Mirror, bone);
                            Matrix4x4 m = container.GetMotionModule().GetBoneTransformation(t, container.Mirror, bone).TransformationTo(reference);
                            Vector3 v = container.GetMotionModule().GetBoneVelocity(t, container.Mirror, bone).DirectionTo(reference);
                            Y.Feed(m.GetPosition(), name+"Position"+id);
                            Y.Feed(v, name+"Velocity"+id);
                        }
                    }

                    for(int i=container.TimeSeries.PivotKey; i<container.TimeSeries.KeyCount; i++) {
                        string id = container.TimeSeries.GetKey(i).Timestamp.ToString();
                        float t = timestamp + container.TimeSeries.GetKey(i).Timestamp;

                        Y.Feed(container.GetStyleModule().GetValues(t, container.Mirror), "Style"+id);
                    }
                }
            }
        }

        private class LowerBodyPredictor {
            public static void Export(ExportPipeline setup, AssetPipeline.Data X, AssetPipeline.Data Y, float timestamp) {
                Container container = new Container(setup, timestamp);

                //Input
                {
                    setup.InputPose(container, container.Timestamp, container.Root, Blueman.LowerBodyIndices, usePositions:true, useRotations:true, useVelocities:true);

                    for(int i=container.TimeSeries.PivotKey+1; i<container.TimeSeries.KeyCount; i++) {
                        string id = container.TimeSeries.GetKey(i).Timestamp.ToString();
                        float t = container.Timestamp + container.TimeSeries.GetKey(i).Timestamp;

                        X.FeedXZ(container.GetRootModule().GetRootVelocity(t, container.Mirror, container.TimeSeries).DirectionTo(container.Root), "RootVelocity"+id, sigmaGroup:"RootVelocity");
                        X.Feed(container.GetRootModule().GetAngularVelocity(t, container.Mirror, container.TimeSeries), "RootAngularVelocity"+id, sigmaGroup:"RootAngularVelocity");

                        X.Feed(container.GetAvatarModule().GetBoneTransformation(t, container.Mirror, Blueman.HipsIndex, container.TimeSeries).GetPosition().PositionTo(container.Root), "HipsPosition"+id, sigmaGroup:"HipsPosition");
                        X.Feed(container.GetAvatarModule().GetUpperBodyCenter(t, container.Mirror, container.TimeSeries).PositionTo(container.Root), "UpperBodyCenter"+id, sigmaGroup:"UpperBodyCenter");
                        X.Feed(container.GetAvatarModule().GetUpperBodyOrientation(t, container.Mirror).DirectionTo(container.Root), "UpperBodyOrientation"+id, sigmaGroup:"UpperBodyOrientation");
                    }
                    
                }
                //Output
                {
                    for(int i=0; i<=setup.SequenceLength; i++) {
                        string id = i.ToString();
                        float from = timestamp + (i-1) / setup.Pipeline.GetEditor().TargetFramerate;
                        float to = timestamp + i / setup.Pipeline.GetEditor().TargetFramerate;

                        Matrix4x4 fromRoot = container.GetRootModule().GetRootTransformation(from, container.Mirror);
                        Matrix4x4 toRoot = container.GetRootModule().GetRootTransformation(to, container.Mirror);
                        Matrix4x4 delta = toRoot.TransformationTo(fromRoot);
                        Y.Feed(new Vector3(delta.GetPosition().x, Vector3.SignedAngle(Vector3.forward, delta.GetForward(), Vector3.up), delta.GetPosition().z), "RootUpdate"+id);
                        Y.Feed(container.GetRootModule().GetRootLock(to, container.Mirror, container.TimeSeries), "RootLock"+id);
                        Y.Feed(container.GetContactModule().GetContacts(to, container.Mirror), "Contacts"+id+":");
                        setup.OutputPose(container, to, toRoot, indices:Blueman.LowerBodyIndices, usePositions:true, useRotations:true, useVelocities:true, id:id);
                    }
                }
            }
        }

        private class TrackedUpperBodyPredictor {
            public static void Export(ExportPipeline setup, AssetPipeline.Data X, AssetPipeline.Data Y, float timestamp) {
                Container container = new Container(setup, timestamp);

                //Input
                {
                    for(int i=0; i<=container.TimeSeries.PivotKey; i++) {
                        float t = container.Timestamp + container.TimeSeries.GetKey(i).Timestamp - setup.Pipeline.GetEditor().TargetDeltaTime;
                        setup.InputPose(container, t, container.Root, Blueman.LowerBodyIndices, usePositions:true, useRotations:true, useVelocities:true);
                    }
                    for(int i=0; i<=container.TimeSeries.PivotKey; i++) {
                        float t = container.Timestamp + container.TimeSeries.GetKey(i).Timestamp;
                        setup.InputPose(container, t, container.Root, Blueman.TrackerIndices, usePositions:true, useRotations:true, useVelocities:true);
                    }
                }
                //Output
                {
                    setup.OutputPose(container, container.Timestamp, container.Root, Blueman.UpperBodyIndices);
                }
            }
        }

        private class UntrackedUpperBodyPredictor {
            public static void Export(ExportPipeline setup, AssetPipeline.Data X, AssetPipeline.Data Y, float timestamp) {
                Container container = new Container(setup, timestamp);

                //Input
                {
                    for(int i=0; i<=container.TimeSeries.PivotKey; i++) {
                        float t = container.Timestamp + container.TimeSeries.GetKey(i).Timestamp - setup.Pipeline.GetEditor().TargetDeltaTime;
                        setup.InputPose(container, t, container.Root, Blueman.LowerBodyIndices, usePositions:true, useRotations:true, useVelocities:true);
                    }
                    {
                        float t = container.Timestamp - setup.Pipeline.GetEditor().TargetDeltaTime;
                        setup.InputPose(container, t, container.Root, Blueman.TrackerIndices, usePositions:true, useRotations:true, useVelocities:true);
                    }
                }
                
                //Output
                {
                    setup.OutputPose(container, container.Timestamp, container.Root, Blueman.UpperBodyIndices);
                }
            }
        }

        private void InputPose(Container current, float timestamp, Matrix4x4 reference, int[] indices=null, bool usePositions=true, bool useRotations=true, bool useVelocities=true, string id=null) {
            indices = indices == null ? ArrayExtensions.CreateEnumerated(current.Asset.Source.Bones.Length) : indices;
            Matrix4x4[] transformations = current.Asset.GetFrame(timestamp).GetBoneTransformations(indices, current.Mirror);
            Vector3[] velocities = current.Asset.GetFrame(timestamp).GetBoneVelocities(indices, current.Mirror);
            for(int k=0; k<indices.Length; k++) {
                string name = current.Asset.Source.Bones[indices[k]].GetName();
                Matrix4x4 m = transformations[k].TransformationTo(reference);
                Vector3 v = velocities[k].DirectionTo(reference);
                if(usePositions) {
                    X.Feed(m.GetPosition(), "BonePosition"+":"+name+":"+id);
                }
                if(useRotations) {
                    X.Feed(m.GetForward(), "BoneForward"+":"+name+":"+id);
                    X.Feed(m.GetUp(), "BoneUp"+":"+name+":"+id);
                }
                if(useVelocities) {
                    X.Feed(v, "BoneVelocity"+":"+name+":"+id);
                }
            }
        }
        
        private void OutputPose(Container current, float timestamp, Matrix4x4 reference, int[] indices=null, bool usePositions=true, bool useRotations=true, bool useVelocities=true, string id=null) {
            indices = indices == null ? ArrayExtensions.CreateEnumerated(current.Asset.Source.Bones.Length) : indices;
            Matrix4x4[] transformations = current.Asset.GetFrame(timestamp).GetBoneTransformations(indices, current.Mirror);
            Vector3[] velocities = current.Asset.GetFrame(timestamp).GetBoneVelocities(indices, current.Mirror);
            for(int k=0; k<indices.Length; k++) {
                string name = current.Asset.Source.Bones[indices[k]].GetName();
                Matrix4x4 m = transformations[k].TransformationTo(reference);
                Vector3 v = velocities[k].DirectionTo(reference);
                if(usePositions) {
                    Y.Feed(m.GetPosition(), "BonePosition"+":"+name+":"+id);
                }
                if(useRotations) {
                    Y.Feed(m.GetForward(), "BoneForward"+":"+name+":"+id);
                    Y.Feed(m.GetUp(), "BoneUp"+":"+name+":"+id);
                }
                if(useVelocities) {
                    Y.Feed(v, "BoneVelocity"+":"+name+":"+id);
                }
            }
        }
        
        private class Container {
            public ExportPipeline Setup;
            public MotionAsset Asset;
            public Frame Frame;
            public bool Mirror;
            public float Timestamp;

            public TimeSeries TimeSeries;
            
            public static RootModule RootModule;
            public static ContactModule ContactModule;
            public static MotionModule MotionModule;
            public static StyleModule StyleModule;
            public static AvatarModule AvatarModule;
            public static CuboidMapModule CuboidMapModule;
            public static ActorSequenceModule ActorSequenceModule;

            public Matrix4x4 Root {get {return GetRootModule().GetRootTransformation(Timestamp, Mirror);}}

            public Container(ExportPipeline setup, float timestamp) {
                Setup = setup;
                MotionEditor editor = setup.Pipeline.GetEditor();
                
                editor.LoadFrame(timestamp);
                Asset = editor.GetSession().Asset;
                Frame = editor.GetCurrentFrame();
                Mirror = editor.Mirror;
                Timestamp = timestamp;
                TimeSeries = editor.GetTimeSeries();

                Blueman.RegisterIndices(Asset);
            }

            public static void ResetStatic() {
                RootModule = null;
                ContactModule = null;
                MotionModule = null;
                StyleModule = null;
                AvatarModule = null;
                CuboidMapModule = null;
                ActorSequenceModule = null;
            }

            public RootModule GetRootModule() {
                if(RootModule == null) {
                    RootModule = Asset.GetModule<RootModule>("BodyWorld");
                }
                return RootModule;
            }

            public ContactModule GetContactModule() {
                if(ContactModule == null) {
                    ContactModule = Asset.GetModule<ContactModule>();
                }
                return ContactModule;
            }

            public MotionModule GetMotionModule() {
                if(MotionModule == null) {
                    MotionModule = Asset.GetModule<MotionModule>();
                }
                return MotionModule;
            }

            public StyleModule GetStyleModule() {
                if(StyleModule == null) {
                    StyleModule = Asset.GetModule<StyleModule>();
                }
                return StyleModule;
            }

            public AvatarModule GetAvatarModule() {
                if(AvatarModule == null) {
                    AvatarModule = Asset.GetModule<AvatarModule>();
                }
                return AvatarModule;
            }

            public ActorSequenceModule GetActorSequenceModule() {
                if(ActorSequenceModule == null) {
                    ActorSequenceModule = Asset.GetModule<ActorSequenceModule>();
                }
                return ActorSequenceModule;
            }

            public CuboidMapModule GetCuboidMapModule() {
                if(CuboidMapModule == null) {
                    CuboidMapModule = Asset.GetModule<CuboidMapModule>();
                }
                return CuboidMapModule;
            }

        }
    }
}
#endif