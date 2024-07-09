using UnityEngine;
using UnityEngine.XR;
using System.Collections;
using System.Collections.Generic;
using AI4Animation;
using System;
using UltimateIK;

namespace SIGGRAPH_2024 {
    public class TrackingSystem : MonoBehaviour {
        public enum ID {Head, LeftWrist, RightWrist}
        public enum BUTTON {Primary, Secondary, Trigger, Grip, Joystick}

        [Header("Tracking")]
        public bool AutoUpdate = false;
        public MotionAsset Asset = null;
        public float AssetTimestamp {get; private set;}
        public bool AssetMirrored {get; private set;}

        [SerializeField]
        public Tracker Headset = null;
        [SerializeField]
        public Tracker LeftController = null;
        [SerializeField]
        public Tracker RightController = null;
        public bool CalculateVelocities = false;
        public Transform CalibrationPose = null;
        private float Scale = 1f;
        private float CalibrationLineThickness = 0.05f;
        private Color CalibrationLineColor = Color.white;

        [Header("Camera")]
        public Transform Avatar;
        public Transform Canvas;
        public Camera QuestCamera;
        public Camera ScreenCamera;
        public Camera HeadCamera;
        public bool UpdateCamera = true;
        public float QuestFOV = 90f;
        public float ScreenFOV = 60f;
        public float HeadFOV = 120f;
        public float ViewDistance = 2.5f;
        public Vector3 SelfOffset = new Vector3(0f, 2f, -3f);
        public Vector3 TargetOffset = new Vector3(0f, 1f, 0f);
        public GameViewRenderMode RenderMode = GameViewRenderMode.OcclusionMesh;
        private Transform TrackingSpace;
        private Transform ScreenSpace;
        private Transform HeadSpace;

        [Header("Prediction")]
        public float Framerate = 30f;
        public SentisNetwork TrackingNetwork;
        public SentisNetwork FutureNetwork;
        public Actor Actor;
        public bool EulerIntegration = true;
        public bool Postprocess = true;
        public ACTIVATION SolverActivation = ACTIVATION.Linear;
        public int SolverIterations = 25;
        private IK LWIK, RWIK;

        [Header("Drawing")]
        public bool ShowActor = true;
        public bool ShowCanvas = true;
        public bool ShowTrackers = false;
        public bool DrawTrackerHistory = false;
        public bool DrawMotionHistory = false;
        public bool DrawMotionFuture = false;
        public bool DrawCalibration = false;
        public bool DrawIK = false;

        [NonSerialized] private Color TrackerColor = UltiDraw.Magenta;
        [NonSerialized] Color RootColor = UltiDraw.Cyan;
        [NonSerialized] Color MotionColor = UltiDraw.Cyan;

        public Matrix4x4 Root {get; private set;}
        private TimeSeries TimeSeries;
        public MotionModule.Series TrackerHistory {get; private set;}
        public RootModule.Series RootSeries {get; private set;}
        public MotionModule.Series MotionSeries {get; private set;}
        public StyleModule.Series StyleSeries {get; private set;}

        void OnValidate() {
            if(Actor != null) {
                Actor.gameObject.SetActive(ShowActor);
            }
            if(Canvas != null) {
                Canvas.gameObject.SetActive(ShowCanvas);
            }
        }

        void Awake() {
            Time.fixedDeltaTime = 1f/Framerate;

            TrackingSpace = transform.Find("TrackingSpace");
            if(!TrackingSpace) {
                Debug.LogWarning("Tracking space could not be found.");
            }
            ScreenSpace = transform.Find("ScreenSpace");
            if(!ScreenSpace) {
                Debug.LogWarning("Screen space could not be found.");
            }
            HeadSpace = transform.Find("HeadSpace");
            if(!HeadSpace) {
                Debug.LogWarning("Head space could not be found.");
            }

            Headset.Initalize(this, XRNode.Head);
            LeftController.Initalize(this, XRNode.LeftHand);
            RightController.Initalize(this, XRNode.RightHand);

            Root = transform.GetWorldMatrix();

            TimeSeries = new TimeSeries(5, 5, 0.5f, 0.5f, 3);
            TrackerHistory = new MotionModule.Series(new TimeSeries(5, 0, 0.5f, 0f, 3), Actor, Blueman.TrackerNames);
            RootSeries = new RootModule.Series(TimeSeries, transform);
            MotionSeries = new MotionModule.Series(TimeSeries, Actor, Actor.GetBoneNames());
            StyleSeries = new StyleModule.Series(TimeSeries, "Jump");

            LWIK = IK.Create(Actor.FindTransform(Blueman.LeftShoulderName), Actor.FindTransform(Blueman.LeftWristName));
            LWIK.FindJoint(Blueman.LeftWristTwistName).Active = false;
            LWIK.FindJoint("b_l_forearm").SetJointType(TYPE.HingeZ);
            LWIK.FindJoint("b_l_forearm").SetLowerLimit(-180f);
            LWIK.FindJoint("b_l_forearm").SetUpperLimit(20f);

            RWIK = IK.Create(Actor.FindTransform(Blueman.RightShoulderName), Actor.FindTransform(Blueman.RightWristName));
            RWIK.FindJoint(Blueman.RightWristTwistName).Active = false;
            RWIK.FindJoint("b_r_forearm").SetJointType(TYPE.HingeZ);
            RWIK.FindJoint("b_r_forearm").SetLowerLimit(-180f);
            RWIK.FindJoint("b_r_forearm").SetUpperLimit(20f);

            TrackingNetwork.CreateSession();
            FutureNetwork.CreateSession();
        }

        void OnDestroy() {
            TrackingNetwork.CloseSession();
            FutureNetwork.CloseSession();
        }

        void Update() {
            //Update Camera
            void SetCameraPosition(Transform space, Camera camera, Vector3 position) {
                space.position = Avatar.position - Root.GetPosition() + position - camera.transform.localPosition;
                space.rotation = Quaternion.identity;
            }
            if(TrackingSpace != null && Avatar != null && UpdateCamera) {
                if(IsConnected()) {
                    TrackingSpace.gameObject.SetActive(true);
                    XRSettings.gameViewRenderMode = RenderMode;
                    QuestCamera.fieldOfView = QuestFOV;
                    SetCameraPosition(TrackingSpace, QuestCamera, CalculateViewPoint(ViewDistance));
                } else {
                    TrackingSpace.gameObject.SetActive(false);
                }

                ScreenCamera.fieldOfView = ScreenFOV;
                ScreenSpace.position = Avatar.position + SelfOffset;
                ScreenSpace.LookAt(Avatar.position + TargetOffset);

                HeadCamera.fieldOfView = HeadFOV;
                HeadSpace.position = QuestCamera.transform.position;
                HeadSpace.rotation = QuestCamera.transform.rotation;
            }
        }

        void FixedUpdate() {
            if(AutoUpdate) {
                Iterate();
            }
        }

        public void SetMotionAsset(MotionAsset asset) {
            Asset = asset;
            AssetTimestamp = 0f;
            AssetMirrored = false;

            Root = Asset.GetModule<RootModule>("BodyWorld").GetRootTransformation(AssetTimestamp, AssetMirrored);
            Actor.GetRoot().SetTransformation(Root);
            Actor.SetBoneTransformations(asset.GetFrame(AssetTimestamp).GetBoneTransformations(Actor.GetBoneNames(), AssetMirrored));
            Actor.SetBoneVelocities(asset.GetFrame(AssetTimestamp).GetBoneVelocities(Actor.GetBoneNames(), AssetMirrored));

            TrackerHistory = asset.GetModule<MotionModule>().ExtractSeries(TrackerHistory, AssetTimestamp, AssetMirrored, Blueman.TrackerNames) as MotionModule.Series;
            RootSeries = asset.GetModule<RootModule>("BodyWorld").ExtractSeries(RootSeries, AssetTimestamp, AssetMirrored) as RootModule.Series;
            MotionSeries = asset.GetModule<MotionModule>().ExtractSeries(MotionSeries, AssetTimestamp, AssetMirrored, Actor.GetBoneNames()) as MotionModule.Series;
            StyleSeries = asset.GetModule<StyleModule>().ExtractSeries(StyleSeries, AssetTimestamp, AssetMirrored) as StyleModule.Series;
        }

        public void Iterate() {
            if(Actor != null) {
                Actor.gameObject.SetActive(ShowActor);
            }
            if(Canvas != null) {
                Canvas.gameObject.SetActive(ShowCanvas);
            }
            if(IsCalibrating()) {
                CalibrationPose.gameObject.SetActive(true);
                CalibrationPose.transform.localScale = Vector3.one;
                CalibrationPose.transform.position = Root.GetPosition();
                CalibrationPose.transform.rotation = Root.GetRotation();
                float defaultHeight = FindCalibrationBone().position.y;
                float userHeight = Headset.GetTransformation().GetPosition().y;
                Scale = userHeight / defaultHeight;
            } else {
                CalibrationPose.gameObject.SetActive(false);
            }

            if(Asset != null) {
                AssetTimestamp = Mathf.Repeat(AssetTimestamp + Time.fixedDeltaTime, Asset.GetTotalTime());
            }

            //Update Tracker State
            TrackerHistory.Increment(0, TrackerHistory.Pivot);
            void UpdateTracker(Tracker tracker, MotionModule.Trajectory trajectory) {
                tracker.Device.gameObject.SetActive(ShowTrackers);
                tracker.Device.SetTransformation(tracker.GetDeviceTransformation(Scale));
                trajectory.Transformations[TimeSeries.Pivot] = tracker.GetBoneTransformation(Scale);
                trajectory.Velocities[TimeSeries.Pivot] = CalculateVelocities ? (trajectory.GetPosition(TimeSeries.Pivot) - trajectory.GetPosition(TimeSeries.Pivot-1)) / Time.fixedDeltaTime : tracker.GetBoneVelocity(Scale);
            }
            UpdateTracker(Headset, TrackerHistory.GetTrajectory(Blueman.HeadName));
            UpdateTracker(LeftController, TrackerHistory.GetTrajectory(Blueman.LeftWristName));
            UpdateTracker(RightController, TrackerHistory.GetTrajectory(Blueman.RightWristName));

            //Predict Current State
            PredictTracking(TrackingNetwork);

            //Correct Current State
            if(Postprocess) {
                Actor.Bone hip = Actor.FindBone(Blueman.HipsName);
                Actor.Bone head = Actor.FindBone(Blueman.HeadName);
                Matrix4x4 currentHead = head.GetTransformation();
                Matrix4x4 targetHead = TrackerHistory.GetTrajectory(Blueman.HeadName).Transformations[TrackerHistory.Pivot];
                Vector3 deltaHead = targetHead.GetPosition() - currentHead.GetPosition();
                hip.SetPosition(hip.GetPosition() + deltaHead);
                head.SetRotation(targetHead.GetRotation());

                foreach(IK solver in new IK[]{LWIK, RWIK}) {
                    solver.Activation = SolverActivation;
                    solver.Iterations = SolverIterations;
                    solver.Objectives[0].SolvePosition = true;
                    solver.Objectives[0].SolveRotation = false;
                    solver.Objectives.Last().SolvePosition = true;
                    solver.Objectives.Last().SolveRotation = false;
                }
                
                Matrix4x4 targetLeftWrist = TrackerHistory.GetTrajectory(Blueman.LeftWristName).Transformations[TrackerHistory.Pivot];
                LWIK.Objectives.Last().SetTarget(targetLeftWrist);
                LWIK.Solve();
                LWIK.Joints.Last().Transform.rotation = targetLeftWrist.GetRotation();

                Matrix4x4 targetRightWrist = TrackerHistory.GetTrajectory(Blueman.RightWristName).Transformations[TrackerHistory.Pivot];
                RWIK.Objectives.Last().SetTarget(targetRightWrist);
                RWIK.Solve();
                RWIK.Joints.Last().Transform.rotation = targetRightWrist.GetRotation();
            }

            RootSeries.Increment(0, RootSeries.Pivot);
            RootSeries.Transformations[RootSeries.Pivot] = Root;
            RootSeries.Velocities[RootSeries.Pivot] = (RootSeries.GetPosition(RootSeries.Pivot) - RootSeries.GetPosition(RootSeries.Pivot-1)) / Time.fixedDeltaTime;
            RootSeries.AngularVelocities[RootSeries.Pivot] = Vector3.SignedAngle(RootSeries.GetDirection(RootSeries.Pivot-1), RootSeries.GetDirection(RootSeries.Pivot), Vector3.up) / 180f / Time.fixedDeltaTime;

            MotionSeries.Increment(0, MotionSeries.Pivot);
            for(int i=0; i<MotionSeries.Trajectories.Length; i++) {
                MotionSeries.Trajectories[i].Transformations[MotionSeries.Pivot] = Actor.Bones[i].GetTransformation();
                MotionSeries.Trajectories[i].Velocities[MotionSeries.Pivot] = Actor.Bones[i].GetVelocity();
            }

            //Predict Future States
            PredictFuture(FutureNetwork);
        }

        private void PredictTracking(SentisNetwork network) {
            if(network.GetSession() == null){return;}

            using(SentisNetwork.Input input = network.GetSession().GetInput("X")) {
                Matrix4x4 reference = Root;
                foreach(MotionModule.Trajectory trajectory in TrackerHistory.Trajectories) {
                    for(int i=0; i<TrackerHistory.SampleCount; i++) {
                        Matrix4x4 m = trajectory.Transformations[i].TransformationTo(reference);
                        Vector3 v = trajectory.Velocities[i].DirectionTo(reference);
                        input.Feed(m.GetPosition());
                        input.Feed(m.GetForward());
                        input.Feed(m.GetUp());
                        input.Feed(v);
                    }
                }
            }

            network.RunSession();

            using(SentisNetwork.Output output = network.GetSession().GetOutput("Y")) {
                Matrix4x4 reference = Root;
                Vector3[] seed = Actor.GetBonePositions();
                Root = reference * output.ReadRootDelta();
                Actor.GetRoot().SetTransformation(Root);
                for(int i=0; i<Actor.Bones.Length; i++) {
                    Vector3 position = output.ReadVector3().PositionFrom(Root);
                    Quaternion rotation = output.ReadRotation3D().RotationFrom(Root);
                    Vector3 velocity = output.ReadVector3().DirectionFrom(Root);
                    position = Vector3.Lerp(position, seed[i] + velocity * Time.fixedDeltaTime, EulerIntegration ? 0.5f : 0f);
                    Actor.Bones[i].SetPosition(position);
                    Actor.Bones[i].SetRotation(rotation);
                    Actor.Bones[i].SetVelocity(velocity);
                }
                Actor.RestoreAlignment();
            }
        }

        private void PredictFuture(SentisNetwork network) {
            if(network.GetSession() == null){return;}

            using(SentisNetwork.Input input = network.GetSession().GetInput("X")) {
                for(int i=0; i<=TimeSeries.PivotKey; i++) {
                    int index = TimeSeries.GetKey(i).Index;

                    {
                        Matrix4x4 root = Root;
                        input.FeedXZ(RootSeries.GetPosition(index).PositionTo(root));
                        input.FeedXZ(RootSeries.GetRotation(index).GetForward().DirectionTo(root));
                        input.FeedXZ(RootSeries.GetVelocity(index).DirectionTo(root));
                        input.Feed(RootSeries.AngularVelocities[index]);
                    }

                    foreach(MotionModule.Trajectory trajectory in MotionSeries.Trajectories) {
                        Matrix4x4 reference = trajectory.Transformations[TimeSeries.Pivot];
                        Matrix4x4 m = trajectory.Transformations[index].TransformationTo(reference);
                        Vector3 v = trajectory.Velocities[index].DirectionTo(reference);
                        input.Feed(m.GetPosition());
                        input.Feed(v);
                    }
                }
            }

            network.RunSession();

            using(SentisNetwork.Output output = network.GetSession().GetOutput("Y")) {
                for(int i=TimeSeries.PivotKey+1; i<TimeSeries.KeyCount; i++) {
                    int index = TimeSeries.GetKey(i).Index;

                    {
                        Matrix4x4 root = Root;
                        Vector3 p = output.ReadXZ().PositionFrom(root);
                        Quaternion r = output.ReadRotation2D().RotationFrom(root);
                        Vector3 v = output.ReadXZ().DirectionFrom(root);
                        float a = output.Read();
                        p = Vector3.Lerp(p, RootSeries.GetPosition(index) + v * Time.fixedDeltaTime, EulerIntegration ? 0.5f : 0f);
                        RootSeries.SetPosition(index, p);
                        RootSeries.SetRotation(index, r);
                        RootSeries.SetVelocity(index, v);
                        RootSeries.AngularVelocities[index] = a;
                    }

                    foreach(MotionModule.Trajectory trajectory in MotionSeries.Trajectories) {
                        Matrix4x4 reference = trajectory.Transformations[TimeSeries.Pivot];
                        Vector3 p = output.ReadVector3().PositionFrom(reference);
                        Vector3 v = output.ReadVector3().DirectionFrom(reference);
                        p = Vector3.Lerp(p, trajectory.GetPosition(index) + v * Time.fixedDeltaTime, EulerIntegration ? 0.5f : 0f);
                        trajectory.SetPosition(index, p);
                        trajectory.SetVelocity(index, v);
                    }
                }

                for(int i=TimeSeries.PivotKey; i<TimeSeries.KeyCount; i++) {
                    int index = TimeSeries.GetKey(i).Index;
                
                    StyleSeries.Values[index] = output.Read(StyleSeries.Styles.Length, 0f, 1f);
                }
            }
        }

        public float GetScale() {
            return Scale;
        }

        public bool IsConnected() {
            return Headset.IsConnected() && LeftController.IsConnected() && RightController.IsConnected();
        }

        public Tracker GetTracker(XRNode node) {
            switch(node) {
                case XRNode.Head: return Headset;
                case XRNode.LeftHand: return LeftController;
                case XRNode.RightHand: return RightController;
            }
            return null;
        }

        // public Vector3 GetCameraPosition() {
        //     return Camera.transform.position;
        // }

        // public Quaternion GetCameraRotation() {
        //     return Camera.transform.rotation;
        // }

        // public Vector3 GetCameraDirection() {
        //     return Camera.transform.forward;
        // }

        public Vector3 CalculateViewPoint(float distanceToHead) {
            Vector3 centerEyePosition = 0.5f*(Actor.FindTransform("b_l_eye").position+Actor.FindTransform("b_r_eye").position);
            Vector3 headForward = Actor.GetBoneTransformation(Blueman.HeadName).GetUp();
            return centerEyePosition - distanceToHead * headForward;
        }

        private bool IsCalibrating() {
            return LeftController.GetButton(BUTTON.Primary);
        }

        private Transform FindCalibrationBone() {
            return CalibrationPose.FindRecursive(Blueman.HeadName);
        }

        public Camera GetMainCamera() {
            return IsConnected() ? QuestCamera : ScreenCamera;
        }

        void OnRenderObject() {
            if(DrawTrackerHistory) {
                TrackerHistory.Draw(
                    TrackerColor, 
                    0, 
                    TrackerHistory.KeyCount,
                    2f,
                    drawConnections:true,
                    drawPositions:true,
                    drawRotations:false,
                    drawVelocities:false
                );
            }

            if(DrawMotionHistory || DrawMotionFuture) {
                RootSeries.Draw(
                    DrawMotionHistory ? 0 : TimeSeries.PivotKey, 
                    DrawMotionFuture ? MotionSeries.KeyCount : TimeSeries.PivotKey+1,
                    RootColor, 
                    RootColor.Lighten(0.5f),
                    RootColor.Darken(0.5f), 
                    1f,
                    drawConnections:true,
                    drawPositions:true,
                    drawDirections:true,
                    drawVelocities:false,
                    drawAngularVelocities:false,
                    drawLocks:false
                );
                MotionSeries.Draw(
                    MotionColor, 
                    DrawMotionHistory ? 0 : TimeSeries.PivotKey, 
                    DrawMotionFuture ? MotionSeries.KeyCount : TimeSeries.PivotKey+1,
                    1f,
                    drawConnections:true,
                    drawPositions:true,
                    drawRotations:false,
                    drawVelocities:false
                );
            }

            if(DrawIK) {
                LWIK.Draw();
                RWIK.Draw();
            }
            
            // UltiDraw.Begin();
            // UltiDraw.DrawTranslateGizmo(Root, 0.5f);
            // UltiDraw.End();

            if(DrawCalibration) {
                if(IsCalibrating()) {
                    UltiDraw.Begin();
                    UltiDraw.DrawLine(FindCalibrationBone().position, FindCalibrationBone().position.ZeroY(), CalibrationLineThickness, CalibrationLineColor);
                    UltiDraw.DrawLine(Headset.Device.position, Headset.Device.position.ZeroY(), CalibrationLineThickness, CalibrationLineColor);
                    UltiDraw.End();
                }
            }
        }

        [Serializable]
        public class Tracker {
            public Transform Device = null;
            public Vector3 DeltaPosition = Vector3.zero;
            public Vector3 DeltaRotation = Vector3.zero;
            public string Name = string.Empty;
            [NonSerialized] private TrackingSystem TS;
            [NonSerialized] private InputDevice Input;
            [NonSerialized] private XRNode Node;
            [NonSerialized] private OVRPlugin.Node OVRNode;

            public class InputDevice {
                public UnityEngine.XR.InputDevice Input;
                public InputDevice(UnityEngine.XR.InputDevice input) {
                    Input = input;
                }
            }

            public void Initalize(TrackingSystem ts, XRNode node) {
                TS = ts;
                Node = node;
                if(Node == XRNode.Head) {
                    OVRNode = OVRPlugin.Node.Head;
                }
                if(Node == XRNode.LeftHand) {
                    OVRNode = OVRPlugin.Node.HandLeft;
                }
                if(Node == XRNode.RightHand) {
                    OVRNode = OVRPlugin.Node.HandRight;
                }
            }

            private string GetTag() {
                if(Name != string.Empty) {
                    return Name;
                }
                if(Node == XRNode.Head) {
                    return Blueman.HeadName;
                }
                if(Node == XRNode.LeftHand) {
                    return Blueman.LeftWristName;
                }
                if(Node == XRNode.RightHand) {
                    return Blueman.RightWristName;
                }
                return string.Empty;
            }

            private InputDevice GetDevice() {
                if(Input != null) {
                    return Input;
                }
                List<UnityEngine.XR.InputDevice> devices = new List<UnityEngine.XR.InputDevice>();
                InputDevices.GetDevicesAtXRNode(Node, devices);
                if(devices.Count == 1) {
                    Input = new InputDevice(devices[0]);
                }
                return Input;
            }

            public XRNode GetNode() {
                return Node;
            }

            public bool IsConnected() {
                return GetDevice() != null;
            }

            public Vector2 GetJoystickAxis() {
                Vector2 value = Vector2.zero;
                if(IsConnected()) {
                    if(!GetDevice().Input.TryGetFeatureValue(CommonUsages.primary2DAxis, out value)) {
                        Debug.Log("Joystick axis could not be obtained from node: " + Node.ToString());
                    }
                }
                return value;
            }

            public bool GetButton(BUTTON type) {
                bool value = false;
                if(IsConnected()) {
                    if(type == BUTTON.Primary) {
                        if(!GetDevice().Input.TryGetFeatureValue(CommonUsages.primaryButton, out value)) {
                            Debug.Log("Button " + type.ToString() + " could not be obtained from node: " + Node.ToString());
                        }
                    }
                    if(type == BUTTON.Secondary) {
                        if(!GetDevice().Input.TryGetFeatureValue(CommonUsages.secondaryButton, out value)) {
                            Debug.Log("Button " + type.ToString() + " could not be obtained from node: " + Node.ToString());
                        }
                    }
                    if(type == BUTTON.Trigger) {
                        if(!GetDevice().Input.TryGetFeatureValue(CommonUsages.triggerButton, out value)) {
                            Debug.Log("Button " + type.ToString() + " could not be obtained from node: " + Node.ToString());
                        }
                    }
                    if(type == BUTTON.Grip) {
                        if(!GetDevice().Input.TryGetFeatureValue(CommonUsages.gripButton, out value)) {
                            Debug.Log("Button " + type.ToString() + " could not be obtained from node: " + Node.ToString());
                        }
                    }
                    if(type == BUTTON.Joystick) {
                        if(!GetDevice().Input.TryGetFeatureValue(CommonUsages.primary2DAxisClick, out value)) {
                            Debug.Log("Button " + type.ToString() + " could not be obtained from node: " + Node.ToString());
                        }
                    }
                }
                return value;
            }

            // Transformation between spaces:
            // Tracker = Scale x Bone x Delta
            // Bone = Scale.inverse * Tracker * Delta.inverse

            public Matrix4x4 GetTransformation() {
                if(!IsConnected()) {
                    if(TS.Asset != null) {
                        return 
                            TS.Asset.GetFrame(TS.AssetTimestamp).GetBoneTransformation(GetTag(), false) * 
                            Matrix4x4.TRS(DeltaPosition, Quaternion.Euler(DeltaRotation), Vector3.one);
                    }
                    #if UNITY_EDITOR
                    if(MotionEditor.GetInstance() != null) {
                            return MotionEditor.GetInstance().GetSession().GetActor().GetBoneTransformation(GetTag()) *
                            Matrix4x4.TRS(DeltaPosition, Quaternion.Euler(DeltaRotation), Vector3.one);
                    }
                    #endif
                    return Device.GetWorldMatrix();
                }
                OVRPlugin.Posef result = OVRPlugin.GetNodePoseStateAtTime(OVRPlugin.GetTimeInSeconds(), OVRNode).Pose;
                Vector3 position = new Vector3(result.Position.x, result.Position.y, -result.Position.z);
                Quaternion rotation = new Quaternion(result.Orientation.x, result.Orientation.y, -result.Orientation.z, -result.Orientation.w).normalized;
                return Matrix4x4.TRS(position, rotation, Vector3.one);
            }

            public Vector3 GetVelocity() {
                if(!IsConnected()) {
                    if(TS.Asset != null) {
                        return TS.Asset.GetFrame(TS.AssetTimestamp).GetBoneVelocity(GetTag(), false);
                    }
                    #if UNITY_EDITOR
                    if(MotionEditor.GetInstance() != null) {
                        return MotionEditor.GetInstance().GetSession().GetActor().GetBoneVelocity(GetTag());
                    }
                    #endif
                    return Vector3.zero;
                }
                OVRPlugin.Vector3f result = OVRPlugin.GetNodePoseStateAtTime(OVRPlugin.GetTimeInSeconds(), OVRNode).Velocity;
                return new Vector3(result.x, result.y, -result.z);
            }

            public Matrix4x4 GetDeviceTransformation(float scale) {
                return Matrix4x4.TRS(Vector3.zero, Quaternion.identity, 1f/scale*Vector3.one) * GetTransformation();
            }

            public Vector3 GetDeviceVelocity(float scale) {
                return 1f/scale * GetVelocity();
            }

            public Matrix4x4 GetBoneTransformation(float scale) {
                Matrix4x4 scaling = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, 1f/scale*Vector3.one);
                Matrix4x4 device = GetTransformation();
                Matrix4x4 delta = Matrix4x4.TRS(DeltaPosition, Quaternion.Euler(DeltaRotation), Vector3.one);
                return scaling * device * delta.inverse;
            }

            public Vector3 GetBoneVelocity(float scale) {
                return 1f/scale * GetVelocity();
            }

            //Old code, probably wrong...
            // public Matrix4x4 GetBoneTransformation(float scale) {
            //     Matrix4x4 m = GetDeviceTransformation();
            //     Quaternion rotation = m.GetRotation() * Quaternion.Euler(DeltaRotation);
            //     Vector3 offset = rotation * (scale * DeltaPosition);
            //     Vector3 position = scale * (m.GetPosition() + offset);
            //     return Matrix4x4.TRS(position, rotation, Vector3.one);
            // }
            // public Vector3 GetBoneVelocity(float scale) {
            //     return scale * GetDeviceVelocity();
            // }
        }
    }
}