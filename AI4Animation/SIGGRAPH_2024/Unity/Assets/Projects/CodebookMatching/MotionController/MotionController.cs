    using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using AI4Animation;
using MagicIK;
using UnityEngine.EventSystems;

namespace SIGGRAPH_2024 {
    public class MotionController : MonoBehaviour {
        public enum UPDATE_MODE {Update, FixedUpdate}
        public enum UPPERBODY_MODE {Tracked, Untracked}
        public enum TRANSFORM_MODE {World, Local}

        public UPDATE_MODE UpdateMode = UPDATE_MODE.Update;

        [Header("Network Settings")]
        [Range(1f,60f)] public float Framerate = 30f;
        [Range(1,15)] public int SequenceLength = 15;
        public int CodebookChannels = 128;
        public int CodebookDim = 8;
        public SentisNetwork LowerBodyNetwork;
        public SentisNetwork TrackedUpperBodyNetwork;
        public SentisNetwork UntrackedUpperBodyNetwork;

        [Header("Character Control")]
        public Vector3 MotionOffset = Vector3.zero;
        public bool RestrictArea = true;
        public bool RestrictBounds = true;
        public Vector2 BoundingBox = new Vector2(-6.5f, 6.5f);
        [Range(0f, 5f)] public float ControlRadius = 1f;
        [Range(0f, 5f)] public float RotationRadius = 0.25f;
        [Range(1,15)] public int RolloutLength = 1;
        [Range(1,25)] public int KNN = 1;
        [Range(0f,1f)] public float Noise = 0f;
        [Range(0f,1f)] public float TargetBias = 1f;
        [Range(0f,1f)] public float TargetBlend = 0.75f;
        [Range(0f,1f)] public float HybridBlend = 1f;
        [Range(0f,1f)] public float IKBlend = 0.25f;
        public TRANSFORM_MODE JoystickSpace = TRANSFORM_MODE.Local;
        public bool EulerIntegration = true;
        public bool RestoreAlignment = true;
        public bool MatchEntryPoint = true;
        public bool EnableRootLocking = true;
        public bool UpdateTrackingSystem = true;
        public float ControlStrength = 2f;
        public float TurnStrength = 2f;
        public float MoveSpeed = 2f;
        public LayerMask SelectionMask = ~0;
        public UPPERBODY_MODE UpperBodyMode = UPPERBODY_MODE.Tracked;
        public TRANSFORM_MODE HeadPositionTransform = TRANSFORM_MODE.World;
        public TRANSFORM_MODE HeadRotationTransform = TRANSFORM_MODE.World;
        public TRANSFORM_MODE LeftPositionTransform = TRANSFORM_MODE.World;
        public TRANSFORM_MODE LeftRotationTransform = TRANSFORM_MODE.World;
        public TRANSFORM_MODE RightPositionTransform = TRANSFORM_MODE.World;
        public TRANSFORM_MODE RightRotationTransform = TRANSFORM_MODE.World;

        [Header("Inverse Kinematics")]
        public bool SolverUpperBodyIK = true;
        public bool SolveLowerBodyIK = true;
        public IK LowerBodyIK, UpperBodyIK;
        private float LeftAnkleBaseline, LeftToeBaseline;
        private float RightAnkleBaseline, RightToeBaseline;
        private Vector3 LeftToeUp = Vector3.zero;
        private Vector3 RightToeUp = Vector3.zero;
        
        [Header("Drawing Settings")]
        public Camera Screen;
        public bool DrawTarget = false;
        public bool DrawWireCircles = false;
        public bool DrawRootControl = false;
        public bool DrawAvatarControl = false;
        public bool DrawControlHistory = false;
        public bool DrawCurrentSequence = false;
        public bool DrawAllSequences = false;
        public bool DrawSimilarityMap = false;
        public bool DrawStateVariables = false;
        public bool DrawHistory = false;
        public bool DrawInferenceTime = false;
        public bool DrawInverseKinematics = false;
        public bool DrawCodebook = false;
        public bool DrawLabels = true;
        public float LabelSize = 0.045f;
        public float TrajectoryScale = 1f;
        public Color PositionColor = UltiDraw.Black;
        [NonSerialized] public Color DirectionColor = UltiDraw.Orange;
        public Color VelocityColor = UltiDraw.Green;
        public Color SequenceColor = UltiDraw.White;
        public float TargetScale = 1f;
        public Color TargetCube = UltiDraw.Cyan;
        public Color TargetArrow = UltiDraw.Magenta;
        public UltiDraw.GUIRect CodebookRect = new UltiDraw.GUIRect(0.1f, 0.4875f, 0.182f, 0.9f);
        public UltiDraw.GUIRect CategoricalRect = new UltiDraw.GUIRect(0.9f, 0.4875f, 0.182f, 0.9f);
        public UltiDraw.GUIRect DrawArea = new UltiDraw.GUIRect(0.5f, 0.95f, 0.25f, 0.1f);
        public float LabelShift = 0f;
        [Range(0f,1f)] public float SequenceOpacity = 0.5f;

        //Runtime Variables
        private Actor Actor;
        private TrackingSystem TrackingSystem;

        private RootModule.Series RootControl;
        private AvatarModule.Series AvatarControl;
        private RootModule.Series TargetControl;
        private MotionModule.Series MotionHistory;

        private string[] LowerBodyNames = Blueman.LowerBodyNames;
        private string[] UpperBodyNames = Blueman.UpperBodyNames;
        private Actor.Bone[] LowerBodyBones = null;
        private Actor.Bone[] UpperBodyBones = null;
        private Actor.Bone[] TrackerBones = null;
        private List<State[]> Sequences = new List<State[]>();
        private State Current = null;

        private Matrix4x4 PreviousRoot;
        private Vector3[] PreviousPositions;
        private Quaternion[] PreviousRotations;
        private Vector3[] PreviousVelocities;

        private float HybridSteps = 0;
        private float HybridWeight {get{return HybridBlend * (1f-Mathf.Pow(1f-HybridSteps/(2*TimeSeries.FutureSamples), 2f));}}
        public TimeSeries TimeSeries {get; private set;}
        private int Step = 0;

        private float[][] Code = null;
        
        void Awake() {
            // Utility.SetFPS(Mathf.RoundToInt(Framerate));
            // Utility.SetFPS(-1);
            Time.fixedDeltaTime = 1f/Framerate;
            
            Actor = GetComponent<Actor>();
            TrackingSystem = FindObjectOfType<TrackingSystem>();
            // TrackingSystem = Array.Find(FindObjectsOfType<TrackingSystem>(), x => x.Avatar == transform);
            if(TrackingSystem == null) {
                Debug.LogError("Tracking system could not be found!");
            }
        }

        void OnEnable() {
            TimeSeries = new TimeSeries(5, 5, 0.5f, 0.5f, 3);

            LowerBodyBones = Actor.FindBones(LowerBodyNames);
            UpperBodyBones = Actor.FindBones(UpperBodyNames);
            TrackerBones = Actor.FindBones(Blueman.TrackerNames);

            PreviousRoot = Actor.GetRoot().GetWorldMatrix();
            PreviousPositions = Actor.GetBonePositions();
            PreviousRotations = Actor.GetBoneRotations();
            PreviousVelocities = Actor.GetBoneVelocities();

            RootControl = new RootModule.Series(TimeSeries, Actor.GetRoot());
            AvatarControl = new AvatarModule.Series(TimeSeries, Actor);
            TargetControl = new RootModule.Series(TimeSeries, Matrix4x4.TRS(MotionOffset, Quaternion.identity, Vector3.one));
            MotionHistory = new MotionModule.Series(TimeSeries, Actor, Blueman.LowerBodyNames);

            LeftAnkleBaseline = Actor.GetBonePosition(Blueman.LeftAnkleName).y;
            LeftToeBaseline = Actor.GetBonePosition(Blueman.LeftToeName).y;
            LeftToeUp = Actor.GetBoneRotation(Blueman.LeftToeName).GetUp();

            RightAnkleBaseline = Actor.GetBonePosition(Blueman.RightAnkleName).y;
            RightToeBaseline = Actor.GetBonePosition(Blueman.RightToeName).y;
            RightToeUp = Actor.GetBoneRotation(Blueman.RightToeName).GetUp();

            LowerBodyNetwork.CreateSession();
            TrackedUpperBodyNetwork.CreateSession();
            UntrackedUpperBodyNetwork.CreateSession();
        }

        void OnDisable() {
            LowerBodyNetwork.CloseSession();
            TrackedUpperBodyNetwork.CloseSession();
            UntrackedUpperBodyNetwork.CloseSession();
        }

        void Update() {
            if(UpdateMode == UPDATE_MODE.Update) {
                Iterate();
            }
        }

        void FixedUpdate() {
            if(UpdateMode == UPDATE_MODE.FixedUpdate) {
                Iterate();
            }
        }

        private void Iterate() {
            if(UpdateTrackingSystem) {
                TrackingSystem.Iterate();
            }
            Control();
            Animate();
        }

        public void Initialize(Matrix4x4 referencePoint, Matrix4x4[] referencePose, Vector3[] referenceVelocities) {
            Actor.transform.SetTransformation(referencePoint);
            Actor.SetBoneTransformations(referencePose);
            Actor.SetBoneVelocities(referenceVelocities);
            Actor.transform.position += MotionOffset;

            PreviousRoot = Actor.GetRoot().GetWorldMatrix();
            PreviousPositions = Actor.GetBonePositions();
            PreviousRotations = Actor.GetBoneRotations();
            PreviousVelocities = Actor.GetBoneVelocities();

            RootControl = new RootModule.Series(TimeSeries, Actor.GetRoot());
            AvatarControl = new AvatarModule.Series(TimeSeries, Actor);
            TargetControl = new RootModule.Series(TimeSeries, Matrix4x4.TRS(MotionOffset, Quaternion.identity, Vector3.one));
            MotionHistory = new MotionModule.Series(TimeSeries, Actor, Blueman.LowerBodyNames);

            Sequences = new List<State[]>();
            Current = null;

            HybridSteps = 0;

            foreach(Solver.Objective objective in LowerBodyIK.Solver.Objectives) {
                objective.Position = Actor.GetBonePosition(objective.Node.Transform.name);
                objective.Rotation = Actor.GetBoneRotation(objective.Node.Transform.name);
            }
        }

        private void Control() {
            UpdateTargetControl(TrackingSystem.GetMainCamera());
            for(int i=TimeSeries.PivotKey; i<TimeSeries.KeyCount; i++) {
                int index = TimeSeries.GetKey(i).Index;

                {
                    RootControl.Transformations[index] = TargetControl.Transformations[index] * TrackingSystem.RootSeries.GetTransformation(index);
                    RootControl.Transformations[index] = Utility.Interpolate(RootControl.Transformations[index], RootControl.Transformations.Last(), TargetBias * HybridWeight * index.Ratio(TimeSeries.Pivot, TimeSeries.SampleCount-1));
                    
                    TimeSeries.Sample target = TimeSeries.Samples.Last();
                    TimeSeries.Sample current = TimeSeries.GetKey(0f);
                    TimeSeries.Sample b = TimeSeries.GetKey(i);
                    TimeSeries.Sample a = TimeSeries.GetKey(Mathf.Max(0f, TimeSeries.GetKey(i-1).Timestamp));
                    Vector3 v_current = (RootControl.GetPosition(target.Index) - Actor.GetRoot().position) / (target.Timestamp-current.Timestamp);
                    Vector3 v_step = Vector3.zero;
                    float a_current = Vector3.SignedAngle(Actor.GetRoot().forward, RootControl.GetDirection(target.Index), Vector3.up) / 180f / (target.Timestamp-current.Timestamp);
                    float a_step = 0f;
                    RootControl.Velocities[index] = ControlStrength * (v_current + v_step);
                    RootControl.AngularVelocities[index] = TurnStrength * (a_current + a_step);
                }

                {
                    //State
                    Matrix4x4 from = TrackingSystem.RootSeries.Transformations[index];
                    Matrix4x4 to = RootControl.Transformations[index];
                    AvatarControl.HipsTransformations[index] = TrackingSystem.MotionSeries.GetTrajectory(Blueman.HipsName).Transformations[index].TransformationFromTo(from, to);
                    AvatarControl.HipsTransformations[index] = AvatarControl.HipsTransformations[index];
                }

                {
                    //State
                    Matrix4x4 from = TrackingSystem.RootSeries.Transformations[index];
                    Matrix4x4 to = RootControl.Transformations[index];

                    {
                        Vector3 value = Vector3.zero;
                        for(int k=1; k<TrackingSystem.MotionSeries.Trajectories.Length; k++) {
                            value += TrackingSystem.MotionSeries.Trajectories[k].GetPosition(index) - TrackingSystem.MotionSeries.Trajectories.First().GetPosition(index);
                        }
                        value = value.normalized;
                        AvatarControl.Orientations[index] = value.DirectionFromTo(from, to);
                        AvatarControl.Orientations[index] = AvatarControl.Orientations[index];
                    }
                    
                    {
                        Vector3 value = Vector3.zero;
                        for(int k=0; k<TrackingSystem.MotionSeries.Trajectories.Length; k++) {
                            value += TrackingSystem.MotionSeries.Trajectories[k].GetPosition(index);
                        }
                        value /= TrackingSystem.MotionSeries.Trajectories.Length;
                        AvatarControl.Centers[index] = value.PositionFromTo(from, to);
                        AvatarControl.Centers[index] = AvatarControl.Centers[index];
                    }
                }
            }
        }

        private void Animate() {
            if(Step == 0 || Current == null || Current.Next == null) {
                //Reset rollout step to zero
                Step = 0;

                //Predict candidate sequences
                Sequences = PredictSequences(LowerBodyNetwork);

                //Compute distances across all candidate sequences
                for(int i=0; i<KNN; i++) {
                    for(int j=0; j<Sequences[i].Length; j++) {
                        Sequences[i][j].ComputeDistance(Actor, LowerBodyNames);
                    }
                }

                if(MatchEntryPoint) {
                    //Match the state as entry point that is most similar to the current state
                    foreach(State candidate in Sequences.ToArray().Flatten().OrderBy(x => x.Distance)) {
                        Current = candidate;
                        if(Current.Next != null) {
                            break;
                        }
                    }
                } else {
                    //Select the closest frame predicted by the model
                    Current = Sequences[0][0];
                }
            }

            //Advance current state by one step
            if(Current.Next != null) {
                Current = Current.Next;
            }

            //Predict Lower Body
            {
                Matrix4x4 root = Actor.GetRoot().GetWorldMatrix() * Utility.Interpolate(Current.Delta, Matrix4x4.identity, EnableRootLocking ? Current.RootLock : 0f);
                Actor.GetRoot().SetTransformation(root);
                for(int i=0; i<LowerBodyBones.Length; i++) {
                    Vector3 position = Vector3.Lerp(Current.Positions[i].PositionFrom(root), PreviousPositions[LowerBodyBones[i].GetIndex()] + Current.Velocities[i].DirectionFrom(root) * Time.fixedDeltaTime, EulerIntegration ? 0.5f : 0f);
                    Quaternion rotation = Current.Rotations[i].RotationFrom(root);
                    Vector3 velocity = Current.Velocities[i].DirectionFrom(root);
                    LowerBodyBones[i].SetTransformation(Matrix4x4.TRS(position, rotation, Vector3.one));
                    LowerBodyBones[i].SetVelocity(velocity);
                }
            }
            
            //Predict Upper Body
            {
                if(UpperBodyMode == UPPERBODY_MODE.Tracked) {
                    PredictTrackedUpperBody(TrackedUpperBodyNetwork);
                }
                if(UpperBodyMode == UPPERBODY_MODE.Untracked) {
                    PredictUntrackedUpperBody(UntrackedUpperBodyNetwork);
                }
            }

            if(RestoreAlignment) {
                Actor.RestoreAlignment();
            }

            //Postprocessing Block
            {
                if(UpperBodyMode == UPPERBODY_MODE.Tracked) {
                    Vector3 GetIKPosition(string name, TRANSFORM_MODE mode) {
                        Matrix4x4 local = TrackingSystem.Actor.GetBoneTransformation(name).TransformationFromTo(TrackingSystem.Root, Actor.GetRoot().GetWorldMatrix());
                        Matrix4x4 world = TrackingSystem.Actor.GetBoneTransformation(name).TransformationFrom(TargetControl.Transformations[TimeSeries.Pivot]);
                        world = Utility.Interpolate(local, world, IKBlend);
                        Matrix4x4 current = Actor.GetBoneTransformation(name);
                        if(mode == TRANSFORM_MODE.World) {
                            return Utility.Interpolate(Utility.Interpolate(world, local, HybridWeight), current, HybridWeight).GetPosition();
                        }
                        if(mode == TRANSFORM_MODE.Local) {
                            return Utility.Interpolate(local, current, HybridWeight).GetPosition();
                        }
                        return Vector3.zero;
                    }

                    Quaternion GetIKRotation(string name, TRANSFORM_MODE mode) {
                        Matrix4x4 local = TrackingSystem.Actor.GetBoneTransformation(name).TransformationFromTo(TrackingSystem.Root, Actor.GetRoot().GetWorldMatrix());
                        Matrix4x4 world = TrackingSystem.Actor.GetBoneTransformation(name).TransformationFrom(TargetControl.Transformations[TimeSeries.Pivot]);
                        world = Utility.Interpolate(local, world, IKBlend);
                        Matrix4x4 current = Actor.GetBoneTransformation(name);
                        if(mode == TRANSFORM_MODE.World) {
                            return Utility.Interpolate(Utility.Interpolate(world, local, HybridWeight), current, HybridWeight).GetRotation();
                        }
                        if(mode == TRANSFORM_MODE.Local) {
                            return Utility.Interpolate(local, current, HybridWeight).GetRotation();
                        }
                        return Quaternion.identity;
                    }

                    //Adjust Vertical Error
                    {
                        float currentHeight = Actor.FindBone(Blueman.HeadName).GetPosition().y;
                        float targetHeight = TrackingSystem.Actor.FindBone(Blueman.HeadName).GetPosition().y;

                        float jump = 0f;
                        for(int i=TrackingSystem.StyleSeries.Pivot; i<TrackingSystem.StyleSeries.SampleCount; i++) {
                            jump += TrackingSystem.StyleSeries.Values[i].Sum();
                        }
                        jump = Mathf.Clamp(jump, 0f, 1f);

                        float delta = Mathf.Lerp(0f, targetHeight-currentHeight, jump);
                        Actor.FindBone(Blueman.HipsName).GetTransform().position += new Vector3(0f, delta, 0f);
                    }

                    //Upper Body
                    if(SolverUpperBodyIK) {
                        //Head
                        {
                            UpperBodyIK.Solver.Objectives[0].Position = GetIKPosition(Blueman.LeftWristName, LeftPositionTransform);
                            UpperBodyIK.Solver.Objectives[0].Rotation = GetIKRotation(Blueman.LeftWristName, LeftRotationTransform);
                        }
                        //Left Wrist
                        {
                            UpperBodyIK.Solver.Objectives[1].Position = GetIKPosition(Blueman.RightWristName, RightPositionTransform);
                            UpperBodyIK.Solver.Objectives[1].Rotation = GetIKRotation(Blueman.RightWristName, RightRotationTransform);
                        }
                        //Right Wrist
                        {
                            UpperBodyIK.Solver.Objectives[2].Position = GetIKPosition(Blueman.HeadName, HeadPositionTransform);
                            UpperBodyIK.Solver.Objectives[2].Rotation = GetIKRotation(Blueman.HeadName, HeadRotationTransform);
                        }
                        UpperBodyIK.Solver.Solve();
                    }
                }

                //Lower Body
                if(SolveLowerBodyIK) {
                    void ComputeAnkleObjective(Solver.Objective objective, float baseline, float contact) {
                        Vector3 position = Vector3.Lerp(objective.Node.Transform.position, objective.Position, contact);
                        position.y = Mathf.Max(Mathf.Lerp(position.y, baseline, contact), baseline);
                        objective.Position = position;
                        Quaternion rotation = objective.Node.Transform.rotation;
                        objective.Rotation = rotation;
                    }
                    ComputeAnkleObjective(LowerBodyIK.Solver.Objectives[0], LeftAnkleBaseline, Current.Contacts[0]);
                    ComputeAnkleObjective(LowerBodyIK.Solver.Objectives[1], RightAnkleBaseline, Current.Contacts[2]);
                    void ComputeToeObjective(Solver.Objective objective, float baseline, float contact, Vector3 up) {
                        Vector3 position = Vector3.Lerp(objective.Node.Transform.position, objective.Position, contact);
                        position.y = Mathf.Max(Mathf.Lerp(position.y, baseline, contact), baseline);
                        objective.Position = position;
                        Quaternion toeRotation = objective.Node.Transform.rotation;
                        toeRotation = Quaternion.Slerp(toeRotation, Quaternion.FromToRotation(toeRotation.GetUp(), up) * toeRotation, contact);
                        objective.Rotation = toeRotation;
                    }
                    ComputeToeObjective(LowerBodyIK.Solver.Objectives[2], LeftToeBaseline, Current.Contacts[1], LeftToeUp);
                    ComputeToeObjective(LowerBodyIK.Solver.Objectives[3], RightToeBaseline, Current.Contacts[3], RightToeUp);
                    LowerBodyIK.Solver.Solve();
                }
            }

            // Rifle.Solve();

            //Update Time Series
            RootControl.Increment(0, TimeSeries.Pivot);
            AvatarControl.Increment(0, TimeSeries.Pivot);
            MotionHistory.Increment(0, TimeSeries.Pivot);
            foreach(MotionModule.Trajectory trajectory in MotionHistory.Trajectories) {
                trajectory.Transformations[TimeSeries.Pivot] = Actor.GetBoneTransformation(trajectory.Name);
                trajectory.Velocities[TimeSeries.Pivot] = Actor.GetBoneVelocity(trajectory.Name);
            }

            //Save Previous State
            PreviousRoot = Actor.GetRoot().GetWorldMatrix();
            PreviousPositions = Actor.GetBonePositions();
            PreviousRotations = Actor.GetBoneRotations();
            PreviousVelocities = Actor.GetBoneVelocities();

            //Increment Rollout Step
            int length = RolloutLength;

            //If close to target, smooth out motion
            if(Vector3.Distance(Actor.GetRoot().position, RootControl.Transformations.Last().GetPosition()) < 0.1f) {
                length = 10;
            }
            Step = (Step + 1) % length;
        }

        private List<State[]> PredictSequences(SentisNetwork network) {
            using(SentisNetwork.Input input = network.GetSession().GetInput("X")) {
                Matrix4x4 root = Actor.GetRoot().GetWorldMatrix();
                
                foreach(Actor.Bone bone in LowerBodyBones) {
                    Matrix4x4 m = bone.GetTransformation().TransformationTo(root);
                    Vector3 v = bone.GetVelocity().DirectionTo(root);
                    input.Feed(m.GetPosition());
                    input.Feed(m.GetForward());
                    input.Feed(m.GetUp());
                    input.Feed(v);
                }

                for(int i=TimeSeries.PivotKey+1; i<TimeSeries.KeyCount; i++) {
                    int index = TimeSeries.GetKey(i).Index;
                    input.FeedXZ(RootControl.Velocities[index].DirectionTo(root));
                    input.Feed(RootControl.AngularVelocities[index]);
                    input.Feed(AvatarControl.HipsTransformations[index].GetPosition().PositionTo(root));
                    input.Feed(AvatarControl.Centers[index].PositionTo(root));
                    input.Feed(AvatarControl.Orientations[index].DirectionTo(root));
                }
            }

            if(network.GetSession().HasInput("K")) {
                using(SentisNetwork.Input input = network.GetSession().GetInput("K")) {
                    input.SetDynamicTensorSize(KNN);
                    for(int i=0; i<KNN; i++) {
                        input.Feed(Noise * i.Ratio(0, KNN));
                    }
                }
            }
            
            network.RunSession();

            if(DrawCodebook) {
                using(SentisNetwork.Output output = network.GetSession().GetOutput("input.3")) {
                    Code = new float[CodebookChannels][];
                    for(int i=0; i<CodebookChannels; i++) {
                        Code[i] = output.Read(CodebookDim);
                    }
                }
            }

            using(SentisNetwork.Output output = network.GetSession().GetOutput("Y")) {
                List<State[]> sequences = new List<State[]>();
                for(int k=0; k<KNN; k++) {
                    State[] states = new State[SequenceLength+1];
                    for(int i=0; i<states.Length; i++) {
                        Matrix4x4 delta = output.ReadRootDelta();
                        float rootLock = output.Read(0f, 1f);
                        float[] contacts = output.Read(4, 0f, 1f);
                        string[] names = LowerBodyNames;
                        Vector3[] positions = new Vector3[LowerBodyBones.Length];
                        Quaternion[] rotations = new Quaternion[LowerBodyBones.Length];
                        Vector3[] velocities = new Vector3[LowerBodyBones.Length];
                        for(int b=0; b<LowerBodyBones.Length; b++) {
                            positions[b] = output.ReadVector3();
                            rotations[b] = output.ReadRotation3D();
                            velocities[b] = output.ReadVector3();
                        }
                        states[i] = new State(i==0 ? null : states[i-1], Actor.GetRoot().GetWorldMatrix(), delta, names, positions, rotations, velocities, rootLock, contacts);
                    }
                    sequences.Add(states);
                }
                return sequences;
            }
        }

        private void PredictTrackedUpperBody(SentisNetwork network) {
            for(int i=1; i<UpperBodyBones.Length; i++) {
                UpperBodyBones[i].SetTransformation(TargetControl.Transformations[TimeSeries.Pivot] * TrackingSystem.Actor.Bones[i].GetTransformation());
                UpperBodyBones[i].SetVelocity(TargetControl.Transformations[TimeSeries.Pivot] * TrackingSystem.Actor.Bones[i].GetVelocity());
            }

            using(SentisNetwork.Input input = network.GetSession().GetInput("X")) {
                Matrix4x4 root = PreviousRoot;
                for(int i=0; i<=TimeSeries.PivotKey; i++) {
                    int index = TimeSeries.GetKey(i).Index;
                    foreach(MotionModule.Trajectory trajectory in MotionHistory.Trajectories) {
                        Matrix4x4 m = trajectory.Transformations[index].TransformationTo(root);
                        Vector3 v = trajectory.Velocities[index].DirectionTo(root);
                        input.Feed(m.GetPosition());
                        input.Feed(m.GetForward());
                        input.Feed(m.GetUp());
                        input.Feed(v);
                    }
                }
                for(int i=0; i<=TimeSeries.PivotKey; i++) {
                    int index = TimeSeries.GetKey(i).Index;
                    foreach(MotionModule.Trajectory trajectory in TrackingSystem.TrackerHistory.Trajectories) {
                        Matrix4x4 m = trajectory.Transformations[index].TransformationTo(TrackingSystem.Root);
                        Vector3 v = trajectory.Velocities[index].DirectionTo(TrackingSystem.Root);
                        input.Feed(m.GetPosition());
                        input.Feed(m.GetForward());
                        input.Feed(m.GetUp());
                        input.Feed(v);
                    }
                }
            }

            network.RunSession();

            using(SentisNetwork.Output output = network.GetSession().GetOutput("Y")) {
                Matrix4x4 root = Actor.GetRoot().GetWorldMatrix();
                Vector3 hipPosition = output.ReadVector3().PositionFrom(root);
                Quaternion hipRotation = output.ReadRotation3D().RotationFrom(root);
                Vector3 hipVelocity = output.ReadVector3().DirectionFrom(root);
                hipPosition = Vector3.Lerp(hipPosition, PreviousPositions[0] + hipVelocity * Time.fixedDeltaTime, EulerIntegration ? 0.5f : 0f);
                Vector3 delta = Actor.Bones.First().GetPosition() - hipPosition;
                for(int i=1; i<UpperBodyBones.Length; i++) {
                    Vector3 p = output.ReadVector3().PositionFrom(root);
                    Quaternion r = output.ReadRotation3D().RotationFrom(root);
                    Vector3 v = output.ReadVector3().DirectionFrom(root);
                    p = Vector3.Lerp(p, PreviousPositions[UpperBodyBones[i].GetIndex()] + v * Time.fixedDeltaTime, EulerIntegration ? 0.5f : 0f) + delta;
                    UpperBodyBones[i].SetPosition(p);
                    UpperBodyBones[i].SetRotation(r);
                    UpperBodyBones[i].SetVelocity(v);
                }
            }
        }

        private void PredictUntrackedUpperBody(SentisNetwork network) {
            using(SentisNetwork.Input input = network.GetSession().GetInput("X")) {
                Matrix4x4 root = PreviousRoot;
                for(int i=0; i<=TimeSeries.PivotKey; i++) {
                    int index = TimeSeries.GetKey(i).Index;
                    foreach(MotionModule.Trajectory trajectory in MotionHistory.Trajectories) {
                        Matrix4x4 m = trajectory.Transformations[index].TransformationTo(root);
                        Vector3 v = trajectory.Velocities[index].DirectionTo(root);
                        input.Feed(m.GetPosition());
                        input.Feed(m.GetForward());
                        input.Feed(m.GetUp());
                        input.Feed(v);
                    }
                }
                foreach(Actor.Bone bone in Actor.FindBones(Blueman.TrackerNames)) {
                    input.Feed(PreviousPositions[bone.GetIndex()].PositionTo(root));
                    input.Feed(PreviousRotations[bone.GetIndex()].GetForward().DirectionTo(root));
                    input.Feed(PreviousRotations[bone.GetIndex()].GetUp().DirectionTo(root));
                    input.Feed(PreviousVelocities[bone.GetIndex()].DirectionTo(root));
                }
            }

            network.RunSession();

            using(SentisNetwork.Output output = network.GetSession().GetOutput("Y")) {
                Matrix4x4 root = Actor.GetRoot().GetWorldMatrix();
                Vector3 hipPosition = output.ReadVector3().PositionFrom(root);
                Quaternion hipRotation = output.ReadRotation3D().RotationFrom(root);
                Vector3 hipVelocity = output.ReadVector3().DirectionFrom(root);
                hipPosition = Vector3.Lerp(hipPosition, PreviousPositions[0] + hipVelocity * Time.fixedDeltaTime, EulerIntegration ? 0.5f : 0f);
                Vector3 delta = Actor.Bones.First().GetPosition() - hipPosition;
                for(int i=1; i<UpperBodyBones.Length; i++) {
                    Vector3 p = output.ReadVector3().PositionFrom(root);
                    Quaternion r = output.ReadRotation3D().RotationFrom(root);
                    Vector3 v = output.ReadVector3().DirectionFrom(root);
                    p = Vector3.Lerp(p, PreviousPositions[UpperBodyBones[i].GetIndex()] + v * Time.fixedDeltaTime, EulerIntegration ? 0.5f : 0f) + delta;
                    UpperBodyBones[i].SetPosition(p);
                    UpperBodyBones[i].SetRotation(r);
                    UpperBodyBones[i].SetVelocity(v);
                }
            }
        }

        public void UpdateTargetControl(Camera camera) {
            HybridSteps = Mathf.Max(HybridSteps-1, 0);
            if(TrackingSystem.IsConnected()) {
                Vector2 leftAxis = TrackingSystem.LeftController.GetJoystickAxis();
                // Vector2 rightAxis = TrackingSystem.RightController.GetJoystickAxis();
                // if(leftAxis.magnitude > 0f || rightAxis.magnitude > 0f) {
                //     HybridSteps = 2*TimeSeries.FutureSamples;
                // }
                if(leftAxis.magnitude > 0f) {
                    HybridSteps = 2*TimeSeries.FutureSamples;
                }

                Vector3 move = Vector3.zero;
                if(JoystickSpace == TRANSFORM_MODE.Local) {
                    move = Quaternion.LookRotation(camera.transform.forward.ZeroY().normalized, Vector3.up) * new Vector3(leftAxis.x, 0f, leftAxis.y) * MoveSpeed * Time.fixedDeltaTime;
                } else {
                    move = transform.rotation * new Vector3(leftAxis.x, 0f, leftAxis.y) * MoveSpeed * Time.fixedDeltaTime;
                }
                // Quaternion rotate = rightAxis.magnitude > 0.5f ? Quaternion.LookRotation(new Vector3(rightAxis.x, 0f, rightAxis.y), Vector3.up).RotationTo(JoystickControl.Transformations.Last().GetRotation()) : Quaternion.identity;
                Quaternion rotate = Quaternion.identity;

                Vector3 position = TargetControl.Transformations.Last().GetPosition() + move;
                Quaternion rotation = TargetControl.Transformations.Last().GetRotation() * rotate;
                Matrix4x4 target = Matrix4x4.TRS(position, rotation, Vector3.one);

                TargetControl.Increment(0, TimeSeries.SampleCount-1);
                TargetControl.Transformations[TimeSeries.SampleCount-1] = target;
            } else {
                //Get Mouse Input
                bool blocked = EventSystem.current == null ? false : EventSystem.current.IsPointerOverGameObject();
                Vector3 pointer = Utility.GetMousePosition(SelectionMask).SetY(0f);
                if(RestrictArea) {
                    Vector3 direction = pointer - Actor.GetRoot().position;
                    pointer = Actor.GetRoot().position + Mathf.Clamp(direction.magnitude, 0f, ControlRadius) * direction.normalized;
                }
                if(RestrictBounds) {
                    pointer = new Vector3(
                        Mathf.Clamp(pointer.x, BoundingBox.x, BoundingBox.y), 
                        pointer.y, 
                        Mathf.Clamp(pointer.z, BoundingBox.x, BoundingBox.y)
                    );
                }

                //Compute Target
                Vector3 position = TargetControl.Transformations.Last().GetPosition();
                Quaternion rotation = TargetControl.Transformations.Last().GetRotation();
                if(!blocked) {
                    if(Input.GetMouseButton(0) && Input.GetMouseButton(1)) {
                        float ratio = Mathf.Clamp(Vector3.Distance(Actor.GetRoot().position, pointer), 0f, RotationRadius).Normalize(0f, RotationRadius, 0f, 1f);
                        rotation = Quaternion.Slerp(
                            rotation,
                            Quaternion.LookRotation(pointer - Actor.GetRoot().position, Vector3.up) * TrackingSystem.RootSeries.Transformations.Last().GetRotation().GetInverse(),
                            Mathf.Pow(ratio, 2f)
                        );
                        HybridSteps = 2*TimeSeries.FutureSamples;
                    }
                    if(Input.GetMouseButton(0)) {
                        position = pointer - rotation * TrackingSystem.RootSeries.Transformations.Last().GetPosition();
                        HybridSteps = 2*TimeSeries.FutureSamples;
                    }
                }

                //Update Trajectory
                TargetControl.Increment(0, TimeSeries.SampleCount-1);
                TargetControl.SetTransformation(TimeSeries.SampleCount-1, Matrix4x4.TRS(position, rotation, Vector3.one), TargetBlend);
            }
        }

        public RootModule.Series GetRootControl() {
            return RootControl;
        }

        void OnGUI() {
            if(Screen != null && Screen != Camera.current) {
                return;
            }

            if(DrawLabels) {
                if(DrawCodebook) {
                    UltiDraw.Begin();
                    UltiDraw.OnGUILabel(CodebookRect.GetCenter() + new Vector2(0f, LabelShift), CodebookRect.GetSize(), LabelSize, "Codebook Vector", UltiDraw.Black, TextAnchor.UpperCenter);
                    UltiDraw.End();
                }

                if(DrawSimilarityMap) {
                    UltiDraw.Begin();
                    UltiDraw.OnGUILabel(CategoricalRect.GetCenter() + new Vector2(0f, LabelShift), CategoricalRect.GetSize(), LabelSize, "Categorical Sample", UltiDraw.Black, TextAnchor.UpperCenter);
                    UltiDraw.End();
                }
            }
        }

        void OnRenderObject() {
            if(Screen != null && Screen != Camera.current) {
                return;
            }

            if(DrawRootControl) {
                RootControl.Draw(
                    DrawControlHistory ? 0 : TimeSeries.PivotKey, 
                    TimeSeries.KeyCount,
                    PositionColor, 
                    DirectionColor,
                    VelocityColor,
                    TrajectoryScale,
                    drawConnections:true,
                    drawPositions:true,
                    drawDirections:true,
                    drawVelocities:true,
                    drawAngularVelocities:false,
                    drawLocks:false
                );
            }
            if(DrawAvatarControl) {
                AvatarControl.Draw(
                    DrawControlHistory ? 0 : TimeSeries.PivotKey+1, 
                    TimeSeries.KeyCount
                );
            }
            if(DrawWireCircles) {
                UltiDraw.Begin();
                UltiDraw.SetDepthRendering(true);
                UltiDraw.DrawWireCircle(Actor.GetRoot().position + new Vector3(0f, 0.001f, 0f), Quaternion.Euler(90f, 0f, 0f), 2f*ControlRadius, UltiDraw.Black);
                UltiDraw.DrawWireCircle(Actor.GetRoot().position + new Vector3(0f, 0.001f, 0f), Quaternion.Euler(90f, 0f, 0f), 2f*RotationRadius, UltiDraw.Red);
                UltiDraw.SetDepthRendering(false);
                UltiDraw.End();
            }
            if(DrawTarget) {
                UltiDraw.Begin();
                Matrix4x4 m = RootControl.Transformations.Last();
                UltiDraw.DrawLine(m.GetPosition(), m.GetPosition() + TargetScale*0.3f*m.GetForward(), Vector3.up, TargetScale*0.2f, 0f, TargetArrow.Opacity(HybridWeight));
                UltiDraw.DrawCube(m.GetPosition(), m.GetRotation(), TargetScale*0.1f, TargetCube.Opacity(HybridWeight));
                UltiDraw.End();
            }

            if(Current == null) {return;}

            if(DrawCurrentSequence) {
                if(Sequences.Count == 1) {
                    Current.DrawSequence(Actor, LowerBodyNames, SequenceColor, SequenceOpacity);
                } else {
                    for(int i=0; i<Sequences.Count; i++) {
                        for(int j=0; j<Sequences[i].Length; j++) {
                            if(Current == Sequences[i][j]) {
                                Color color = UltiDraw.GetRainbowColor(i, Sequences.Count);
                                Current.DrawSequence(Actor, LowerBodyNames, color, SequenceOpacity);
                                break;
                            }
                        }
                    }
                }
            }

            if(DrawAllSequences) {
                for(int i=0; i<Sequences.Count; i++) {
                    State.DrawSequence(Sequences[i], Actor, LowerBodyNames, UltiDraw.GetRainbowColor(i, Sequences.Count), SequenceOpacity);
                }
            }

            if(DrawStateVariables) {
                UltiDraw.Begin();
                UltiDraw.PlotHorizontalBar(new Vector2(DrawArea.X, DrawArea.Y + 0.25f*DrawArea.H), new Vector2(DrawArea.W, 0.5f*DrawArea.H), Current.RootLock, fillColor:UltiDraw.DarkRed);
                UltiDraw.PlotBars(new Vector2(DrawArea.X, DrawArea.Y - 0.25f*DrawArea.H), new Vector2(DrawArea.W, 0.5f*DrawArea.H), Current.Contacts, yMin:0f, yMax:1f, barColor:UltiDraw.DarkGreen);
                UltiDraw.PlotHorizontalBar(new Vector2(DrawArea.X, DrawArea.Y - DrawArea.H), new Vector2(DrawArea.W, 1f*DrawArea.H), HybridWeight, fillColor:UltiDraw.DarkRed);
                UltiDraw.End();
            }

            if(DrawHistory) {
                MotionHistory.Draw(
                    UltiDraw.Green.Opacity(0.5f),
                    0,
                    TimeSeries.PivotKey+1,
                    1f
                );
            }

            if(DrawInferenceTime) {
                LowerBodyNetwork.GetSession().DrawTimeHistory(new Vector2(0.5f, 0.1f), new Vector2(0.2f, 0.1f), 0.01f);
                TrackedUpperBodyNetwork.GetSession().DrawTimeHistory(new Vector2(0.5f, 0.2f), new Vector2(0.2f, 0.1f), 0.01f);
                UntrackedUpperBodyNetwork.GetSession().DrawTimeHistory(new Vector2(0.5f, 0.3f), new Vector2(0.2f, 0.1f), 0.01f);
            }

            if(DrawInverseKinematics) {
                LowerBodyIK.Solver.Draw();
                UpperBodyIK.Solver.Draw();
            }

            if(DrawCodebook) {
                if(Code != null) {
                    UltiDraw.Begin();
                    UltiDraw.PlotBars(CodebookRect.GetCenter(), CodebookRect.GetSize(), Code, 0f, 1f);
                    UltiDraw.End();
                }
            }

            if(DrawSimilarityMap) {
                int knn = Sequences.Count;
                int horizon = Sequences.First().Length;
                float[][] values = new float[knn][];
                float[][] similarity = new float[knn][];
                float min = float.MaxValue;
                float max = 0f;
                for(int i=0; i<knn; i++) {
                    values[i] = new float[horizon];
                    similarity[i] = new float[horizon];
                    for(int j=0; j<horizon; j++) {
                        values[i][j] = 1f;
                        similarity[i][j] = Sequences[i][j].Distance;
                        min = Mathf.Min(min, similarity[i][j]);
                        max = Mathf.Max(max, similarity[i][j]);
                    }
                }
                for(int i=0; i<knn; i++) {
                    for(int j=0; j<horizon; j++) {
                        similarity[i][j] = similarity[i][j].Normalize(min, max, 1f, 0f);
                    }
                }
                UltiDraw.Begin();
                Color[][] colors = new Color[knn][];
                for(int i=0; i<knn; i++) {
                    colors[i] = new Color[horizon];
                    for(int j=0; j<horizon; j++) {
                        if(Sequences[i][j] == Current) {
                            for(int k=0; k<j; k++) {
                                colors[i][k] = UltiDraw.GetRainbowColor(i, knn).Opacity(similarity[i][k]);
                            }
                            break;
                        } else {
                            colors[i][j] = UltiDraw.Black.Opacity(similarity[i][j]);
                        }
                    }
                }
                UltiDraw.PlotBars(CategoricalRect.GetCenter(), CategoricalRect.GetSize(), values, yMin:0f, yMax:1f, barColors:colors);
                UltiDraw.End();
            }
        }
    }
}