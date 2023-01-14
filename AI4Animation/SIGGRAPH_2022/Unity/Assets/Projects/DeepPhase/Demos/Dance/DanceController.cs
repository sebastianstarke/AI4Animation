using UnityEngine;
using AI4Animation;
using UltimateIK;
using Unity.Barracuda;

namespace DeepPhase {
    public class DanceController : MonoBehaviour {

        public float Framerate = 60f;
        public ONNXNetwork NeuralNetwork;
        public KeyCode NoiseSampler = KeyCode.N;

        public bool DrawGUI = true;
        public bool DrawDebug = true;
        public bool DrawAudio = false;
        public bool DrawNoise = false;

        public int Channels = 10;
        public int NoiseSamples = 8;
        [Range(0f,5f)] public float NoiseStrength = 0f;
        [Range(0f,5f)] public float NoiseReset = 1f;
        [Range(0f,1f)] public float MinAmplitude = 0f;

        public float FootSafetyDistance = 0f;

        [Range(0f, 1f)] public float PhaseStability = 0.5f;

        public bool Postprocessing = true;
        public float ContactPower = 3f;
        public float ContactThreshold = 2f/3f;

        private Actor Actor = null;
        
        private TimeSeries TimeSeries;
        private TimeSeries MusicSeries;

        private RootModule.Series RootSeries;
        private ContactModule.Series ContactSeries;
        private DeepPhaseModule.Series PhaseSeries;
        private AudioSpectrumModule.Series SpectrumSeries;

        private IK LeftFootIK;
        private IK RightFootIK;

        private float[] NoiseA = null;
        private float[] NoiseB = null;
        private float[] Noise = null;
        private float NoiseTimer = 1f;

        public RootModule.Series GetRootSeries() {
            return RootSeries;
        }

        void Start() {
            Actor = GetComponent<Actor>();

            TimeSeries = new TimeSeries(6, 6, 1f, 1f, 10);
            MusicSeries = new TimeSeries(20, 20, 1f, 1f, 3);

            RootSeries = new RootModule.Series(TimeSeries, transform);
            ContactSeries = new ContactModule.Series(TimeSeries, "Left Foot", "Right Foot");
            PhaseSeries = new DeepPhaseModule.Series(TimeSeries, Channels);
            SpectrumSeries = new AudioSpectrumModule.Series(MusicSeries);

            LeftFootIK = IK.Create(Actor.FindTransform("LeftUpLeg"), Actor.GetBoneTransforms("LeftToeBase"));
            RightFootIK = IK.Create(Actor.FindTransform("RightUpLeg"), Actor.GetBoneTransforms("RightToeBase"));

            NoiseA = new float[NoiseSamples];
            NoiseB = new float[NoiseSamples];
            Noise = new float[NoiseSamples];

            RootSeries.DrawGUI = DrawGUI;
            ContactSeries.DrawGUI = DrawGUI;
            PhaseSeries.DrawGUI = DrawGUI;
            SpectrumSeries.DrawGUI = DrawGUI;
            RootSeries.DrawScene = DrawDebug;
            ContactSeries.DrawScene = DrawDebug;
            PhaseSeries.DrawScene = DrawDebug;
            SpectrumSeries.DrawScene = DrawDebug;

            NeuralNetwork.CreateSession();
        }

		void OnDestroy() {
            NeuralNetwork.CloseSession();
		}

        string GetTag() {
            return Channels + "Channels";
        }

        public void InitializeDance(params object[] objects) {
            MotionAsset asset = (MotionAsset)objects[0];
            float timestamp = (float)objects[1];
            bool mirrored = (bool)objects[2];
            Matrix4x4 reference = asset.GetModule<RootModule>().GetRootTransformation(timestamp, mirrored);
            Matrix4x4 root = reference;
            Actor.GetRoot().transform.position = reference.GetPosition();
            Actor.GetRoot().transform.rotation = reference.GetRotation();
            Actor.SetBoneTransformations(asset.GetFrame(timestamp).GetBoneTransformations(Actor.GetBoneNames(), mirrored).TransformationsFromTo(reference, root, true));
            Actor.SetBoneVelocities(asset.GetFrame(timestamp).GetBoneVelocities(Actor.GetBoneNames(), mirrored).DirectionsFromTo(reference, root, true));
            RootSeries = asset.GetModule<RootModule>().ExtractSeries(TimeSeries, timestamp, mirrored) as RootModule.Series;
            RootSeries.TransformFromTo(reference, root);
            
            ContactSeries = asset.GetModule<ContactModule>().ExtractSeries(TimeSeries, timestamp, mirrored) as ContactModule.Series;
            PhaseSeries = asset.GetModule<DeepPhaseModule>(GetTag()).ExtractSeries(TimeSeries, timestamp, mirrored) as DeepPhaseModule.Series;
            SpectrumSeries = asset.GetModule<AudioSpectrumModule>().ExtractSeries(MusicSeries, timestamp, mirrored) as AudioSpectrumModule.Series;
            foreach(Objective o in LeftFootIK.Objectives) {
                o.TargetPosition = LeftFootIK.Joints[o.Joint].Transform.position;
                o.TargetRotation = LeftFootIK.Joints[o.Joint].Transform.rotation;
            }
            foreach(Objective o in RightFootIK.Objectives) {
                o.TargetPosition = RightFootIK.Joints[o.Joint].Transform.position;
                o.TargetRotation = RightFootIK.Joints[o.Joint].Transform.rotation;
            }
        }

        public void AnimateDance(object[] parameters) {
            MusicControl(parameters);
            Feed();
            NeuralNetwork.RunSession();
            Read();
        }

        private void HandleNoise() {
            //Sample Noise
            if(Input.GetKeyDown(NoiseSampler)) {
                for(int i=0; i<NoiseSamples; i++) {
                    NoiseA[i] = Utility.GaussianValue(0f, NoiseStrength);
                    NoiseB[i] = NoiseA[i];
                }
                NoiseTimer = NoiseReset;
            }
            if(NoiseTimer <= 0f) {
                for(int i=0; i<NoiseSamples; i++) {
                    NoiseA[i] = NoiseB[i];
                    NoiseB[i] = Utility.GaussianValue(0f, NoiseStrength);
                }
                NoiseTimer = NoiseReset;
            }
            NoiseTimer -= 1f/Framerate;

            float lerp = GetLerp();
            for(int i=0; i<NoiseSamples; i++) {
                Noise[i] = Mathf.Lerp(NoiseA[i], NoiseB[i], lerp);
            }
        }

        private float GetLerp() {
            float min = 0f;
            float max = NoiseReset;
            float lerp = Mathf.Clamp(NoiseTimer, min, max).Normalize(min, max, 1f, 0f);
            return lerp;
        }

        private void MusicControl(object[] parameters) {
            AudioSpectrum spectrum = (AudioSpectrum)parameters[0];
            float timestamp = (float)parameters[1];
            float pitch = (float)parameters[2];

            //Set Music
            SpectrumSeries.Increment(0, TimeSeries.Pivot);
            for(int i=0; i<SpectrumSeries.Samples.Length; i++) {
                if(SpectrumSeries.Values[i] == null) {
                    SpectrumSeries.Values[i] = spectrum.GetFiltered(timestamp + pitch*SpectrumSeries.Samples[i].Timestamp, MusicSeries.MaximumFrequency);
                }
            }
            for(int i=SpectrumSeries.Pivot; i<SpectrumSeries.Samples.Length; i++) {
                SpectrumSeries.Values[i] = spectrum.GetFiltered(timestamp + pitch*SpectrumSeries.Samples[i].Timestamp, MusicSeries.MaximumFrequency);
            }
            
            //Noise Sampler
            HandleNoise();
        }

        private void Feed() {
            //Get Root
            Matrix4x4 prev = RootSeries.Transformations[RootSeries.Pivot-1];
            Matrix4x4 root = Actor.GetRoot().GetWorldMatrix();

            //Input Timeseries
            for(int i=0; i<TimeSeries.KeyCount; i++) {
                int index = TimeSeries.GetKey(i).Index;
                NeuralNetwork.FeedXZ(RootSeries.GetPosition(index).PositionTo(prev));
                NeuralNetwork.FeedXZ(RootSeries.GetDirection(index).DirectionTo(prev));
                NeuralNetwork.FeedXZ(RootSeries.Velocities[index].DirectionTo(prev));
            }

            //Input Audio
            for(int i=0; i<MusicSeries.KeyCount; i++) {
                int index = MusicSeries.GetKey(i).Index;
                NeuralNetwork.Feed(SpectrumSeries.Values[index].Spectogram);
                NeuralNetwork.Feed(SpectrumSeries.Values[index].Beats);
                NeuralNetwork.Feed(SpectrumSeries.Values[index].Flux);
                NeuralNetwork.Feed(SpectrumSeries.Values[index].MFCC);
                NeuralNetwork.Feed(SpectrumSeries.Values[index].Chroma);
                NeuralNetwork.Feed(SpectrumSeries.Values[index].ZeroCrossing);
            }

            //Input Character
            for(int i=0; i<Actor.Bones.Length; i++) {
                NeuralNetwork.Feed(Actor.Bones[i].GetTransform().position.PositionTo(root));
                NeuralNetwork.Feed(Actor.Bones[i].GetTransform().forward.DirectionTo(root));
                NeuralNetwork.Feed(Actor.Bones[i].GetTransform().up.DirectionTo(root));
                NeuralNetwork.Feed(Actor.Bones[i].GetVelocity().DirectionTo(root));
            }

            //Input Gating Features
            NeuralNetwork.Feed(PhaseSeries.GetAlignment());

            //Feed Noise
            NeuralNetwork.Feed(Noise);
        }

        private void Read() {
            //Update Past States
            RootSeries.Increment(0, TimeSeries.Pivot);
            ContactSeries.Increment(0, TimeSeries.Pivot);
            PhaseSeries.Increment(0, TimeSeries.Pivot);

            //Update Root State
            Vector3 offset = NeuralNetwork.ReadVector3();

            Matrix4x4 reference = Actor.GetRoot().GetWorldMatrix();
            Matrix4x4 root = reference * Matrix4x4.TRS(new Vector3(offset.x, 0f, offset.z), Quaternion.AngleAxis(offset.y, Vector3.up), Vector3.one);
            RootSeries.Transformations[TimeSeries.Pivot] = root;
            RootSeries.Velocities[TimeSeries.Pivot] = NeuralNetwork.ReadXZ().DirectionFrom(root);

            //Read Future States
            for(int i=TimeSeries.PivotKey+1; i<TimeSeries.KeyCount; i++) {
                int index = TimeSeries.GetKey(i).Index;
                RootSeries.Transformations[index] = Utility.Interpolate(
                    RootSeries.Transformations[index],
                    Matrix4x4.TRS(NeuralNetwork.ReadXZ().PositionFrom(reference), Quaternion.LookRotation(NeuralNetwork.ReadXZ().DirectionFrom(reference).normalized, Vector3.up), Vector3.one),
                    1f,
                    1f
                );
                RootSeries.Velocities[index] = NeuralNetwork.ReadXZ().DirectionFrom(reference);
            }

            //Read Posture
            Vector3[] positions = new Vector3[Actor.Bones.Length];
            Vector3[] forwards = new Vector3[Actor.Bones.Length];
            Vector3[] upwards = new Vector3[Actor.Bones.Length];
            Vector3[] velocities = new Vector3[Actor.Bones.Length];
            for(int i=0; i<Actor.Bones.Length; i++) {
                Vector3 position = NeuralNetwork.ReadVector3().PositionFrom(root);
                Vector3 forward = forward = NeuralNetwork.ReadVector3().normalized.DirectionFrom(root);
                Vector3 upward = NeuralNetwork.ReadVector3().normalized.DirectionFrom(root);
                Vector3 velocity = NeuralNetwork.ReadVector3().DirectionFrom(root);
                velocities[i] = velocity;
                positions[i] = Vector3.Lerp(Actor.Bones[i].GetTransform().position + velocity / Framerate, position, 0.5f);
                forwards[i] = forward;
                upwards[i] = upward;
            }

            //Update Contacts
            float[] contacts = NeuralNetwork.Read(ContactSeries.Bones.Length, 0f, 1f);
            for(int i=0; i<ContactSeries.Bones.Length; i++) {
                ContactSeries.Values[TimeSeries.Pivot][i] = contacts[i].SmoothStep(ContactPower, ContactThreshold);
            }

            //Update Phases
            PhaseSeries.UpdateAlignment(NeuralNetwork.Read((1+PhaseSeries.FutureKeys) * PhaseSeries.Channels * 4), PhaseStability, 1f/Framerate, MinAmplitude);

            //Interpolate Timeseries
            RootSeries.Interpolate(TimeSeries.Pivot, TimeSeries.Samples.Length);
            PhaseSeries.Interpolate(TimeSeries.Pivot, TimeSeries.Samples.Length);

            //Assign Posture
            transform.position = RootSeries.GetPosition(TimeSeries.Pivot);
            transform.rotation = RootSeries.GetRotation(TimeSeries.Pivot);
            for(int i=0; i<Actor.Bones.Length; i++) {
                Actor.Bones[i].SetVelocity(velocities[i]);
                Actor.Bones[i].SetPosition(positions[i]);
                Actor.Bones[i].SetRotation(Quaternion.LookRotation(forwards[i], upwards[i]));
            }

            //Correct Twist
            Actor.RestoreAlignment();

            //Process Contact States
            ProcessFootIK(LeftFootIK, ContactSeries.Values[TimeSeries.Pivot][0]);
            ProcessFootIK(RightFootIK, ContactSeries.Values[TimeSeries.Pivot][1]);

            PhaseSeries.AddGatingHistory(NeuralNetwork.GetOutput("W").AsFloats());
        }

        private void ProcessFootIK(IK ik, float contact) {
            if(!Postprocessing) {
                return;
            }
            ik.Activation = UltimateIK.ACTIVATION.Linear;
            for(int i=0; i<ik.Objectives.Length; i++) {
                Vector3 self = ik.Joints[ik.Objectives[i].Joint].Transform.position;
                Vector3 other = ik == LeftFootIK ? RightFootIK.Joints.Last().Transform.position : LeftFootIK.Joints.Last().Transform.position;
                // Vector3 target = other + Mathf.Max(FootSafetyDistance, Vector3.Distance(self, other)) * (self-other).normalized;
                Vector3 target = Vector3.Lerp(self, other + Mathf.Max(FootSafetyDistance, Vector3.Distance(self, other)) * (self-other).normalized, 1f-contact);
                ik.Objectives[i].SetTarget(Vector3.Lerp(ik.Objectives[i].TargetPosition, target, 1f-contact));
                ik.Objectives[i].SetTarget(ik.Joints[ik.Objectives[i].Joint].Transform.rotation);
            }
            ik.Iterations = 25;
            ik.Solve();
        }

        void OnGUI() {
            RootSeries.DrawGUI = DrawGUI;
            RootSeries.GUI();

            // ContactSeries.DrawGUI = DrawGUI;
            // ContactSeries.GUI();

            PhaseSeries.DrawGUI = DrawGUI;
            PhaseSeries.GUI();

            if(DrawAudio) {
                SpectrumSeries.Draw();
            }
        }

        void OnRenderObject() {
            RootSeries.DrawScene = DrawDebug;
            RootSeries.Draw();

            // ContactSeries.DrawScene = DrawDebug;
            // ContactSeries.Draw();

            PhaseSeries.DrawScene = DrawDebug;
            PhaseSeries.Draw();

            // UltiDraw.Begin();
            // List<float> values = new List<float>();
            // for(int i=PhaseSeries.Pivot; i<PhaseSeries.Samples.Length; i++) {
            //     float ratio = i.Ratio(PhaseSeries.Pivot-1, PhaseSeries.Samples.Length-1);
            //     float blend = ratio.SmoothStep(2f, 1f-TransitionBlend);
            //     values.Add(blend);
            // }
            // UltiDraw.PlotFunction(new Vector2(0.5f, 0.5f), new Vector2(0.75f, 0.25f), values.ToArray(), 0f, 1f);
            // UltiDraw.End();

            if(DrawNoise) {
                UltiDraw.Begin();
                UltiDraw.PlotBars(new Vector2(0.5f, 0.1f), new Vector2(0.5f, 0.1f), Noise);
                UltiDraw.End();
            }

            if(DrawAudio) {
                SpectrumSeries.Draw();
            }
        }

    }
}