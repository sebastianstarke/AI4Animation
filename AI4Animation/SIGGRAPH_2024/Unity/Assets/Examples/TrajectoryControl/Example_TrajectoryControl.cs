using UnityEngine;

public class TrajectoryControlExample : MonoBehaviour {

    public RootSeries Trajectory;

    public Transform Target;

    void Start() {
        TimeSeries timeSeries = new TimeSeries(0, 5, 0f, 0.5f, 1);
        // TimeSeries timeSeries = new TimeSeries(5, 5, 0.5f, 0.5f, 1); //Smoother Blend
        Trajectory = new RootSeries(timeSeries);
    }

    void Update() {
        Trajectory.Generate(transform.GetWorldMatrix(), Target.GetWorldMatrix());
        // Trajectory.Control(Target.position, Target.forward, 1f, 0.5f, 0.5f, 0.5f); //Smoother Blend
    }

    void OnRenderObject() {
        Trajectory.Draw();
    }

    public class RootSeries : TimeSeries.Component {

        public Matrix4x4[] Deltas;
        public Matrix4x4[] Transformations;
        public Vector3[] Velocities;
        public float[] AngularVelocities;
        public float[] Locks;

        public RootSeries(TimeSeries global) : base(global) {
            Deltas = new Matrix4x4[Samples.Length];
            Transformations = new Matrix4x4[Samples.Length];
            Velocities = new Vector3[Samples.Length];
            AngularVelocities = new float[Samples.Length];
            Locks = new float[Samples.Length];
            for(int i=0; i<Samples.Length; i++) {
                Deltas[i] = Matrix4x4.identity;
                Transformations[i] = Matrix4x4.identity;
                Velocities[i] = Vector3.zero;
                AngularVelocities[i] = 0f;
                Locks[i] = 0f;
            }
        }

        public void SetTransformation(int index, Matrix4x4 transformation) {
            Transformations[index] = transformation;
        }

        public void SetTransformation(int index, Matrix4x4 transformation, float weight) {
            Transformations[index] = Utility.Interpolate(Transformations[index], transformation, weight);
        }

        public Matrix4x4 GetTransformation(int index) {
            return Transformations[index];
        }

        public void SetPosition(int index, Vector3 value) {
            Matrix4x4Extensions.SetPosition(ref Transformations[index], value);
        }

        public Vector3 GetPosition(int index) {
            return Transformations[index].GetPosition();
        }

        public void SetRotation(int index, Quaternion value) {
            Matrix4x4Extensions.SetRotation(ref Transformations[index], value);
        }

        public Quaternion GetRotation(int index) {
            return Transformations[index].GetRotation();
        }

        public void SetDirection(int index, Vector3 value) {
            Matrix4x4Extensions.SetRotation(ref Transformations[index], Quaternion.LookRotation(value == Vector3.zero ? Vector3.forward : value, Vector3.up));
        }

        public Vector3 GetDirection(int index) {
            return Transformations[index].GetForward();
        }

        public void SetVelocity(int index, Vector3 value) {
            Velocities[index] = value;
        }

        public void SetVelocity(int index, Vector3 value, float weight) {
            Velocities[index] = Vector3.Lerp(Velocities[index], value, weight);
        }

        public Vector3 GetVelocity(int index) {
            return Velocities[index];
        }

        public void TransformFrom(Matrix4x4 space) {
            Transformations.TransformationsFrom(space, true);
            Velocities.DirectionsFrom(space, true);
        }

        public void TransformTo(Matrix4x4 space) {
            Transformations.TransformationsTo(space, true);
            Velocities.DirectionsTo(space, true);
        }

        public void TransformFromTo(Matrix4x4 from, Matrix4x4 to) {
            Transformations.TransformationsFromTo(from, to, true);
            Velocities.DirectionsFromTo(from, to, true);
        }

        public void TransformFromTo(int index, Matrix4x4 from, Matrix4x4 to) {
            Transformations[index] = Transformations[index].TransformationFromTo(from, to);
            Velocities[index] = Velocities[index].DirectionFromTo(from, to);
        }

        public void Generate(Matrix4x4 sourceTransformation, Matrix4x4 targetTransformation) {
            for(int i=Pivot; i<Samples.Length; i++) {
                float weight = i.Ratio(Pivot, Samples.Length-1);
                Transformations[i] = Utility.Interpolate(sourceTransformation, targetTransformation, weight);
            }
        }

        private Vector3[] CopyPositions;
        private Quaternion[] CopyRotations;
        private Vector3[] CopyVelocities;
        public void Control(Vector3 move, Vector3 face, float weight, float positionBias=1f, float directionBias=1f, float velocityBias=1f) {
            float ControlWeight(float x, float weight) {
                return x.SmoothStep(2f, 1f-weight);
            }
            CopyPositions = new Vector3[Samples.Length];
            CopyRotations = new Quaternion[Samples.Length];
            CopyVelocities = new Vector3[Samples.Length];
            for(int i=0; i<Samples.Length; i++) {
                CopyPositions[i] = GetPosition(i);
                CopyRotations[i] = GetRotation(i);
                CopyVelocities[i] = GetVelocity(i);
            }
            for(int i=Pivot; i<Samples.Length; i++) {
                float ratio = i.Ratio(Pivot-1, Samples.Length-1);
                //Root Positions
                CopyPositions[i] = CopyPositions[i-1] +
                    Vector3.Lerp(
                        GetPosition(i) - GetPosition(i-1),
                        1f/FutureSamples * move,
                        weight * ControlWeight(ratio, positionBias)
                    );

                //Root Rotations
                CopyRotations[i] = CopyRotations[i-1] *
                    Quaternion.Slerp(
                        GetRotation(i).RotationTo(GetRotation(i-1)),
                        face != Vector3.zero ? Quaternion.LookRotation(face, Vector3.up).RotationTo(CopyRotations[i-1]) : Quaternion.identity,
                        weight * ControlWeight(ratio, directionBias)
                    );

                //Root Velocities
                CopyVelocities[i] = CopyVelocities[i-1] +
                    Vector3.Lerp(
                        GetVelocity(i) - GetVelocity(i-1),
                        move-CopyVelocities[i-1],
                        weight * ControlWeight(ratio, velocityBias)
                    );
            }
            for(int i=0; i<Samples.Length; i++) {
                SetPosition(i, CopyPositions[i]);
                SetRotation(i, CopyRotations[i]);
                SetVelocity(i, CopyVelocities[i]);
            }
        }

        public override void GUI() {
            
        }

        public override void Draw() {
            if(DrawScene) {
                Draw(0, KeyCount);
            }
        }

        public void Draw(int start, int end) {
            Draw(start, end, UltiDraw.Black, UltiDraw.Orange.Opacity(0.75f), UltiDraw.Green.Opacity(0.25f), 2f);
        }
        
        public void Draw(
            int start, 
            int end, 
            Color positionColor, 
            Color directionColor, 
            Color velocityColor, 
            float thickness=1f,
            bool drawConnections=true,
            bool drawPositions=true,
            bool drawDirections=true,
            bool drawVelocities=true,
            bool drawAngularVelocities=true,
            bool drawLocks=true
        ) {
            UltiDraw.Begin();

            //Connections
            if(drawConnections) {
                for(int i=start; i<end-1; i++) {
                    int current = GetKey(i).Index;
                    int next = GetKey(i+1).Index;
                    UltiDraw.DrawLine(Transformations[current].GetPosition(), Transformations[next].GetPosition(), Transformations[current].GetUp(), thickness*0.01f, positionColor);
                }
            }

            //Positions
            if(drawPositions) {
                for(int i=start; i<end; i++) {
                    int index = GetKey(i).Index;
                    UltiDraw.DrawSphere(Transformations[index].GetPosition(), Quaternion.identity, thickness*0.025f, positionColor);
                }
            }

            //Locks
            if(drawLocks) {
                for(int i=start; i<end; i++) {
                    int index = GetKey(i).Index;
                    UltiDraw.DrawSphere(Transformations[index].GetPosition(), Quaternion.identity, 0.1f, UltiDraw.Red.Opacity(Locks[index]));
                }
            }

            //Directions
            if(drawDirections) {
                for(int i=start; i<end; i++) {
                    int index = GetKey(i).Index;
                    UltiDraw.DrawLine(Transformations[index].GetPosition(), Transformations[index].GetPosition() + 0.25f*Transformations[index].GetForward(), Transformations[index].GetUp(), thickness*0.025f, 0f, directionColor);
                }
            }

            //Velocities
            if(drawVelocities) {
                for(int i=start; i<end; i++) {
                    int index = GetKey(i).Index;
                    UltiDraw.DrawLine(Transformations[index].GetPosition(), Transformations[index].GetPosition() + GetTemporalScale(Velocities[index]), Transformations[index].GetUp(), thickness*0.0125f, 0f, velocityColor);
                }
            }

            //Angular Velocities
            if(drawAngularVelocities) {
                for(int i=start; i<end; i++) {
                    int index = GetKey(i).Index;
                    UltiDraw.DrawLine(GetPosition(index), GetPosition(index) + GetTemporalScale(AngularVelocities[index]) * GetRotation(index).GetRight(), thickness*0.0125f, 0f, UltiDraw.Red);
                }
            }

            UltiDraw.End();
        }
    }

    public class TimeSeries {
        
        public abstract class Component : TimeSeries {
            public bool DrawGUI = true;
            public bool DrawScene = true;
            public Component(TimeSeries global) : base(global) {}
            public abstract void GUI();
            public abstract void Draw();
        }

        public int PastKeys {get; private set;}
        public int FutureKeys {get; private set;}
        public float PastWindow {get; private set;}
        public float FutureWindow {get; private set;}
        public int Resolution {get; private set;}
        public Sample[] Samples {get; private set;}

        public int Pivot {
            get {return PastSamples;}
        }
        public int SampleCount {
            get {return PastSamples + FutureSamples + 1;}
        }
        public int PastSamples {
            get {return PastKeys * Resolution;}
        }
        public int FutureSamples {
            get {return FutureKeys * Resolution;}
        }
        public int PivotKey {
            get {return PastKeys;}
        }
        public int KeyCount {
            get {return PastKeys + FutureKeys + 1;}
        }
        public float Window {
            get {return PastWindow + FutureWindow;}
        }
        public float DeltaTime {
            get {return Window / (SampleCount-1);}
        }
        public float MaximumFrequency {
            get {return 0.5f * KeyCount / Window;} //Shannon-Nyquist Sampling Theorem fMax <= 0.5*fSignal
        }

        public class Sample {
            public int Index;
            public float Timestamp;
            public Sample(int index, float timestamp) {
                Index = index;
                Timestamp = timestamp;
            }
        }

        public TimeSeries(int pastKeys, int futureKeys, float pastWindow, float futureWindow, int resolution) {
            PastKeys = pastKeys;
            FutureKeys = futureKeys;
            PastWindow = pastWindow;
            FutureWindow = futureWindow;
            Resolution = resolution;
            Samples = new Sample[SampleCount];
            for(int i=0; i<Pivot; i++) {
                Samples[i] = new Sample(i, -PastWindow+i*PastWindow/PastSamples);
            }
            Samples[Pivot] = new Sample(Pivot, 0f);
            for(int i=Pivot+1; i<Samples.Length; i++) {
                Samples[i] = new Sample(i, (i-Pivot)*FutureWindow/FutureSamples);
            }
        }

        protected TimeSeries(TimeSeries global) {
            SetTimeSeries(global);
        }

        public void SetTimeSeries(TimeSeries global) {
            PastKeys = global.PastKeys;
            FutureKeys = global.FutureKeys;
            PastWindow = global.FutureWindow;
            FutureWindow = global.FutureWindow;
            Resolution = global.Resolution;
            Samples = global.Samples;
        }
        
        public float[] GetTimestamps() {
            float[] timestamps = new float[Samples.Length];
            for(int i=0; i<timestamps.Length; i++) {
                timestamps[i] = Samples[i].Timestamp;
            }
            return timestamps;
        }

        public float GetTemporalScale(float value) {
            // return value;
            return value / KeyCount;
        }

        public Vector2 GetTemporalScale(Vector2 value) {
            // return value;
            return value / KeyCount;
        }

        public Vector3 GetTemporalScale(Vector3 value) {
            // return value;
            return value / KeyCount;
        }

        public Sample GetPivot() {
            return Samples[Pivot];
        }

        public Sample GetKey(float timestamp) {
            if(timestamp < -PastWindow || timestamp > FutureWindow) {
                Debug.Log("Given timestamp was " + timestamp + " but must be within " + PastWindow + " and " + FutureWindow + ".");
                return null;
            }
            return Samples[Mathf.RoundToInt(timestamp.Normalize(-PastWindow, FutureWindow, 0f, SampleCount-1))];
        }

        public Sample GetKey(int index) {
            if(index < 0 || index >= KeyCount) {
                Debug.Log("Given key was " + index + " but must be within 0 and " + (KeyCount-1) + ".");
                return null;
            }
            return Samples[index*Resolution];
        }

        public Sample GetPreviousKey(int sample) {
            if(sample < 0 || sample >= Samples.Length) {
                Debug.Log("Given index was " + sample + " but must be within 0 and " + (Samples.Length-1) + ".");
                return null;
            }
            return GetKey(sample/Resolution);
        }

        public Sample GetNextKey(int sample) {
            if(sample < 0 || sample >= Samples.Length) {
                Debug.Log("Given index was " + sample + " but must be within 0 and " + (Samples.Length-1) + ".");
                return null;
            }
            if(sample % Resolution == 0) {
                return GetKey(sample/Resolution);
            } else {
                return GetKey(sample/Resolution + 1);
            }
        }
    }
}
