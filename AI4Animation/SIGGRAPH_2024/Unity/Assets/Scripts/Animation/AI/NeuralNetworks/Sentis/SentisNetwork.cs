using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
    [Serializable]
    public class SentisNetwork {

        public ModelAsset Model = null;
        public BackendType Device = BackendType.CPU;

        private Inference Session = null;
        
        public Inference CreateSession(bool debug=true) {
            if(IsActive()) {
                if(debug) {
                    Debug.Log("Session is already active.");
                }
                return null;
            }
            if(Model == null) {
                if(debug) {
                    Debug.Log("Failed creating neural network session because no model was assigned.");
                }
                return null;
            }
            Session = new Inference(Model, Device);
            return Session;
        }

        public void CloseSession(bool debug=true) {
            if(IsActive()) { 
                Session.Dispose();
                Session = null;
            } else {
                if(debug) {
                    Debug.Log("No session currently active.");
                }
            }
        }

        public void RunSession() {
            if(IsActive()) {
                Session.Run();
            } else {
                Debug.Log("No session currently active.");
            }
        }

        public Inference GetSession() {
            return Session;
        }

        public bool IsActive() {
            return Session != null;
        }

        public class Inference : IDisposable {

            private List<float> TimeHistory = new List<float>();
            private const int TimeHorizon = 300;

            private Dictionary<string, Input> Inputs = new Dictionary<string, Input>();
            private Dictionary<string, Output> Outputs = new Dictionary<string, Output>();

            public ModelAsset Asset = null;
            public Model Model = null;
            public IWorker Engine = null;

            public Inference(ModelAsset model, BackendType device) {
                if(model == null) {
                    Debug.Log("No model has been assigned.");
                    return;
                }
                Asset = model;
                Model = ModelLoader.Load(model);
                Engine = WorkerFactory.CreateWorker(device, Model);
                foreach(Model.Input input in Model.inputs) {
                    Inputs.Add(input.name, new Input(input.name, input));
                    Debug.Log("Added Input: " + input.name);
                }
                foreach(string name in Model.outputs) {
                    Outputs.Add(name, new Output(name));
                    Debug.Log("Added Output: " + name);
                }
            }

            public void Dispose() {
                if(Engine != null) {
                    Engine.Dispose();
                }
            }

            public void Run() {
                DateTime timestamp = Utility.GetTimestamp();

                //Run Inference
                try {
                    Dictionary<string, Tensor> tensors = new Dictionary<string, Tensor>();
                    foreach(Input input in Inputs.Values) {
                        tensors.Add(input.Name, new TensorFloat(input.Shape, input.Values));
                        if(input.Pivot != input.Values.Length) {
                            Debug.LogWarning("Input " + input.Name + " in model " + Asset.name + " did not receive all features.");
                        }
                    }
                    Engine.Execute(tensors);
                    foreach(Tensor tensor in tensors.Values) {
                        tensor.Dispose();
                    }
                    foreach(Output output in Outputs.Values) {
                        output.Values = (Engine.PeekOutput(output.Name) as TensorFloat).ToReadOnlyArray();
                    }
                } catch(Exception e) {
                    Debug.Log(e);
                }

                //Update Time History
                while(TimeHistory.Count > TimeHorizon) {
                    TimeHistory.RemoveAt(0);
                }
                TimeHistory.Add((float)Utility.GetElapsedTime(timestamp));

                foreach(Input input in Inputs.Values) {
                    input.Pivot = 0;
                    input.Exceptions.Clear();
                }
                foreach(Output output in Outputs.Values) {
                    output.Pivot = 0;
                    output.Exceptions.Clear();
                }
            }

            public Input GetInput(string name) {
                if(Inputs.ContainsKey(name)) {
                    return Inputs[name];
                } else {
                    Debug.Log("Input tensor with name " + name + " does not exist in model " + Asset.name + ".");
                    return null;
                }
            }

            public Output GetOutput(string name) {
                if(Outputs.ContainsKey(name)) {
                    return Outputs[name];
                } else {
                    Debug.Log("Output tensor with name " + name + " does not exist in model " + Asset.name + ".");
                    return null;
                }
            }

            public bool HasInput(string name) {
                return Inputs.ContainsKey(name);
            }

            public bool HasOutput(string name) {
                return Outputs.ContainsKey(name);
            }

            public void DrawTimeHistory(Vector2 center, Vector2 size, float yMax=float.NaN) {
                UltiDraw.Begin();
                UltiDraw.PlotFunction(center, size, TimeHistory.ToArray(), yMin:0f, yMax:yMax);
                UltiDraw.End();
            }
        }

        public class Input : IDisposable {
            public string Name;
            public SymbolicTensorShape SymbolicShape;
            public TensorShape Shape;
            public float[] Values;

            public int Pivot;
            public HashSet<string> Exceptions = new HashSet<string>();

            public Input(string name, Model.Input node) {
                Name = name;
                SymbolicShape = node.shape;
                if(SymbolicShape.IsFullyKnown()) {
                    Shape = SymbolicShape.ToTensorShape();
                    // Values = new TensorFloat(Shape.ToTensorShape(), new float[Shape.ToTensorShape().ToArray().Product()]);
                    // Values.MakeReadable();
                } else {
                    Shape = new TensorShape(1);
                    // Values = new TensorFloat(new TensorShape(1), new float[1]);
                    // Values.MakeReadable();
                }
                Values = new float[Shape.ToArray().Product()];
            }

            public void Dispose() {

            }

            public void SetDynamicTensorSize(int size) {
                // if(Values.ToReadOnlySpan().Length != size) {
                //     Values = new TensorFloat(new TensorShape(size), new float[size]);
                //     Values.MakeReadable();
                // }
                if(Values.Length != size) {
                    Shape = new TensorShape(size);
                    Values = new float[size];
                }
            }

            public void Feed(float value) {
                try {
                    Values[Pivot] = value;
                    Pivot += 1;
                } catch(Exception e) {
                    if(Exceptions.Add(e.Message)) {
                        Debug.LogWarning(e);
                    }
                }
            }

            public void Feed(float[] values) {
                foreach(float value in values) {
                    Feed(value);
                }
                // try {
                //     Array.Copy(values, 0, Values, Pivot, values.Length);
                //     Pivot += values.Length;
                // } catch(Exception e) {
                //     if(Exceptions.Add(e.Message)) {
                //         Debug.LogWarning(e);
                //     }
                // }
            }

            public void Feed(int[] values) {
                foreach(int value in values) {
                    Feed(value);
                }
            }

            public void Feed(Vector3 vector) {
                Feed(vector.x);
                Feed(vector.y);
                Feed(vector.z);
            }

            public void Feed(Vector3[] vectors) {
                foreach(Vector3 vector in vectors){
                    Feed(vector);
                }
            }

            public void Feed(Vector2 vector) {
                Feed(vector.x);
                Feed(vector.y);
            }

            public void Feed(Vector2[] vectors) {
                foreach(Vector2 vector in vectors){
                    Feed(vector);
                }
            }

            public void FeedXY(Vector3 vector) {
                Feed(vector.x);
                Feed(vector.y);
            }

            public void FeedXZ(Vector3 vector) {
                Feed(vector.x);
                Feed(vector.z);
            }

            public void FeedYZ(Vector3 vector) {
                Feed(vector.y);
                Feed(vector.z);
            }

            public void Feed(Quaternion vector) {
                Feed(vector.x);
                Feed(vector.y);
                Feed(vector.z);
                Feed(vector.w);
            }
        }

        public class Output : IDisposable {
            public string Name;
            public float[] Values;

            public int Pivot;
            public HashSet<string> Exceptions = new HashSet<string>();

            public Output(string name) {
                Name = name;
                Values = null;
            }

            public void Dispose() {
                
            }

            public float Read() {
                try {
                    float value = Values[Pivot];
                    Pivot += 1;
                    return value;
                } catch(Exception e) {
                    if(Exceptions.Add(e.Message)) {
                        Debug.LogWarning(e);
                    }
                    return 0f;
                }
            }

            public float[] ReadAll() {
                // Pivot = Values.ToReadOnlySpan().Length;
                // return Values.ToReadOnlyArray();
                Pivot = Values.Length;
                return Values;
            }

            public float Read(float min, float max) {
                return Mathf.Clamp(Read(), min, max);
            }

            public float[] Read(int count) {
                float[] values = new float[count];
                for(int i=0; i<count; i++) {
                    values[i] = Read();
                }
                return values;
            }

            public float[] Read(int count, float min, float max) {
                float[] values = new float[count];
                for(int i=0; i<count; i++) {
                    values[i] = Read(min, max);
                }
                return values;
            }

            public float ReadBinary() {
                return Read() > 0.5f ? 1f : 0f;
            }

            public float[] ReadBinary(int count) {
                float[] values = new float[count];
                for(int i=0; i<count; i++) {
                    values[i] = ReadBinary();
                }
                return values;
            }

            public Quaternion ReadQuaternion() {
                return new Quaternion(Read(), Read(), Read(), Read()).normalized;
            }

            public Vector3 ReadVector3() {
                return new Vector3(Read(), Read(), Read());
            }

            public Vector2 ReadVector2() {
                return new Vector2(Read(), Read());
            }

            public Vector2[] ReadVector2(int count) {
                Vector2[] values = new Vector2[count];
                for(int i=0; i<values.Length; i++) {
                    values[i] = ReadVector2();
                }
                return values;
            }

            public Vector3 ReadXY() {
                return new Vector3(Read(), Read(), 0f);
            }

            public Vector3 ReadXZ() {
                return new Vector3(Read(), 0f, Read());
            }

            public Vector3 ReadYZ() {
                return new Vector3(0f, Read(), Read());
            }

            public Quaternion ReadRotation2D() {
                Vector3 forward = ReadXZ().normalized;
                if(forward.magnitude == 0f) {
                    forward = Vector3.forward;
                }
                return Quaternion.LookRotation(forward, Vector3.up);
            }

            public Quaternion ReadRotation3D() {
                Vector3 forward = ReadVector3().normalized;
                Vector3 up = ReadVector3().normalized;
                if(forward.magnitude == 0f) {
                    forward = Vector3.forward;
                }
                if(up.magnitude == 0f) {
                    up = Vector3.up;
                }
                return Quaternion.LookRotation(forward, up);
            }

            public Matrix4x4 ReadMatrix2D() {
                return Matrix4x4.TRS(ReadXZ(), ReadRotation2D(), Vector3.one);
            }

            public Matrix4x4 ReadMatrix3D() {
                return Matrix4x4.TRS(ReadVector3(), ReadRotation3D(), Vector3.one);
            }

            public Matrix4x4 ReadRootDelta() {
                Vector3 offset = ReadVector3();
                return Matrix4x4.TRS(new Vector3(offset.x, 0f, offset.z), Quaternion.AngleAxis(offset.y, Vector3.up), Vector3.one);
            }
        }

        #if UNITY_EDITOR
        public bool Inspector(string id=null) {
            EditorGUI.BeginChangeCheck();
            Utility.SetGUIColor(UltiDraw.White);
            using(new EditorGUILayout.VerticalScope ("Box")) {
                Utility.ResetGUIColor();
                if(id != null) {
                    EditorGUILayout.LabelField(id);
                }
                Model = EditorGUILayout.ObjectField("Model", Model, typeof(ModelAsset), true, GUILayout.Width(EditorGUIUtility.currentViewWidth - 30f)) as ModelAsset;
                Device = (BackendType)EditorGUILayout.EnumPopup("Device", Device, GUILayout.Width(EditorGUIUtility.currentViewWidth - 30f));
            }
            return EditorGUI.EndChangeCheck();
        }
        #endif
    }
}