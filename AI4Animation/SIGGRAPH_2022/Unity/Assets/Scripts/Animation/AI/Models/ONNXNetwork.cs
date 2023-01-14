using System;
using UnityEngine;
using Unity.Barracuda;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
    [Serializable]
    public class ONNXNetwork : NeuralNetwork {

        public NNModel Model = null;
        public WorkerFactory.Device Device = WorkerFactory.Device.CPU;

        public class ONNXInference : Inference {

            public Tensor X = null;
            public Tensor Y = null;
            public Model Model = null;
            public IWorker Engine = null;

            public ONNXInference(NNModel model, WorkerFactory.Device device) {
                if(model == null) {
                    Debug.Log("No model has been assigned.");
                    return;
                }
                Model = ModelLoader.Load(model);
                Engine = Model.CreateWorker(device);
                X = new Tensor(Model.inputs.First().shape);
                Y = null;
            }
            public override void Dispose() {
                if(X != null) {
                    X.Dispose();
                }
                if(Engine != null) {
                    Engine.Dispose();
                }
            }

            public override int GetFeedSize() {
                return X.length;
            }

            public override int GetReadSize() {
                if(Y == null) {
                    Debug.Log("Run inference first to obtain output read size.");
                    return 0;
                }
                return Y.length;
            }

            public override void Feed(float value) {
                X[Pivot] = value;
            }

            public override float Read() {
                if(Y == null) {
                    Debug.Log("Run inference first to obtain output values.");
                    return 0f;
                }
                return Y[Y.length-Pivot];
            }

            public override void Run() {
                //Multiple Inputs
                // Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>();
                // inputs.Add("Input1", new Tensor(new int[]{1,1,1,dim}, X.Flatten()));
                // inputs.Add("Input2", new Tensor(new int[]{1,1,1,dim}, new float[]{1.0f}));

                //Single Input
                Y = Engine.Execute(X).PeekOutput();
            }

        }

        public Tensor[] GetOutputs() {
            if(GetSession() != null) {
                ONNXInference session = (ONNXInference)GetSession();
                Tensor[] outputs = new Tensor[session.Model.outputs.Count];
                for(int i=0; i<outputs.Length; i++) {
                    outputs[i] = session.Engine.PeekOutput(session.Model.outputs[i]);
                }
                return outputs;
            }
            return null;
        }

        public Tensor GetOutput(string name) {
            if(GetSession() != null) {
                ONNXInference session = (ONNXInference)GetSession();
                if(session.Model.outputs.Contains(name)) {
                    return session.Engine.PeekOutput(name);
                } else {
                    Debug.Log("Output with name " + name + " is invalid.");
                }
            }
            return null;
        }

        protected override Inference BuildInference() {
            return new ONNXInference(Model, Device);
        }

        #if UNITY_EDITOR
        public override void Inspect() {
            Model = EditorGUILayout.ObjectField("Model", Model, typeof(NNModel), true) as NNModel;
            Device = (WorkerFactory.Device)EditorGUILayout.EnumPopup("Device", Device);
        }
        #endif

    }
}
