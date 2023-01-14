using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
    
    [Serializable]
    public class SocketNetwork : NeuralNetwork {
        public string IP = "10.20.185.203";
        public int Port = 25001;
        public string ModelPath = "PAE/Training/5_3Channels.pt";

        public class SocketInference : Inference {

            public float[] X;
            public float[] Y;
            private byte[] SendBuffer = null;
            private byte[] ReceiveBuffer = null;
            private Socket Client = null;
            private int Timeout = 5000; //5s

            public Socket GetClient() {
                return Client;
            }

            public SocketInference(SocketNetwork network) {
                Client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
                Client.SendTimeout = Timeout;
                Client.ReceiveTimeout = Timeout;
                IAsyncResult result = Client.BeginConnect(new IPEndPoint(IPAddress.Parse(network.IP), network.Port), null, null);
                bool success = result.AsyncWaitHandle.WaitOne(Timeout, true);
                if(success) {
                    //Setup Connection
                    Client.EndConnect(result);

                    //Send First Message To Server
                    string message = network.ModelPath;
                    Client.Send(Encoding.UTF8.GetBytes(message));

                    //Retrieve First Message From Server
                    int tensors = 2;
                    byte[] dimBuffer = new byte[tensors*sizeof(int)];
                    Client.ReceiveAll(dimBuffer);
                    int[] dimensions = new int[tensors];
                    for(int i=0; i<dimensions.Length; i++) {
                        dimensions[i] = BitConverter.ToInt32(dimBuffer, i*sizeof(Int32));
                    }
                    X = new float[dimensions[0]];
                    Y = new float[dimensions[1]];
                    SendBuffer = new byte[X.Length*sizeof(float)];
                    ReceiveBuffer = new byte[Y.Length*sizeof(float)];
                } else {
                    //Abort Connection
                    Debug.LogWarning("Connection timed out.");
                    Client.Close();
                }
            }

            public override void Dispose() {
                if(Client.IsConnected()) {
                    Client.Send(new byte[0]);
                    Client.Shutdown(SocketShutdown.Both);
                }
                Client.Close();
            }

            public override int GetFeedSize() {
                return X.Length;
            }

            public override int GetReadSize() {
                return Y.Length;
            }

            public override void Feed(float value) {
                X[Pivot] = value;
            }

            public override float Read() {
                return Y[GetReadSize()-Pivot];
            }

            public override void Run() {
                try {
                    Buffer.BlockCopy(X, 0, SendBuffer, 0, SendBuffer.Length);
                    Client.Send(SendBuffer);
                    Client.ReceiveAll(ReceiveBuffer);
                    for(int i=0; i<Y.Length; i++) {
                        Y[i] = BitConverter.ToSingle(ReceiveBuffer, i*sizeof(float));
                    }
                } catch {
                    // Debug.Log("An error happened during inference.");
                }
            }

            public void Test() {
                Debug.Log("Input: " + X.Format());
                Run();
                Debug.Log("Output: " + Y.Format());
            }

        }

        protected override Inference BuildInference() {
            return new SocketInference(this);
        }

        #if UNITY_EDITOR
        public override void Inspect() {
            IP = EditorGUILayout.TextField("IP", IP);
            Port = EditorGUILayout.IntField("Port", Port);
            ModelPath = EditorGUILayout.TextField("Model Path", ModelPath);
            bool connected = GetSession() != null && GetSession().ToType<SocketInference>().GetClient().IsConnected();
            EditorGUILayout.HelpBox(connected ? "Connected" : "Disconnected", MessageType.None);
            if(connected) {
                if(Utility.GUIButton("Test", UltiDraw.DarkGrey, UltiDraw.White)) {
                    GetSession().ToType<SocketInference>().Test();
                }
            }
        }
        #endif
    }

    public static class SocketExtensions {
        public static bool IsConnected(this Socket socket) {
            return !((socket.Poll(socket.ReceiveTimeout, SelectMode.SelectRead) && (socket.Available == 0)) || !socket.Connected);
        }
        
        public static void ReceiveAll(this Socket socket, byte[] buffer) {
            try {
                int dataRead = 0;
                int dataleft = buffer.Length;
                while(dataRead < buffer.Length) {
                    int recv = socket.Receive(buffer, dataRead, dataleft, SocketFlags.None);
                    if(recv == 0) {
                        break;
                    } else {
                        dataRead += recv;
                        dataleft -= recv;
                    }
                }
            } catch(Exception e) {
                Debug.Log(e);
            }
        }
    }

}