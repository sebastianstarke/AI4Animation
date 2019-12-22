using System;
using System.Net;
using System.Collections;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

namespace DeepLearning {
        
    public class SocketNetwork : NeuralNetwork {
        public string IP = "127.0.0.1";
        public int  Port = 25001;
        public string ModelPath = "";
        public string[] TensorNames = null;

        private byte[] SendBuffer = null;
        private byte[] ReceiveBuffer = null;
        private Socket Client = null;

        protected override bool SetupDerived() {
            if(Setup) {
                return true;
            }
            try {
                Client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
                Client.Connect(new IPEndPoint(IPAddress.Parse(IP), Port));
                string message = ModelPath;
                for(int i=0; i<TensorNames.Length; i++) {
                    message += ";" + TensorNames[i];
                }
                Client.Send(Encoding.UTF8.GetBytes(message));
                byte[] dimBuffer = new byte[(2+TensorNames.Length)*sizeof(int)];
                Client.ReceiveAll(dimBuffer);
                int[] dimensions = new int[2+TensorNames.Length];
                for(int i=0; i<dimensions.Length; i++) {
                    dimensions[i] = BitConverter.ToInt32(dimBuffer, i*sizeof(Int32));
                }
                if(dimensions.Sum() == 0) {
                    Client.Shutdown(SocketShutdown.Both);
                    Client.Close();
                    return false;
                } else {
                    X = CreateMatrix(dimensions[0], 1, "X");
                    Y = CreateMatrix(dimensions[1], 1, "Y");
                    SendBuffer = new byte[X.GetRows()*sizeof(float)];
                    ReceiveBuffer = new byte[Y.GetRows()*sizeof(float)];
                    for(int i=0; i<TensorNames.Length; i++){
                        Matrix m = CreateMatrix(dimensions[2+i], 1, TensorNames[i]);
                        ReceiveBuffer = ArrayExtensions.Concat(ReceiveBuffer, new byte[m.GetRows()*sizeof(float)]);
                    }
                    return true;
                }
            } catch {
                return false;
            }
        }

        protected override bool ShutdownDerived() {
            if(Setup) {
                if(Client.Connected) {
                    Client.Send(new byte[0]);
                    Client.Shutdown(SocketShutdown.Both);
                }
                Client.Close();
                DeleteMatrices();
                ResetPredictionTime();
                ResetPivot();
            }
            return false;
        }

        protected override void PredictDerived() {
            try {
                Buffer.BlockCopy(X.Flatten(), 0, SendBuffer, 0, SendBuffer.Length);
                Client.Send(SendBuffer);
                Client.ReceiveAll(ReceiveBuffer);
                int index = 0;
                for(int i=1; i<Matrices.Count; i++) {
                    for(int j=0; j<Matrices[i].GetRows(); j++) {
                        Matrices[i].SetValue(j, 0, BitConverter.ToSingle(ReceiveBuffer, index*sizeof(float)));
                        index += 1;
                    }
                }
            } catch {
                //Debug.Log("Neural network socket was setup but prediction failed.");
                Setup = ShutdownDerived();
            }
        }
    }

    public static class SocketExtensions {
        public static void ReceiveAll(this Socket socket, byte[] buffer) {
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
        }
    }

}