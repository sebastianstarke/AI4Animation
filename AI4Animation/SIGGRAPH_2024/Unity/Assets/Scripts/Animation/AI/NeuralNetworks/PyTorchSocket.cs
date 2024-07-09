using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class PyTorchSocket : MonoBehaviour {
    public string IP = "10.20.185.203";
    public int Port = 25001;
    public string ID = "ID";
    public int BatchSize, InputSize, OutputSize;
    // public string[] Parameters = new string[0];

    private int FeedPivot;
    private int ReadPivot;
    private float[] Send;
    private float[] Receive;
    private byte[] SendBuffer = null;
    private byte[] ReceiveBuffer = null;

    private int ObservationSize = 0;
    private int TargetSize = 0;

    private Socket Client = null;
    private const int Timeout = 5000; //5s

    void Awake() {
        Client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        Client.SendTimeout = Timeout;
        Client.ReceiveTimeout = Timeout;
        IAsyncResult result = Client.BeginConnect(new IPEndPoint(IPAddress.Parse(IP), Port), null, null);
        bool success = result.AsyncWaitHandle.WaitOne(Timeout, true);
        if(success) {
            Send = new float[BatchSize*(InputSize+OutputSize)];
            Receive = new float[BatchSize*OutputSize];
            SendBuffer = new byte[Send.Length*sizeof(float)];
            ReceiveBuffer = new byte[Receive.Length*sizeof(float)];

            Client.EndConnect(result);
            string message = ID;
            message += ";" + BatchSize;
            message += ";" + InputSize;
            message += ";" + OutputSize;
            // for(int i=0; i<Parameters.Length; i++) {
            //     message += ";" + Parameters[i];
            // }
            Client.Send(Encoding.UTF8.GetBytes(message));
        } else {
            Debug.LogWarning("Connection timed out.");
            Client.Close();
        }
    }

    void OnDestroy() {
        if(Client.IsConnected()) {
            Client.Send(new byte[0]);
            Client.Shutdown(SocketShutdown.Both);
        }
        Client.Close();
    }

    public void RunSession() {
        if(FeedPivot == Send.Length) {
            Buffer.BlockCopy(Send, 0, SendBuffer, 0, SendBuffer.Length);
            Client.Send(SendBuffer);
            Client.ReceiveAll(ReceiveBuffer);
            for(int i=0; i<Receive.Length; i++) {
                Receive[i] = BitConverter.ToSingle(ReceiveBuffer, i*sizeof(float));
            }
        } else {
            Debug.Log("Number of given inputs does not match expected number of inputs: " + FeedPivot + " / " + Send.Length + " Actual Input Size: " + ObservationSize + " Actual Output Size: " + TargetSize);
        }
        FeedPivot = 0;
        ReadPivot = 0;
        // Debug.Log("Observation Size: " + ObservationSize + " Target Size: " + TargetSize);
    }

    public void Feed(float value) {
        if(FeedPivot < Send.Length) {
            Send[FeedPivot] = value;
            FeedPivot += 1;
        } else {
            FeedPivot += 1;
            Debug.Log("Attempting to feed more inputs than expected: " + FeedPivot + " / " + Send.Length);
        }
    }
    
    public void Feed(float[] values) {
        for(int i=0; i<values.Length; i++) {
            Feed(values[i]);
        }
    }

    public float Read() {
        if(ReadPivot < Receive.Length) {
            float value = Receive[ReadPivot];
            ReadPivot += 1;
            return value;
        } else {
            ReadPivot += 1;
            Debug.Log("Attempting to read more outputs than available: " + ReadPivot + " / " + Receive.Length);
            return 0f;
        }
    }

    public void Read(int count) {
        float[] values = new float[count];
        for(int i=0; i<count; i++) {
            values[i] = Read();
        }
    }

    public float[] GetOutput() {
        return Receive;
    }

    public void SetObservationSize() {
        ObservationSize = FeedPivot;
    }

    public void SetTargetSize() {
        TargetSize = FeedPivot - ObservationSize;
    }
}
