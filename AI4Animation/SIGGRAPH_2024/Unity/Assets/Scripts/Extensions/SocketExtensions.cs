using System;
using System.Net;
using System.Net.Sockets;
using UnityEngine;

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