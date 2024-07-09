/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Licensed under the Oculus SDK License Agreement (the "License");
 * you may not use the Oculus SDK except in compliance with the License,
 * which is provided at the time of installation or download, or which
 * otherwise accompanies this software in either electronic or hard copy form.
 *
 * You may obtain a copy of the License at
 *
 * https://developer.oculus.com/licenses/oculussdk/
 *
 * Unless required by applicable law or agreed to in writing, the Oculus SDK
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using UnityEngine;

using Debug = UnityEngine.Debug;

public class OVRNetwork
{
	public const int MaxBufferLength = 65536;
	public const int MaxPayloadLength = MaxBufferLength - FrameHeader.StructSize;

	public const uint FrameHeaderMagicIdentifier = 0x5283A76B;

	[StructLayout(LayoutKind.Sequential, Pack = 1)]
	struct FrameHeader
	{
		public uint protocolIdentifier;
		public int payloadType;
		public int payloadLength;

		public const int StructSize = sizeof(uint) + sizeof(int) + sizeof(int);

		// endianness conversion is NOT handled since all our current mobile/PC devices are little-endian
		public byte[] ToBytes()
		{
			int size = Marshal.SizeOf(this);
			Trace.Assert(size == StructSize);

			byte[] arr = new byte[size];

			IntPtr ptr = Marshal.AllocHGlobal(size);
			Marshal.StructureToPtr(this, ptr, true);
			Marshal.Copy(ptr, arr, 0, size);
			Marshal.FreeHGlobal(ptr);
			return arr;
		}

		public static FrameHeader FromBytes(byte[] arr)
		{
			FrameHeader header = new FrameHeader();

			int size = Marshal.SizeOf(header);
			Trace.Assert(size == StructSize);

			IntPtr ptr = Marshal.AllocHGlobal(size);

			Marshal.Copy(arr, 0, ptr, size);

			header = (FrameHeader)Marshal.PtrToStructure(ptr, header.GetType());
			Marshal.FreeHGlobal(ptr);

			return header;
		}
	}

	public class OVRNetworkTcpServer
	{
		public TcpListener tcpListener = null;

		private readonly object clientsLock = new object();
		public readonly List<TcpClient> clients = new List<TcpClient>();

		public void StartListening(int listeningPort)
		{
			if (tcpListener != null)
			{
				Debug.LogWarning("[OVRNetworkTcpServer] tcpListener is not null");
				return;
			}

			IPAddress localAddr = IPAddress.Any;

			tcpListener = new TcpListener(localAddr, listeningPort);
			try
			{
				tcpListener.Start();
				Debug.LogFormat("TcpListener started. Local endpoint: {0}", tcpListener.LocalEndpoint.ToString());
			}
			catch (SocketException e)
			{
				Debug.LogWarningFormat("[OVRNetworkTcpServer] Unsable to start TcpListener. Socket exception: {0}", e.Message);
				Debug.LogWarning("It could be caused by multiple instances listening at the same port, or the port is forwarded to the Android device through ADB");
				Debug.LogWarning("If the port is forwarded through ADB, use the Android Tools in Tools/Oculus/System Metrics Profiler to kill the server");
				tcpListener = null;
			}

			if (tcpListener != null)
			{
				Debug.LogFormat("[OVRNetworkTcpServer] Start Listening on port {0}", listeningPort);

				try
				{
					tcpListener.BeginAcceptTcpClient(new AsyncCallback(DoAcceptTcpClientCallback), tcpListener);
				}
				catch (Exception e)
				{
					Debug.LogWarningFormat("[OVRNetworkTcpServer] can't accept new client: {0}", e.Message);
				}
			}
		}

		public void StopListening()
		{
			if (tcpListener == null)
			{
				Debug.LogWarning("[OVRNetworkTcpServer] tcpListener is null");
				return;
			}

			lock (clientsLock)
			{
				clients.Clear();
			}
			tcpListener.Stop();
			tcpListener = null;

			Debug.Log("[OVRNetworkTcpServer] Stopped listening");
		}

		private void DoAcceptTcpClientCallback(IAsyncResult ar)
		{
			TcpListener listener = ar.AsyncState as TcpListener;
			try
			{
				TcpClient client = listener.EndAcceptTcpClient(ar);
				lock (clientsLock)
				{
					clients.Add(client);
					Debug.Log("[OVRNetworkTcpServer] client added");
				}

				try
				{
					tcpListener.BeginAcceptTcpClient(new AsyncCallback(DoAcceptTcpClientCallback), tcpListener);
				}
				catch (Exception e)
				{
					Debug.LogWarningFormat("[OVRNetworkTcpServer] can't accept new client: {0}", e.Message);
				}
			}
			catch (ObjectDisposedException)
			{
				// Do nothing. It happens when stop preview in editor, which is normal behavior.
			}
			catch (Exception e)
			{
				Debug.LogWarningFormat("[OVRNetworkTcpServer] EndAcceptTcpClient failed: {0}", e.Message);
			}
		}

		public bool HasConnectedClient()
		{
			lock (clientsLock)
			{
				foreach (TcpClient client in clients)
				{
					if (client.Connected)
					{
						return true;
					}
				}
			}
			return false;
		}

		public void Broadcast(int payloadType, byte[] payload)
		{
			if (payload.Length > OVRNetwork.MaxPayloadLength)
			{
				Debug.LogWarningFormat("[OVRNetworkTcpServer] drop payload because it's too long: {0} bytes", payload.Length);
			}

			FrameHeader header = new FrameHeader();
			header.protocolIdentifier = FrameHeaderMagicIdentifier;
			header.payloadType = payloadType;
			header.payloadLength = payload.Length;

			byte[] headerBuffer = header.ToBytes();

			byte[] dataBuffer = new byte[headerBuffer.Length + payload.Length];
			headerBuffer.CopyTo(dataBuffer, 0);
			payload.CopyTo(dataBuffer, headerBuffer.Length);

			lock (clientsLock)
			{
				foreach (TcpClient client in clients)
				{
					if (client.Connected)
					{
						try
						{
							client.GetStream().BeginWrite(dataBuffer, 0, dataBuffer.Length, new AsyncCallback(DoWriteDataCallback), client.GetStream());
						}
						catch (SocketException e)
						{
							Debug.LogWarningFormat("[OVRNetworkTcpServer] close client because of socket error: {0}", e.Message);
							client.GetStream().Close();
							client.Close();
						}
					}
				}
			}
		}

		private void DoWriteDataCallback(IAsyncResult ar)
		{
			NetworkStream stream = ar.AsyncState as NetworkStream;
			stream.EndWrite(ar);
		}
	}

	public class OVRNetworkTcpClient
	{
		public Action connectionStateChangedCallback;
		public Action<int, byte[], int, int> payloadReceivedCallback;

		public enum ConnectionState
		{
			Disconnected,
			Connected,
			Connecting
		}

		public ConnectionState connectionState
		{
			get
			{
				if (tcpClient == null)
				{
					return ConnectionState.Disconnected;
				}
				else
				{
					if (tcpClient.Connected)
					{
						return ConnectionState.Connected;
					}
					else
					{
						return ConnectionState.Connecting;
					}
				}
			}
		}

		public bool Connected
		{
			get
			{
				return connectionState == ConnectionState.Connected;
			}
		}

		TcpClient tcpClient = null;

		byte[][] receivedBuffers = { new byte[OVRNetwork.MaxBufferLength], new byte[OVRNetwork.MaxBufferLength] };
		int receivedBufferIndex = 0;
		int receivedBufferDataSize = 0;
		ManualResetEvent readyReceiveDataEvent = new ManualResetEvent(true);

		public void Connect(int listeningPort)
		{
			if (tcpClient == null)
			{
				receivedBufferIndex = 0;
				receivedBufferDataSize = 0;
				readyReceiveDataEvent.Set();

				string remoteAddress = "127.0.0.1";
				tcpClient = new TcpClient(AddressFamily.InterNetwork);
				tcpClient.BeginConnect(remoteAddress, listeningPort, new AsyncCallback(ConnectCallback), tcpClient);

				if (connectionStateChangedCallback != null)
				{
					connectionStateChangedCallback();
				}
			}
			else
			{
				Debug.LogWarning("[OVRNetworkTcpClient] already connected");
			}
		}

		void ConnectCallback(IAsyncResult ar)
		{
			try
			{
				TcpClient client = ar.AsyncState as TcpClient;
				client.EndConnect(ar);
				Debug.LogFormat("[OVRNetworkTcpClient] connected to {0}", client.ToString());
			}
			catch (Exception e)
			{
				Debug.LogWarningFormat("[OVRNetworkTcpClient] connect error {0}", e.Message);
			}

			if (connectionStateChangedCallback != null)
			{
				connectionStateChangedCallback();
			}
		}

		public void Disconnect()
		{
			if (tcpClient != null)
			{
				if (!readyReceiveDataEvent.WaitOne(5))
				{
					Debug.LogWarning("[OVRNetworkTcpClient] readyReceiveDataEvent not signaled. data receiving timeout?");
				}

				Debug.Log("[OVRNetworkTcpClient] close tcpClient");
				try
				{
					tcpClient.GetStream().Close();
					tcpClient.Close();
				}
				catch (Exception e)
				{
					Debug.LogWarning("[OVRNetworkTcpClient] " + e.Message);
				}
				tcpClient = null;

				if (connectionStateChangedCallback != null)
				{
					connectionStateChangedCallback();
				}
			}
			else
			{
				Debug.LogWarning("[OVRNetworkTcpClient] not connected");
			}
		}

		public void Tick()
		{
			if (tcpClient == null || !tcpClient.Connected)
			{
				return;
			}

			if (readyReceiveDataEvent.WaitOne(TimeSpan.Zero))
			{
				if (tcpClient.GetStream().DataAvailable)
				{
					if (receivedBufferDataSize >= OVRNetwork.MaxBufferLength)
					{
						Debug.LogWarning("[OVRNetworkTcpClient] receive buffer overflow. It should not happen since we have the constraint on message size");
						Disconnect();
						return;
					}

					readyReceiveDataEvent.Reset();
					int maximumDataSize = OVRSystemPerfMetrics.MaxBufferLength - receivedBufferDataSize;

					tcpClient.GetStream().BeginRead(receivedBuffers[receivedBufferIndex], receivedBufferDataSize, maximumDataSize, new AsyncCallback(OnReadDataCallback), tcpClient.GetStream());
				}
			}
		}

		void OnReadDataCallback(IAsyncResult ar)
		{
			NetworkStream stream = ar.AsyncState as NetworkStream;
			try
			{
				int numBytes = stream.EndRead(ar);
				receivedBufferDataSize += numBytes;

				while (receivedBufferDataSize >= FrameHeader.StructSize)
				{
					FrameHeader header = FrameHeader.FromBytes(receivedBuffers[receivedBufferIndex]);
					if (header.protocolIdentifier != OVRNetwork.FrameHeaderMagicIdentifier)
					{
						Debug.LogWarning("[OVRNetworkTcpClient] header mismatch");
						Disconnect();
						return;
					}

					if (header.payloadLength < 0 || header.payloadLength > OVRNetwork.MaxPayloadLength)
					{
						Debug.LogWarningFormat("[OVRNetworkTcpClient] Sanity check failed. PayloadLength %d", header.payloadLength);
						Disconnect();
						return;
					}

					if (receivedBufferDataSize >= FrameHeader.StructSize + header.payloadLength)
					{
						if (payloadReceivedCallback != null)
						{
							payloadReceivedCallback(header.payloadType, receivedBuffers[receivedBufferIndex], FrameHeader.StructSize, header.payloadLength);
						}

						// swap receive buffer
						int newBufferIndex = 1 - receivedBufferIndex;
						int newBufferDataSize = receivedBufferDataSize - (FrameHeader.StructSize + header.payloadLength);
						if (newBufferDataSize > 0)
						{
							Array.Copy(receivedBuffers[receivedBufferIndex], (FrameHeader.StructSize + header.payloadLength), receivedBuffers[newBufferIndex], 0, newBufferDataSize);
						}
						receivedBufferIndex = newBufferIndex;
						receivedBufferDataSize = newBufferDataSize;
					}
				}
				readyReceiveDataEvent.Set();
			}
			catch (SocketException e)
			{
				Debug.LogErrorFormat("[OVRNetworkTcpClient] OnReadDataCallback: socket error: {0}", e.Message);
				Disconnect();
			}
		}
	}
}
