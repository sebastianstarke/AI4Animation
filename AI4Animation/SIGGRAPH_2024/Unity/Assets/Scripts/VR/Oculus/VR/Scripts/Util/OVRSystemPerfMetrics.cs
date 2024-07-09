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
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.Networking;

using Debug = UnityEngine.Debug;

public class OVRSystemPerfMetrics
{
	public const int TcpListeningPort = 32419;
	public const int PayloadTypeMetrics = 100;

	public const int MaxBufferLength = 65536;
	public const int MaxMessageLength = MaxBufferLength - sizeof(int);

	public class PerfMetrics
	{
		public int frameCount;
		public float frameTime;
		public float deltaFrameTime;

		public bool appCpuTime_IsValid;
		public float appCpuTime;
		public bool appGpuTime_IsValid;
		public float appGpuTime;
		public bool compositorCpuTime_IsValid;
		public float compositorCpuTime;
		public bool compositorGpuTime_IsValid;
		public float compositorGpuTime;
		public bool compositorDroppedFrameCount_IsValid;
		public int compositorDroppedFrameCount;
		public bool compositorSpaceWarpMode_IsValid;
		public int compositorSpaceWarpMode;
		public bool systemGpuUtilPercentage_IsValid;
		public float systemGpuUtilPercentage;
		public bool systemCpuUtilAveragePercentage_IsValid;
		public float systemCpuUtilAveragePercentage;
		public bool systemCpuUtilWorstPercentage_IsValid;
		public float systemCpuUtilWorstPercentage;
		public bool deviceCpuClockFrequencyInMHz_IsValid;
		public float deviceCpuClockFrequencyInMHz;
		public bool deviceGpuClockFrequencyInMHz_IsValid;
		public float deviceGpuClockFrequencyInMHz;
		public bool deviceCpuClockLevel_IsValid;
		public int deviceCpuClockLevel;
		public bool deviceGpuClockLevel_IsValid;
		public int deviceGpuClockLevel;
		public bool[] deviceCpuCoreUtilPercentage_IsValid = new bool[OVRPlugin.MAX_CPU_CORES];
		public float[] deviceCpuCoreUtilPercentage = new float[OVRPlugin.MAX_CPU_CORES];

		public string ToJSON()
		{
			OVRSimpleJSON.JSONObject jsonNode = new OVRSimpleJSON.JSONObject();
			jsonNode.Add("frameCount", new OVRSimpleJSON.JSONNumber(frameCount));
			jsonNode.Add("frameTime", new OVRSimpleJSON.JSONNumber(frameTime));
			jsonNode.Add("deltaFrameTime", new OVRSimpleJSON.JSONNumber(deltaFrameTime));
			if (appCpuTime_IsValid)
			{
				jsonNode.Add("appCpuTime", new OVRSimpleJSON.JSONNumber(appCpuTime));
			}
			if (appGpuTime_IsValid)
			{
				jsonNode.Add("appGpuTime", new OVRSimpleJSON.JSONNumber(appGpuTime));
			}
			if (compositorCpuTime_IsValid)
			{
				jsonNode.Add("compositorCpuTime", new OVRSimpleJSON.JSONNumber(compositorCpuTime));
			}
			if (compositorGpuTime_IsValid)
			{
				jsonNode.Add("compositorGpuTime", new OVRSimpleJSON.JSONNumber(compositorGpuTime));
			}
			if (compositorDroppedFrameCount_IsValid)
			{
				jsonNode.Add("compositorDroppedFrameCount", new OVRSimpleJSON.JSONNumber(compositorDroppedFrameCount));
			}
			if (compositorSpaceWarpMode_IsValid)
			{
				jsonNode.Add("compositorSpaceWarpMode", new OVRSimpleJSON.JSONNumber(compositorSpaceWarpMode));
			}
			if (systemGpuUtilPercentage_IsValid)
			{
				jsonNode.Add("systemGpuUtilPercentage", new OVRSimpleJSON.JSONNumber(systemGpuUtilPercentage));
			}
			if (systemCpuUtilAveragePercentage_IsValid)
			{
				jsonNode.Add("systemCpuUtilAveragePercentage", new OVRSimpleJSON.JSONNumber(systemCpuUtilAveragePercentage));
			}
			if (systemCpuUtilWorstPercentage_IsValid)
			{
				jsonNode.Add("systemCpuUtilWorstPercentage", new OVRSimpleJSON.JSONNumber(systemCpuUtilWorstPercentage));
			}
			if (deviceCpuClockFrequencyInMHz_IsValid)
			{
				jsonNode.Add("deviceCpuClockFrequencyInMHz", new OVRSimpleJSON.JSONNumber(deviceCpuClockFrequencyInMHz));
			}
			if (deviceGpuClockFrequencyInMHz_IsValid)
			{
				jsonNode.Add("deviceGpuClockFrequencyInMHz", new OVRSimpleJSON.JSONNumber(deviceGpuClockFrequencyInMHz));
			}
			if (deviceCpuClockLevel_IsValid)
			{
				jsonNode.Add("deviceCpuClockLevel", new OVRSimpleJSON.JSONNumber(deviceCpuClockLevel));
			}
			if (deviceGpuClockLevel_IsValid)
			{
				jsonNode.Add("deviceGpuClockLevel", new OVRSimpleJSON.JSONNumber(deviceGpuClockLevel));
			}
			for (int i = 0; i < OVRPlugin.MAX_CPU_CORES; i++)
			{
				if (deviceCpuCoreUtilPercentage_IsValid[i])
					jsonNode.Add("deviceCpuCore" + i + "UtilPercentage", new OVRSimpleJSON.JSONNumber(deviceCpuCoreUtilPercentage[i]));
			}
			string str = jsonNode.ToString();
			return str;
		}

		public bool LoadFromJSON(string json)
		{
			OVRSimpleJSON.JSONObject jsonNode = OVRSimpleJSON.JSONObject.Parse(json) as OVRSimpleJSON.JSONObject;
			if (jsonNode == null)
			{
				return false;
			}
			frameCount = jsonNode["frameCount"] != null ? jsonNode["frameCount"].AsInt : 0;
			frameTime = jsonNode["frameTime"] != null ? jsonNode["frameTime"].AsFloat : 0;
			deltaFrameTime = jsonNode["deltaFrameTime"] != null ? jsonNode["deltaFrameTime"].AsFloat : 0;
			appCpuTime_IsValid = jsonNode["appCpuTime"] != null;
			appCpuTime = appCpuTime_IsValid ? jsonNode["appCpuTime"].AsFloat : 0;
			appGpuTime_IsValid = jsonNode["appGpuTime"] != null;
			appGpuTime = appGpuTime_IsValid ? jsonNode["appGpuTime"].AsFloat : 0;
			compositorCpuTime_IsValid = jsonNode["compositorCpuTime"] != null;
			compositorCpuTime = compositorCpuTime_IsValid ? jsonNode["compositorCpuTime"].AsFloat : 0;
			compositorGpuTime_IsValid = jsonNode["compositorGpuTime"] != null;
			compositorGpuTime = compositorGpuTime_IsValid ? jsonNode["compositorGpuTime"].AsFloat : 0;
			compositorDroppedFrameCount_IsValid = jsonNode["compositorDroppedFrameCount"] != null;
			compositorDroppedFrameCount = compositorDroppedFrameCount_IsValid ? jsonNode["compositorDroppedFrameCount"].AsInt : 0;
			compositorSpaceWarpMode_IsValid = jsonNode["compositorSpaceWarpMode"] != null;
			compositorSpaceWarpMode = compositorSpaceWarpMode_IsValid ? jsonNode["compositorSpaceWarpMode"].AsInt : 0;
			systemGpuUtilPercentage_IsValid = jsonNode["systemGpuUtilPercentage"] != null;
			systemGpuUtilPercentage = systemGpuUtilPercentage_IsValid ? jsonNode["systemGpuUtilPercentage"].AsFloat : 0;
			systemCpuUtilAveragePercentage_IsValid = jsonNode["systemCpuUtilAveragePercentage"] != null;
			systemCpuUtilAveragePercentage = systemCpuUtilAveragePercentage_IsValid ? jsonNode["systemCpuUtilAveragePercentage"].AsFloat : 0;
			systemCpuUtilWorstPercentage_IsValid = jsonNode["systemCpuUtilWorstPercentage"] != null;
			systemCpuUtilWorstPercentage = systemCpuUtilWorstPercentage_IsValid ? jsonNode["systemCpuUtilWorstPercentage"].AsFloat : 0;
			deviceCpuClockFrequencyInMHz_IsValid = jsonNode["deviceCpuClockFrequencyInMHz"] != null;
			deviceCpuClockFrequencyInMHz = deviceCpuClockFrequencyInMHz_IsValid ? jsonNode["deviceCpuClockFrequencyInMHz"].AsFloat : 0;
			deviceGpuClockFrequencyInMHz_IsValid = jsonNode["deviceGpuClockFrequencyInMHz"] != null;
			deviceGpuClockFrequencyInMHz = deviceGpuClockFrequencyInMHz_IsValid ? jsonNode["deviceGpuClockFrequencyInMHz"].AsFloat : 0;
			deviceCpuClockLevel_IsValid = jsonNode["deviceCpuClockLevel"] != null;
			deviceCpuClockLevel = deviceCpuClockLevel_IsValid ? jsonNode["deviceCpuClockLevel"].AsInt : 0;
			deviceGpuClockLevel_IsValid = jsonNode["deviceGpuClockLevel"] != null;
			deviceGpuClockLevel = deviceGpuClockLevel_IsValid ? jsonNode["deviceGpuClockLevel"].AsInt : 0;
			for (int i = 0; i < OVRPlugin.MAX_CPU_CORES; i++)
			{
				deviceCpuCoreUtilPercentage_IsValid[i] = jsonNode["deviceCpuCore" + i + "UtilPercentage"] != null;
				deviceCpuCoreUtilPercentage[i] = deviceCpuCoreUtilPercentage_IsValid[i] ? jsonNode["deviceCpuCore" + i + "UtilPercentage"].AsFloat : 0;
			}
			return true;
		}
	}

	public class OVRSystemPerfMetricsTcpServer : MonoBehaviour
	{
		public static OVRSystemPerfMetricsTcpServer singleton = null;

		private OVRNetwork.OVRNetworkTcpServer tcpServer = new OVRNetwork.OVRNetworkTcpServer();

		public int listeningPort = OVRSystemPerfMetrics.TcpListeningPort;

		void OnEnable()
		{
			if (singleton != null)
			{
				Debug.LogError("Mutiple OVRSystemPerfMetricsTcpServer exists");
				return;
			}
			else
			{
				singleton = this;
			}

			if (Application.isEditor)
			{
				Application.runInBackground = true;
			}
			tcpServer.StartListening(listeningPort);
		}

		void OnDisable()
		{
			tcpServer.StopListening();

			singleton = null;

			Debug.Log("[OVRSystemPerfMetricsTcpServer] server destroyed");
		}

		private void Update()
		{
			if (tcpServer.HasConnectedClient())
			{
				PerfMetrics metrics = GatherPerfMetrics();
				string json = metrics.ToJSON();
				byte[] bytes = Encoding.UTF8.GetBytes(json);
				tcpServer.Broadcast(OVRSystemPerfMetrics.PayloadTypeMetrics, bytes);
			}
		}

		public PerfMetrics GatherPerfMetrics()
		{
			PerfMetrics metrics = new PerfMetrics();

			metrics.frameCount = Time.frameCount;
			metrics.frameTime = Time.unscaledTime;
			metrics.deltaFrameTime = Time.unscaledDeltaTime;

			float? floatValue;
			int? intValue;

			floatValue = OVRPlugin.GetPerfMetricsFloat(OVRPlugin.PerfMetrics.App_CpuTime_Float);
			metrics.appCpuTime_IsValid = floatValue.HasValue;
			metrics.appCpuTime = floatValue.GetValueOrDefault();

			floatValue = OVRPlugin.GetPerfMetricsFloat(OVRPlugin.PerfMetrics.App_GpuTime_Float);
			metrics.appGpuTime_IsValid = floatValue.HasValue;
			metrics.appGpuTime = floatValue.GetValueOrDefault();

			floatValue = OVRPlugin.GetPerfMetricsFloat(OVRPlugin.PerfMetrics.Compositor_CpuTime_Float);
			metrics.compositorCpuTime_IsValid = floatValue.HasValue;
			metrics.compositorCpuTime = floatValue.GetValueOrDefault();

			floatValue = OVRPlugin.GetPerfMetricsFloat(OVRPlugin.PerfMetrics.Compositor_GpuTime_Float);
			metrics.compositorGpuTime_IsValid = floatValue.HasValue;
			metrics.compositorGpuTime = floatValue.GetValueOrDefault();

			intValue = OVRPlugin.GetPerfMetricsInt(OVRPlugin.PerfMetrics.Compositor_DroppedFrameCount_Int);
			metrics.compositorDroppedFrameCount_IsValid = intValue.HasValue;
			metrics.compositorDroppedFrameCount = intValue.GetValueOrDefault();

			intValue = OVRPlugin.GetPerfMetricsInt(OVRPlugin.PerfMetrics.Compositor_SpaceWarp_Mode_Int);
			metrics.compositorSpaceWarpMode_IsValid = intValue.HasValue;
			metrics.compositorSpaceWarpMode = intValue.GetValueOrDefault();

			floatValue = OVRPlugin.GetPerfMetricsFloat(OVRPlugin.PerfMetrics.System_GpuUtilPercentage_Float);
			metrics.systemGpuUtilPercentage_IsValid = floatValue.HasValue;
			metrics.systemGpuUtilPercentage = floatValue.GetValueOrDefault();

			floatValue = OVRPlugin.GetPerfMetricsFloat(OVRPlugin.PerfMetrics.System_CpuUtilAveragePercentage_Float);
			metrics.systemCpuUtilAveragePercentage_IsValid = floatValue.HasValue;
			metrics.systemCpuUtilAveragePercentage = floatValue.GetValueOrDefault();

			floatValue = OVRPlugin.GetPerfMetricsFloat(OVRPlugin.PerfMetrics.System_CpuUtilWorstPercentage_Float);
			metrics.systemCpuUtilWorstPercentage_IsValid = floatValue.HasValue;
			metrics.systemCpuUtilWorstPercentage = floatValue.GetValueOrDefault();

			floatValue = OVRPlugin.GetPerfMetricsFloat(OVRPlugin.PerfMetrics.Device_CpuClockFrequencyInMHz_Float);
			metrics.deviceCpuClockFrequencyInMHz_IsValid = floatValue.HasValue;
			metrics.deviceCpuClockFrequencyInMHz = floatValue.GetValueOrDefault();

			floatValue = OVRPlugin.GetPerfMetricsFloat(OVRPlugin.PerfMetrics.Device_GpuClockFrequencyInMHz_Float);
			metrics.deviceGpuClockFrequencyInMHz_IsValid = floatValue.HasValue;
			metrics.deviceGpuClockFrequencyInMHz = floatValue.GetValueOrDefault();

			intValue = OVRPlugin.GetPerfMetricsInt(OVRPlugin.PerfMetrics.Device_CpuClockLevel_Int);
			metrics.deviceCpuClockLevel_IsValid = intValue.HasValue;
			metrics.deviceCpuClockLevel = intValue.GetValueOrDefault();

			intValue = OVRPlugin.GetPerfMetricsInt(OVRPlugin.PerfMetrics.Device_GpuClockLevel_Int);
			metrics.deviceGpuClockLevel_IsValid = intValue.HasValue;
			metrics.deviceGpuClockLevel = intValue.GetValueOrDefault();

			for (int i = 0; i < OVRPlugin.MAX_CPU_CORES; i++)
			{
				floatValue = OVRPlugin.GetPerfMetricsFloat(OVRPlugin.PerfMetrics.Device_CpuCore0UtilPercentage_Float);
				metrics.deviceCpuCoreUtilPercentage_IsValid[i] = floatValue.HasValue;
				metrics.deviceCpuCoreUtilPercentage[i] = floatValue.GetValueOrDefault();
			}

			return metrics;
		}

	}
}
