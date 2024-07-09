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

using System.Collections.Generic;
using System;
using System.Reflection;
using UnityEngine;
using UnityEditor;
using System.Text;
using System.IO;

public class OVRSystemProfilerPanel : EditorWindow {
	[MenuItem("Oculus/Tools/(Deprecated) Oculus Profiler Panel", false, 200000)]
	public static void ShowWindow()
	{
		EditorWindow.GetWindow(typeof(OVRSystemProfilerPanel), false, "Oculus Profiler");
		OVRPlugin.SendEvent("oculus_profiler_panel", "show_window");
	}

	bool showAndroidOptions = false;
	OVRNetwork.OVRNetworkTcpClient tcpClient = new OVRNetwork.OVRNetworkTcpClient();
	int remoteListeningPort = OVRSystemPerfMetrics.TcpListeningPort;

	//OVRSystemPerfMetrics.PerfMetrics lastReceivedMetrics;

	const int maxMetricsFrames = 120;
	const float metricsHistoryDuration = 1.0f;

	List<OVRSystemPerfMetrics.PerfMetrics> receivedMetricsList = new List<OVRSystemPerfMetrics.PerfMetrics>();
	bool pauseReceiveMetrics = false;
	bool repaintRequested = false;

	const float labelWidth = 140.0f;
	const float panelInnerRightPad = 6.0f;
	const float progressBarPad = 5.0f;
	const float progressBarHeight = 18.0f;

	string androidSdkRootPath;
	OVRADBTool adbTool;

	private static GUIStyle odhCalloutBackgroundStyle;
	private static GUIStyle odhCalloutTextStyle;

	// The actual window code goes here
	void OnGUI()
	{
		if (odhCalloutBackgroundStyle == null)
		{
			odhCalloutBackgroundStyle = new GUIStyle(EditorStyles.helpBox);
			var odhCalloutBackgroundStyleTex = new Texture2D(1, 1);
			odhCalloutBackgroundStyleTex.SetPixel(0, 0, new Color(0.9f, 0.8f, 0.2f, 0.2f));
			odhCalloutBackgroundStyleTex.Apply();
			odhCalloutBackgroundStyle.normal.background = odhCalloutBackgroundStyleTex;
		}

		if (odhCalloutTextStyle == null)
		{
			odhCalloutTextStyle = new GUIStyle(EditorStyles.label);
			odhCalloutTextStyle.richText = true;
			odhCalloutTextStyle.wordWrap = true;
		}

		// ODH Callout Section
		GUILayout.BeginHorizontal(odhCalloutBackgroundStyle);
		var script = MonoScript.FromScriptableObject(this);
		string assetPath = AssetDatabase.GetAssetPath(script);
		string editorPath = Path.GetDirectoryName(assetPath);
		string odhIconPath = Path.Combine(editorPath, "Textures\\mqdh_icon.png");
		Texture ODHIcon = (Texture)EditorGUIUtility.Load(odhIconPath);
		GUILayout.Box(ODHIcon, GUILayout.Width(60.0f), GUILayout.Height(60.0f));

		GUILayout.BeginVertical();

		EditorGUILayout.LabelField("<b>This tool is deprecated.</b> Profile builds through the Metrics section of "
			+ "<b>Meta Quest Developer Hub</b>, a desktop companion tool that streamlines the Quest development workflow.",
			odhCalloutTextStyle);
		GUIContent ODHLabel = new GUIContent("Download Meta Quest Developer Hub");
#if UNITY_2021_1_OR_NEWER
		if (EditorGUILayout.LinkButton(ODHLabel))
#else
		if (GUILayout.Button(ODHLabel, GUILayout.ExpandWidth(false)))
#endif
		{
#if UNITY_EDITOR_WIN
			Application.OpenURL("https://developer.oculus.com/downloads/package/oculus-developer-hub-win/?source=unity");
#elif UNITY_EDITOR_OSX
			Application.OpenURL("https://developer.oculus.com/downloads/package/oculus-developer-hub-mac/?source=unity");
#endif
		}
		GUILayout.EndVertical();
		GUILayout.EndHorizontal();
		GUILayout.Space(15.0f);

		showAndroidOptions = EditorGUILayout.Foldout(showAndroidOptions, "Android Tools");

		if (showAndroidOptions)
		{
			GUILayout.BeginHorizontal();
			EditorGUILayout.LabelField("Android SDK root path: ", androidSdkRootPath);
			GUILayout.EndHorizontal();

			GUILayout.BeginHorizontal();
			if (GUILayout.Button("Start Server"))
			{
				if (adbTool == null)
				{
					adbTool = new OVRADBTool(androidSdkRootPath);
				}
				if (adbTool.isReady)
				{
					int exitCode = adbTool.StartServer(null);
					EditorUtility.DisplayDialog("ADB StartServer", (exitCode == 0 ? "Success" : "Failure. ExitCode = " + exitCode.ToString()), "Ok");
				}
				else
				{
					EditorUtility.DisplayDialog("Can't locate ADBTool", adbTool.adbPath, "Ok");
				}
			}
			if (GUILayout.Button("Kill Server"))
			{
				if (adbTool == null)
				{
					adbTool = new OVRADBTool(androidSdkRootPath);
				}
				if (adbTool.isReady)
				{
					int exitCode = adbTool.KillServer(null);
					EditorUtility.DisplayDialog("ADB KillServer", (exitCode == 0 ? "Success" : "Failure. ExitCode = " + exitCode.ToString()), "Ok");
				}
				else
				{
					EditorUtility.DisplayDialog("Can't locate ADBTool", adbTool.adbPath, "Ok");
				}
			}
			if (GUILayout.Button("Forward Port"))
			{
				if (adbTool == null)
				{
					adbTool = new OVRADBTool(androidSdkRootPath);
				}
				if (adbTool.isReady)
				{
					int exitCode = adbTool.ForwardPort(remoteListeningPort, null);
					EditorUtility.DisplayDialog("ADB ForwardPort", (exitCode == 0 ? "Success" : "Failure. ExitCode = " + exitCode.ToString()), "Ok");
					OVRPlugin.SendEvent("device_metrics_profiler", (exitCode == 0 ? "adb_forward_success" : "adb_forward_failure"));
				}
				else
				{
					EditorUtility.DisplayDialog("Can't locate ADBTool", adbTool.adbPath, "Ok");
				}
			}
			if (GUILayout.Button("Release Port"))
			{
				if (adbTool == null)
				{
					adbTool = new OVRADBTool(androidSdkRootPath);
				}
				if (adbTool.isReady)
				{
					int exitCode = adbTool.ReleasePort(remoteListeningPort, null);
					EditorUtility.DisplayDialog("ADB ReleasePort", (exitCode == 0 ? "Success" : "Failure. ExitCode = " + exitCode.ToString()), "Ok");
				}
				else
				{
					EditorUtility.DisplayDialog("Can't locate ADBTool", adbTool.adbPath, "Ok");
				}
			}
			GUILayout.EndHorizontal();
		}

		EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

		GUILayout.BeginHorizontal();
		remoteListeningPort = EditorGUILayout.DelayedIntField("Remote Port", remoteListeningPort);

		if (tcpClient.connectionState == OVRNetwork.OVRNetworkTcpClient.ConnectionState.Disconnected)
		{
			if (GUILayout.Button("Connect"))
			{
				ConnectPerfMetricsTcpServer();
				pauseReceiveMetrics = false;
				OVRPlugin.SendEvent("device_metrics_profiler", "connect");
			}
		}
		else
		{
			if (tcpClient.connectionState == OVRNetwork.OVRNetworkTcpClient.ConnectionState.Connecting)
			{
				if (GUILayout.Button("Connecting ... Click again to Cancel"))
				{
					DisconnectPerfMetricsTcpServer();
					OVRPlugin.SendEvent("device_metrics_profiler", "cancel");
				}
			}
			else
			{
				if (GUILayout.Button("Disconnect"))
				{
					DisconnectPerfMetricsTcpServer();
					OVRPlugin.SendEvent("device_metrics_profiler", "disconnect");
				}

				if (GUILayout.Button(pauseReceiveMetrics ? "Continue" : "Pause"))
				{
					pauseReceiveMetrics = !pauseReceiveMetrics;
				}
			}
		}
		GUILayout.EndHorizontal();

		EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

		lock (receivedMetricsList)
		{
			PresentIntProperty("Frame Count", "frameCount");
			PresentIntProperty("Dropped Frame Count", "compositorDroppedFrameCount");

			float? avgFrameTime = GetAveragePerfValueFloat("deltaFrameTime");
			if (avgFrameTime.HasValue)
			{
				float fps = 1.0f / avgFrameTime.Value;
				EditorGUILayout.BeginHorizontal();
				EditorGUILayout.LabelField("FPS", GUILayout.Width(labelWidth));
				EditorGUILayout.LabelField(string.Format("{0:F1}", fps));
				EditorGUILayout.EndHorizontal();
			}

			int? deviceCpuClockLevel = GetLatestPerfValueInt("deviceCpuClockLevel");
			int? deviceGpuClockLevel = GetLatestPerfValueInt("deviceGpuClockLevel");
			float? deviceCpuClockFrequencyInMHz = GetLatestPerfValueFloat("deviceCpuClockFrequencyInMHz");
			float? deviceGpuClockFrequencyInMHz = GetLatestPerfValueFloat("deviceGpuClockFrequencyInMHz");

			if (deviceCpuClockLevel.HasValue || deviceCpuClockFrequencyInMHz.HasValue)
			{
				string cpuLabel;
				string cpuText;
				if (deviceCpuClockLevel.HasValue && deviceCpuClockFrequencyInMHz.HasValue)
				{
					cpuLabel = "CPU Level (Freq)";
					cpuText = string.Format("{0} ({1:F0} MHz)", deviceCpuClockLevel, deviceCpuClockFrequencyInMHz);
				}
				else if (deviceCpuClockLevel.HasValue)
				{
					cpuLabel = "CPU Level";
					cpuText = string.Format("{0}", deviceCpuClockLevel);
				}
				else
				{
					cpuLabel = "CPU Frequency";
					cpuText = string.Format("{0:F0} MHz", deviceCpuClockFrequencyInMHz);
				}
				PresentText(cpuLabel, cpuText);
			}

			if (deviceGpuClockLevel.HasValue || deviceGpuClockFrequencyInMHz.HasValue)
			{
				string cpuLabel;
				string cpuText;
				if (deviceGpuClockLevel.HasValue && deviceGpuClockFrequencyInMHz.HasValue)
				{
					cpuLabel = "GPU Level (Freq)";
					cpuText = string.Format("{0} ({1:F0} MHz)", deviceGpuClockLevel, deviceGpuClockFrequencyInMHz);
				}
				else if (deviceGpuClockLevel.HasValue)
				{
					cpuLabel = "GPU Level";
					cpuText = string.Format("{0}", deviceGpuClockLevel);
				}
				else
				{
					cpuLabel = "GPU Frequency";
					cpuText = string.Format("{0:F0} MHz", deviceGpuClockFrequencyInMHz);
				}
				PresentText(cpuLabel, cpuText);
			}

			PresentColumnTitles("Current", "Average", "Peak");

			PresentFloatTimeInMs("Frame Time", "deltaFrameTime", 0.020f, true, true);
			PresentFloatTimeInMs("App CPU Time", "appCpuTime", 0.020f, true, true);
			PresentFloatTimeInMs("App GPU Time", "appGpuTime", 0.020f, true, true);
			PresentFloatTimeInMs("Compositor CPU Time", "compositorCpuTime", 0.020f, true, true);
			PresentFloatTimeInMs("Compositor GPU Time", "compositorGpuTime", 0.020f, true, true);
			PresentFloatPercentage("CPU Util (Average)", "systemCpuUtilAveragePercentage", false, false);
			PresentFloatPercentage("CPU Util (Worst Core)", "systemCpuUtilWorstPercentage", false, false);
			PresentFloatPercentage("GPU Util", "systemGpuUtilPercentage", false, false);
		}
	}

	void GetMetricsField(string propertyName, out FieldInfo baseFieldInfo, out FieldInfo validalityFieldInfo)
	{
		baseFieldInfo = typeof(OVRSystemPerfMetrics.PerfMetrics).GetField(propertyName);
		validalityFieldInfo = typeof(OVRSystemPerfMetrics.PerfMetrics).GetField(propertyName + "_IsValid");
	}

	bool HasValidPerfMetrics(string propertyName)
	{
		FieldInfo baseFieldInfo, validalityFieldInfo;
		GetMetricsField(propertyName, out baseFieldInfo, out validalityFieldInfo);

		if (baseFieldInfo == null || (validalityFieldInfo != null && validalityFieldInfo.FieldType != typeof(bool)))
		{
			Debug.LogWarning("[OVRSystemProfilerPanel] Unable to find property " + propertyName);
			return false;
		}

		if (validalityFieldInfo == null)
		{
			return true;
		}

		for (int i = receivedMetricsList.Count - 1; i >= 0; --i)
		{
			var metrics = receivedMetricsList[i];
			if (validalityFieldInfo != null && (bool)validalityFieldInfo.GetValue(metrics))
			{
				return true;
			}
		}

		return false;
	}

	int? GetLatestPerfValueInt(string propertyName)
	{
		FieldInfo baseFieldInfo, validalityFieldInfo;
		GetMetricsField(propertyName, out baseFieldInfo, out validalityFieldInfo);

		if (baseFieldInfo == null || baseFieldInfo.FieldType != typeof(int) ||
			(validalityFieldInfo != null && validalityFieldInfo.FieldType != typeof(bool)))
		{
			Debug.LogWarning("[OVRSystemProfilerPanel] GetLatestPerfValueInt(): Type mismatch");
			return null;
		}

		if (receivedMetricsList.Count == 0)
		{
			return null;
		}

		for (int i = receivedMetricsList.Count - 1; i >= 0; --i)
		{
			var metrics = receivedMetricsList[i];
			if (validalityFieldInfo == null || (validalityFieldInfo != null && (bool)validalityFieldInfo.GetValue(metrics)))
			{
				return (int)baseFieldInfo.GetValue(metrics);
			}
		}

		return null;
	}

	float? GetLatestPerfValueFloat(string propertyName)
	{
		FieldInfo baseFieldInfo, validalityFieldInfo;
		GetMetricsField(propertyName, out baseFieldInfo, out validalityFieldInfo);

		if (baseFieldInfo == null || baseFieldInfo.FieldType != typeof(float) ||
			(validalityFieldInfo != null && validalityFieldInfo.FieldType != typeof(bool)))
		{
			Debug.LogWarning("[OVRSystemProfilerPanel] GetLatestPerfValueFloat(): Type mismatch");
			return null;
		}

		if (receivedMetricsList.Count == 0)
		{
			return null;
		}

		for (int i = receivedMetricsList.Count - 1; i >= 0; --i)
		{
			var metrics = receivedMetricsList[i];
			if (validalityFieldInfo == null || (validalityFieldInfo != null && (bool)validalityFieldInfo.GetValue(metrics)))
			{
				return (float)baseFieldInfo.GetValue(metrics);
			}
		}

		return null;
	}

	float? GetAveragePerfValueFloat(string propertyName)
	{
		FieldInfo baseFieldInfo, validalityFieldInfo;
		GetMetricsField(propertyName, out baseFieldInfo, out validalityFieldInfo);

		if (baseFieldInfo == null || baseFieldInfo.FieldType != typeof(float) ||
			(validalityFieldInfo != null && validalityFieldInfo.FieldType != typeof(bool)))
		{
			Debug.LogWarning("[OVRSystemProfilerPanel] GetAveragePerfValueFloat(): Type mismatch");
			return null;
		}

		int count = 0;
		float sum = 0;

		OVRSystemPerfMetrics.PerfMetrics lastMetrics = null;
		int metricsIndex;
		for (metricsIndex = receivedMetricsList.Count - 1; metricsIndex >= 0; --metricsIndex)
		{
			var metrics = receivedMetricsList[metricsIndex];
			if (validalityFieldInfo != null && !(bool)validalityFieldInfo.GetValue(metrics))
			{
				continue;
			}
			lastMetrics = metrics;
			break;
		}

		if (lastMetrics == null)
		{
			return null;
		}

		for (; metricsIndex >=0; -- metricsIndex)
		{
			var metrics = receivedMetricsList[metricsIndex];
			if (metrics.frameTime < lastMetrics.frameTime - metricsHistoryDuration)
			{
				break;
			}
			if (validalityFieldInfo != null && !(bool)validalityFieldInfo.GetValue(metrics))
			{
				continue;
			}
			sum += (float)baseFieldInfo.GetValue(metrics);
			count++;
		}

		if (count == 0)
		{
			return null;
		}
		else
		{
			return sum / count;
		}
	}

	float? GetMaxPerfValueFloat(string propertyName)
	{
		FieldInfo baseFieldInfo, validalityFieldInfo;
		GetMetricsField(propertyName, out baseFieldInfo, out validalityFieldInfo);

		if (baseFieldInfo == null || baseFieldInfo.FieldType != typeof(float) ||
			(validalityFieldInfo != null && validalityFieldInfo.FieldType != typeof(bool)))
		{
			Debug.LogWarning("[OVRSystemProfilerPanel] GetMaxPerfValueFloat(): Type mismatch");
			return null;
		}

		OVRSystemPerfMetrics.PerfMetrics lastMetrics = null;
		int metricsIndex;
		for (metricsIndex = receivedMetricsList.Count - 1; metricsIndex >= 0; --metricsIndex)
		{
			var metrics = receivedMetricsList[metricsIndex];
			if (validalityFieldInfo != null && !(bool)validalityFieldInfo.GetValue(metrics))
			{
				continue;
			}
			lastMetrics = metrics;
			break;
		}

		if (lastMetrics == null)
		{
			return null;
		}

		float? result = null;

		for (; metricsIndex >= 0; --metricsIndex)
		{
			var metrics = receivedMetricsList[metricsIndex];
			if (metrics.frameTime < lastMetrics.frameTime - metricsHistoryDuration)
			{
				break;
			}
			if (validalityFieldInfo != null && !(bool)validalityFieldInfo.GetValue(metrics))
			{
				continue;
			}
			else
			{
				float value = (float)baseFieldInfo.GetValue(metrics);
				if (!result.HasValue || result.Value < value)
				{
					result = value;
				}
			}
		}

		return result;
	}

	void PresentFloatPercentage(string label, string propertyName, bool displayAverage, bool displayMaximum)
	{
		float? lastValue = GetLatestPerfValueFloat(propertyName);
		if (!lastValue.HasValue)
		{
			return;
		}

		EditorGUILayout.BeginHorizontal();
		EditorGUILayout.LabelField(label, GUILayout.Width(labelWidth));

		Rect r = EditorGUILayout.BeginVertical();

		float barWidth = (r.width - panelInnerRightPad - progressBarPad * 2) / 3.0f;
		EditorGUI.ProgressBar(new Rect(r.x, r.y, barWidth, r.height), lastValue.Value, string.Format("{0:F1}%", lastValue.Value * 100.0f));

		if (displayAverage)
		{
			float? averageValue = GetAveragePerfValueFloat(propertyName);
			if (averageValue.HasValue)
			{
				EditorGUI.ProgressBar(new Rect(r.x + barWidth + progressBarPad, r.y, barWidth, r.height), averageValue.Value, string.Format("{0:F1}%", averageValue.Value * 100.0f));
			}
		}

		if (displayMaximum)
		{
			float? maxValue = GetMaxPerfValueFloat(propertyName);
			if (maxValue.HasValue)
			{
				EditorGUI.ProgressBar(new Rect(r.x + (barWidth + progressBarPad) * 2, r.y, barWidth, r.height), maxValue.Value, string.Format("{0:F1}%", maxValue.Value * 100.0f));
			}
		}

		GUILayout.Space(progressBarHeight);

		EditorGUILayout.EndVertical();

		EditorGUILayout.EndHorizontal();
	}

	void PresentFloatTimeInMs(string label, string propertyName, float maxScale, bool displayAverage, bool displayMaximum)
	{
		float? lastValue = GetLatestPerfValueFloat(propertyName);
		if (!lastValue.HasValue)
		{
			return;
		}

		EditorGUILayout.BeginHorizontal();
		EditorGUILayout.LabelField(label, GUILayout.Width(labelWidth));

		Rect r = EditorGUILayout.BeginVertical();

		float barWidth = (r.width - panelInnerRightPad - progressBarPad * 2) / 3.0f;
		EditorGUI.ProgressBar(new Rect(r.x, r.y, barWidth, r.height), lastValue.Value/maxScale, string.Format("{0:F1} ms", lastValue.Value * 1000.0f));

		if (displayAverage)
		{
			float? averageValue = GetAveragePerfValueFloat(propertyName);
			if (averageValue.HasValue)
			{
				EditorGUI.ProgressBar(new Rect(r.x + barWidth + progressBarPad, r.y, barWidth, r.height), averageValue.Value / maxScale, string.Format("{0:F1} ms", averageValue.Value * 1000.0f));
			}
		}

		if (displayMaximum)
		{
			float? maxValue = GetMaxPerfValueFloat(propertyName);
			if (maxValue.HasValue)
			{
				EditorGUI.ProgressBar(new Rect(r.x + (barWidth + progressBarPad) * 2, r.y, barWidth, r.height), maxValue.Value / maxScale, string.Format("{0:F1} ms", maxValue.Value * 1000.0f));
			}
		}

		GUILayout.Space(progressBarHeight);

		EditorGUILayout.EndVertical();

		EditorGUILayout.EndHorizontal();
	}

	void PresentIntProperty(string label, string propertyName)
	{
		int? lastValue = GetLatestPerfValueInt(propertyName);
		if (!lastValue.HasValue)
		{
			return;
		}

		PresentText(label, lastValue.Value.ToString());
	}

	void PresentText(string label, string text)
	{
		EditorGUILayout.BeginHorizontal();
		EditorGUILayout.LabelField(label, GUILayout.Width(labelWidth));
		EditorGUILayout.LabelField(text);
		EditorGUILayout.EndHorizontal();
	}

	void PresentColumnTitles(string title0, string title1, string title2)
	{
		EditorGUILayout.BeginHorizontal();
		EditorGUILayout.LabelField("", GUILayout.Width(labelWidth));

		float windowWidth = position.width;
		float barWidth = (windowWidth - labelWidth - panelInnerRightPad * 3 ) / 3.0f;
		EditorGUILayout.LabelField(title0, GUILayout.Width(barWidth));
		EditorGUILayout.LabelField(title1, GUILayout.Width(barWidth));
		EditorGUILayout.LabelField(title2, GUILayout.Width(barWidth));

		EditorGUILayout.EndHorizontal();
	}

	// Called as the new window is opened.
	private void Awake()
	{
		InitializeAndroidSdkPath();
		minSize = new Vector2(400, 300);
	}

	void InitializeAndroidSdkPath()
	{
		androidSdkRootPath = OVRConfig.Instance.GetAndroidSDKPath();
	}

	// OnDestroy is called to close the EditorWindow window.
	private void OnDestroy()
	{
		DisconnectPerfMetricsTcpServer();
	}

	// Called multiple times per second on all visible windows.
	private void Update()
	{
		if (tcpClient != null && tcpClient.Connected)
		{
			tcpClient.Tick();
		}

		if (repaintRequested)
		{
			Repaint();
			repaintRequested = false;
		}
	}

	void OnConnectionStateChanged()
	{
		repaintRequested = true;

		if (tcpClient.connectionState == OVRNetwork.OVRNetworkTcpClient.ConnectionState.Disconnected)
		{
			tcpClient.connectionStateChangedCallback -= OnConnectionStateChanged;
			tcpClient.payloadReceivedCallback -= OnPayloadReceived;
		}
	}

	void OnPayloadReceived(int payloadType, byte[] buffer, int start, int length)
	{
		if (payloadType == OVRSystemPerfMetrics.PayloadTypeMetrics)
		{
			string message = Encoding.UTF8.GetString(buffer, start, length);
			OnMessageReceived(message);
		}
		else
		{
			Debug.LogWarningFormat("[OVRSystemProfilerPanel] unrecongized payload type {0}", payloadType);
		}
	}

	void OnMessageReceived(string message)
	{
		if (pauseReceiveMetrics)
		{
			return;
		}

		var metrics = new OVRSystemPerfMetrics.PerfMetrics();
		if (!metrics.LoadFromJSON(message))
		{
			Debug.LogWarning("Cannot analyze metrics: " + message);
			return;
		}
		lock(receivedMetricsList)
		{
			if (receivedMetricsList.Count >= maxMetricsFrames)
			{
				receivedMetricsList.RemoveAt(0);
			}
			receivedMetricsList.Add(metrics);
		}
		repaintRequested = true;
	}

	void ConnectPerfMetricsTcpServer()
	{
		tcpClient.connectionStateChangedCallback += OnConnectionStateChanged;
		tcpClient.payloadReceivedCallback += OnPayloadReceived;

		tcpClient.Connect(remoteListeningPort);

		EditorApplication.playModeStateChanged += OnApplicationPlayModeStateChanged;
	}

	void DisconnectPerfMetricsTcpServer()
	{
		EditorApplication.playModeStateChanged -= OnApplicationPlayModeStateChanged;

		tcpClient.Disconnect();
	}

	void OnApplicationPlayModeStateChanged(PlayModeStateChange change)
	{
		Debug.LogFormat("[OVRSystemPerfMetricsWindow] OnApplicationPlayModeStateChanged {0}", change.ToString());
		if (change == PlayModeStateChange.ExitingPlayMode)
		{
			tcpClient.Disconnect();
		}
	}

}
