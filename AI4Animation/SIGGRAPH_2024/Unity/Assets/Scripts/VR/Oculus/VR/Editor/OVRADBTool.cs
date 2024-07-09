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
using System.Text;
using System.Threading;
using UnityEngine;

using Debug = UnityEngine.Debug;

public class OVRADBTool
{
	public bool isReady;

	public string androidSdkRoot;
	public string androidPlatformToolsPath;
	public string adbPath;

	public OVRADBTool(string androidSdkRoot)
	{
		if (!String.IsNullOrEmpty(androidSdkRoot))
		{
			this.androidSdkRoot = androidSdkRoot;
		}
		else
		{
			this.androidSdkRoot = String.Empty;
		}

		if (this.androidSdkRoot.EndsWith("\\") || this.androidSdkRoot.EndsWith("/"))
		{
			this.androidSdkRoot = this.androidSdkRoot.Remove(this.androidSdkRoot.Length - 1);
		}
		androidPlatformToolsPath = Path.Combine(this.androidSdkRoot, "platform-tools");
		adbPath = Path.Combine(androidPlatformToolsPath, "adb.exe");
		isReady = File.Exists(adbPath);
	}

	public static bool IsAndroidSdkRootValid(string androidSdkRoot)
	{
		OVRADBTool tool = new OVRADBTool(androidSdkRoot);
		return tool.isReady;
	}

	public delegate void WaitingProcessToExitCallback();

	public int StartServer(WaitingProcessToExitCallback waitingProcessToExitCallback)
	{
		string outputString;
		string errorString;

		int exitCode = RunCommand(new string[] { "start-server" }, waitingProcessToExitCallback, out outputString, out errorString);
		return exitCode;
	}

	public int KillServer(WaitingProcessToExitCallback waitingProcessToExitCallback)
	{
		string outputString;
		string errorString;

		int exitCode = RunCommand(new string[] { "kill-server" }, waitingProcessToExitCallback, out outputString, out errorString);
		return exitCode;
	}

	public int ForwardPort(int port, WaitingProcessToExitCallback waitingProcessToExitCallback)
	{
		string outputString;
		string errorString;

		string portString = string.Format("tcp:{0}", port);

		int exitCode = RunCommand(new string[] { "forward", portString, portString }, waitingProcessToExitCallback, out outputString, out errorString);
		return exitCode;
	}

	public int ReleasePort(int port, WaitingProcessToExitCallback waitingProcessToExitCallback)
	{
		string outputString;
		string errorString;

		string portString = string.Format("tcp:{0}", port);

		int exitCode = RunCommand(new string[] { "forward", "--remove", portString }, waitingProcessToExitCallback, out outputString, out errorString);
		return exitCode;
	}

	public List<string> GetDevices()
	{
		string outputString;
		string errorString;

		RunCommand(new string[] { "devices" }, null, out outputString, out errorString);
		string[] devices = outputString.Split(new string[] { "\r\n" }, StringSplitOptions.RemoveEmptyEntries);

		List<string> deviceList = new List<string>(devices);
		deviceList.RemoveAt(0);

		for(int i = 0; i < deviceList.Count; i++)
		{
			string deviceName = deviceList[i];
			int index = deviceName.IndexOf('\t');
			if (index >= 0)
				deviceList[i] = deviceName.Substring(0, index);
			else
				deviceList[i] = "";

		}

		return deviceList;
	}

	private StringBuilder outputStringBuilder = null;
	private StringBuilder errorStringBuilder = null;

	public int RunCommand(string[] arguments, WaitingProcessToExitCallback waitingProcessToExitCallback, out string outputString, out string errorString)
	{
		int exitCode = -1;

		if (!isReady)
		{
			Debug.LogWarning("OVRADBTool not ready");
			outputString = string.Empty;
			errorString = "OVRADBTool not ready";
			return exitCode;
		}

		string args = string.Join(" ", arguments);

		ProcessStartInfo startInfo = new ProcessStartInfo(adbPath, args);
		startInfo.WorkingDirectory = androidSdkRoot;
		startInfo.CreateNoWindow = true;
		startInfo.UseShellExecute = false;
		startInfo.WindowStyle = ProcessWindowStyle.Hidden;
		startInfo.RedirectStandardOutput = true;
		startInfo.RedirectStandardError = true;

		outputStringBuilder = new StringBuilder("");
		errorStringBuilder = new StringBuilder("");

		Process process = Process.Start(startInfo);
		process.OutputDataReceived += new DataReceivedEventHandler(OutputDataReceivedHandler);
		process.ErrorDataReceived += new DataReceivedEventHandler(ErrorDataReceivedHandler);

		process.BeginOutputReadLine();
		process.BeginErrorReadLine();

		try
		{
			do
			{
				if (waitingProcessToExitCallback != null)
				{
					waitingProcessToExitCallback();
				}
			} while (!process.WaitForExit(100));

			process.WaitForExit();
		}
		catch (Exception e)
		{
			Debug.LogWarningFormat("[OVRADBTool.RunCommand] exception {0}", e.Message);
		}

		exitCode = process.ExitCode;

		process.Close();

		outputString = outputStringBuilder.ToString();
		errorString = errorStringBuilder.ToString();

		outputStringBuilder = null;
		errorStringBuilder = null;

		if (!string.IsNullOrEmpty(errorString))
		{
			if (errorString.Contains("Warning"))
			{
				UnityEngine.Debug.LogWarning("OVRADBTool " + errorString);
			}
			else
			{
				UnityEngine.Debug.LogError("OVRADBTool " + errorString);
			}
		}

		return exitCode;
	}

	public Process RunCommandAsync(string[] arguments, DataReceivedEventHandler outputDataRecievedHandler)
	{
		if (!isReady)
		{
			Debug.LogWarning("OVRADBTool not ready");
			return null;
		}

		string args = string.Join(" ", arguments);

		ProcessStartInfo startInfo = new ProcessStartInfo(adbPath, args);
		startInfo.WorkingDirectory = androidSdkRoot;
		startInfo.CreateNoWindow = true;
		startInfo.UseShellExecute = false;
		startInfo.WindowStyle = ProcessWindowStyle.Hidden;
		startInfo.RedirectStandardOutput = true;
		startInfo.RedirectStandardError = true;

		Process process = Process.Start(startInfo);
		if (outputDataRecievedHandler != null)
		{
			process.OutputDataReceived += new DataReceivedEventHandler(outputDataRecievedHandler);
		}

		process.BeginOutputReadLine();
		process.BeginErrorReadLine();

		return process;
	}

	private void OutputDataReceivedHandler(object sendingProcess, DataReceivedEventArgs args)
	{
		// Collect the sort command output.
		if (!string.IsNullOrEmpty(args.Data))
		{
			// Add the text to the collected output.
			outputStringBuilder.Append(args.Data);
			outputStringBuilder.Append(Environment.NewLine);
		}
	}

	private void ErrorDataReceivedHandler(object sendingProcess, DataReceivedEventArgs args)
	{
		// Collect the sort command output.
		if (!string.IsNullOrEmpty(args.Data))
		{
			// Add the text to the collected output.
			errorStringBuilder.Append(args.Data);
			errorStringBuilder.Append(Environment.NewLine);
		}
	}
}
