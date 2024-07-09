#if OVR_UNITY_ASSET_STORE

#if USING_XR_MANAGEMENT && (USING_XR_SDK_OCULUS || USING_XR_SDK_OPENXR)
#define USING_XR_SDK
#endif

#if UNITY_2020_1_OR_NEWER
#define REQUIRES_XR_SDK
#endif

using System.IO;
using UnityEngine;
using UnityEditor;
using System;

public class MetaXRSimulatorEnabler : MonoBehaviour
{
	const string OpenXrRuntimeEnvKey = "XR_RUNTIME_JSON";
	const string PreviousOpenXrRuntimeEnvKey = "__PREVIOUS_XR_RUNTIME_JSON";

	private static string GetSimulatorJsonPath()
	{
		string rootPath = OVRPluginUpdater.GetEnabledUtilsPluginRootPath();
		if (!string.IsNullOrEmpty(rootPath))
		{
			return rootPath + "\\MetaXRSimulator\\SIMULATOR.json";
		}
		else
		{
			return null;
		}
	}

	private static string GetSimulatorDllPath()
	{
		string rootPath = OVRPluginUpdater.GetEnabledUtilsPluginRootPath();
		if (!string.IsNullOrEmpty(rootPath))
		{
			return rootPath + "\\MetaXRSimulator\\SIMULATOR.dll";
		}
		else
		{
			return null;
		}
	}

	private static string GetCurrentProjectPath()
	{
		return Directory.GetParent(Application.dataPath).FullName;
	}

	private static bool HasSimulatorInstalled()
	{
		string simulatorJsonPath = GetSimulatorJsonPath();
		string simulatorDllPath = GetSimulatorDllPath();

		if (!string.IsNullOrEmpty(simulatorJsonPath) &&
			!string.IsNullOrEmpty(simulatorDllPath) &&
			File.Exists(simulatorJsonPath) &&
			File.Exists(simulatorDllPath))
		{
			return true;
		}

		return false;
	}

	private static bool IsSimulatorActivated()
	{
		return Environment.GetEnvironmentVariable(OpenXrRuntimeEnvKey) == GetSimulatorJsonPath();
	}

	const string kActivateSimulator = "Oculus/Meta XR Simulator (Experimental)/Activate";
	[MenuItem(kActivateSimulator, true, 0)]
	private static bool ValidateSimulatorActivated()
	{
		bool checkMenuItem = HasSimulatorInstalled() && IsSimulatorActivated();
		Menu.SetChecked(kActivateSimulator, checkMenuItem);
		return true;
	}

	[MenuItem(kActivateSimulator, false, 0)]
	private static void ActivateSimulator()
	{
		if (!HasSimulatorInstalled())
		{
			EditorUtility.DisplayDialog("Meta XR Simulator Not Found", "SIMULATOR.json is not found. Please enable OVRPlugin through Oculus/Tools/OVR Utilities Plugin/Set OVRPlugin To OpenXR", "Ok");
			return;
		}

		if (IsSimulatorActivated())
		{
			EditorUtility.DisplayDialog("Meta XR Simulator", "Meta XR Simulator is already activated.", "Ok");
			return;
		}

		Environment.SetEnvironmentVariable(PreviousOpenXrRuntimeEnvKey, Environment.GetEnvironmentVariable(OpenXrRuntimeEnvKey));
		Environment.SetEnvironmentVariable(OpenXrRuntimeEnvKey, GetSimulatorJsonPath());
		UnityEngine.Debug.LogFormat("Meta XR Simulator is activated. OpenXR Runtime set to [{0}]", Environment.GetEnvironmentVariable(OpenXrRuntimeEnvKey));
		EditorUtility.DisplayDialog("Meta XR Simulator", "Meta XR Simulator is activated.", "Ok");
	}

	const string kDeactivateSimulator = "Oculus/Meta XR Simulator (Experimental)/Deactivate";
	[MenuItem(kDeactivateSimulator, true, 1)]
	private static bool ValidateSimulatorDeactivated()
	{
		bool checkMenuItem = !HasSimulatorInstalled() || !IsSimulatorActivated();
		Menu.SetChecked(kDeactivateSimulator, checkMenuItem);
		return true;
	}

	[MenuItem(kDeactivateSimulator, false, 1)]
	private static void DeactivateSimulator()
	{
		if (!HasSimulatorInstalled())
		{
			EditorUtility.DisplayDialog("Meta XR Simulator Not Found", "SIMULATOR.json is not found. Please enable OVRPlugin through Oculus/Tools/OVR Utilities Plugin/Set OVRPlugin To OpenXR", "Ok");
		}

		if (!IsSimulatorActivated())
		{
			EditorUtility.DisplayDialog("Meta XR Simulator", "Meta XR Simulator is not activated.", "Ok");
			return;
		}

		Environment.SetEnvironmentVariable(OpenXrRuntimeEnvKey, Environment.GetEnvironmentVariable(PreviousOpenXrRuntimeEnvKey));
		Environment.SetEnvironmentVariable(PreviousOpenXrRuntimeEnvKey, "");
		UnityEngine.Debug.LogFormat("Meta XR Simulator is deactivated. OpenXR Runtime set to [{0}]", Environment.GetEnvironmentVariable(OpenXrRuntimeEnvKey));
		EditorUtility.DisplayDialog("Meta XR Simulator", "Meta XR Simulator is deactivated.", "Ok");
	}
}

#endif // #if OVR_UNITY_ASSET_STORE
