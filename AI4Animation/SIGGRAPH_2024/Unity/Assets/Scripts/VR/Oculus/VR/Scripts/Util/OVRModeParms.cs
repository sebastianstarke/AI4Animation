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

using UnityEngine;

/// <summary>
/// Logs when the application enters power save mode and allows you to a low-power CPU/GPU level with a button press.
/// </summary>
public class OVRModeParms : MonoBehaviour
{
#region Member Variables

	/// <summary>
	/// The gamepad button that will switch the application to CPU level 0 and GPU level 1.
	/// </summary>
	public OVRInput.RawButton	resetButton = OVRInput.RawButton.X;

#endregion

	/// <summary>
	/// Invoke power state mode test.
	/// </summary>
	void Start()
	{
		if (!OVRManager.isHmdPresent)
		{
			enabled = false;
			return;
		}

		// Call TestPowerLevelState after 10 seconds
		// and repeats every 10 seconds.
		InvokeRepeating ( "TestPowerStateMode", 10, 10.0f );
	}

	/// <summary>
	/// Change default vr mode parms dynamically.
	/// </summary>
	void Update()
	{
		// NOTE: some of the buttons defined in OVRInput.RawButton are not available on the Android game pad controller
		if ( OVRInput.GetDown(resetButton))
		{
			//*************************
			// Dynamically change VrModeParms cpu and gpu level.
			// NOTE: Reset will cause 1 frame of flicker as it leaves
			// and re-enters Vr mode.
			//*************************
			OVRPlugin.suggestedCpuPerfLevel = OVRPlugin.ProcessorPerformanceLevel.PowerSavings;
			OVRPlugin.suggestedGpuPerfLevel = OVRPlugin.ProcessorPerformanceLevel.SustainedLow;
		}
	}

	/// <summary>
	/// Check current power state mode.
	/// </summary>
	void TestPowerStateMode()
	{
		//*************************
		// Check power-level state mode
		//*************************
		if (OVRPlugin.powerSaving)
		{
			// The device has been throttled
			Debug.Log("POWER SAVE MODE ACTIVATED");
		}
	}
}
