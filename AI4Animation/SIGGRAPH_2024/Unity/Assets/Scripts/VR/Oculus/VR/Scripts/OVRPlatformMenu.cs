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
using System.Collections;
using System.Collections.Generic;

/// <summary>
/// Shows the Oculus plaform UI.
/// </summary>
public class OVRPlatformMenu : MonoBehaviour
{
	/// <summary>
	/// The key code.
	/// </summary>
	private OVRInput.RawButton inputCode = OVRInput.RawButton.Back;

	public enum eHandler
	{
		ShowConfirmQuit,
		RetreatOneLevel,
	};

	public eHandler shortPressHandler = eHandler.ShowConfirmQuit;

	/// <summary>
	/// Callback to handle short press. Returns true if ConfirmQuit menu should be shown.
	/// </summary>
	public System.Func<bool> OnShortPress;
	private static Stack<string> sceneStack = new Stack<string>();

	enum eBackButtonAction
	{
		NONE,
		SHORT_PRESS
	};

	eBackButtonAction HandleBackButtonState()
	{
		eBackButtonAction action = eBackButtonAction.NONE;

		if (OVRInput.GetDown(inputCode))
		{
			action = eBackButtonAction.SHORT_PRESS;
		}

		return action;
	}

	/// <summary>
	/// Instantiate the cursor timer
	/// </summary>
	void Awake()
	{
		if (shortPressHandler == eHandler.RetreatOneLevel && OnShortPress == null)
			OnShortPress = RetreatOneLevel;

		if (!OVRManager.isHmdPresent)
		{
			enabled = false;
			return;
		}

		sceneStack.Push(UnityEngine.SceneManagement.SceneManager.GetActiveScene().name);
	}

	/// <summary>
	/// Show the confirm quit menu
	/// </summary>
	void ShowConfirmQuitMenu()
	{
#if UNITY_ANDROID && !UNITY_EDITOR
		Debug.Log("[PlatformUI-ConfirmQuit] Showing @ " + Time.time);
		OVRManager.PlatformUIConfirmQuit();
#endif
	}

	/// <summary>
	/// Sample handler for short press which retreats to the previous scene that used OVRPlatformMenu.
	/// </summary>
	private static bool RetreatOneLevel()
	{
		if (sceneStack.Count > 1)
		{
			string parentScene = sceneStack.Pop();
			UnityEngine.SceneManagement.SceneManager.LoadSceneAsync (parentScene);
			return false;
		}

		return true;
	}

	/// <summary>
	/// Tests for long-press and activates global platform menu when detected.
	/// as per the Unity integration doc, the back button responds to "mouse 1" button down/up/etc
	/// </summary>
	void Update()
	{
#if UNITY_ANDROID
		eBackButtonAction action = HandleBackButtonState();
		if (action == eBackButtonAction.SHORT_PRESS)
		{
			if (OnShortPress == null || OnShortPress())
			{
				ShowConfirmQuitMenu();
			}
		}
#endif
	}
}
