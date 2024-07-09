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

/// <summary>
/// Sample that allows you to play with various VR settings.
/// </summary>
public class OVRSceneSampleController : MonoBehaviour
{
    /// <summary>
    /// The key that quits the application.
    /// </summary>
    public KeyCode quitKey = KeyCode.Escape;

    /// <summary>
    /// An optional texture that appears before the menu fades in.
    /// </summary>
    public Texture fadeInTexture = null;

    /// <summary>
    /// Controls how quickly the player's speed and rotation change based on input.
    /// </summary>
    public float speedRotationIncrement = 0.05f;

    private OVRPlayerController playerController = null;

    // Handle to OVRCameraRig
    private OVRCameraRig cameraController = null;

    /// <summary>
    /// We can set the layer to be anything we want to, this allows
    /// a specific camera to render it.
    /// </summary>
    public string layerName = "Default";

    // Vision mode on/off
    private bool visionMode = true;

    // We want to hold onto GridCube, for potential sharing
    // of the menu RenderTarget
    OVRGridCube gridCube = null;

#if	SHOW_DK2_VARIABLES
	private string strVisionMode = "Vision Enabled: ON";
#endif

    #region MonoBehaviour Message Handlers
    /// <summary>
    /// Awake this instance.
    /// </summary>
    void Awake()
    {
        // Find camera controller
        OVRCameraRig[] cameraControllers;
        cameraControllers = gameObject.GetComponentsInChildren<OVRCameraRig>();

        if (cameraControllers.Length == 0)
        {
            Debug.LogWarning("OVRMainMenu: No OVRCameraRig attached.");
        }
        else if (cameraControllers.Length > 1)
        {
            Debug.LogWarning("OVRMainMenu: More then 1 OVRCameraRig attached.");
        }
        else
        {
            cameraController = cameraControllers[0];
        }

        // Find player controller
        OVRPlayerController[] playerControllers;
        playerControllers = gameObject.GetComponentsInChildren<OVRPlayerController>();

        if (playerControllers.Length == 0)
        {
            Debug.LogWarning("OVRMainMenu: No OVRPlayerController attached.");
        }
        else if (playerControllers.Length > 1)
        {
            Debug.LogWarning("OVRMainMenu: More then 1 OVRPlayerController attached.");
        }
        else
        {
            playerController = playerControllers[0];
        }
    }

    /// <summary>
    /// Start this instance.
    /// </summary>
    void Start()
    {
        // Make sure to hide cursor
        if (Application.isEditor == false)
        {
			Cursor.visible = false;
			Cursor.lockState = CursorLockMode.Locked;
        }

        // CameraController updates
        if (cameraController != null)
        {
            // Add a GridCube component to this object
            gridCube = gameObject.AddComponent<OVRGridCube>();
            gridCube.SetOVRCameraController(ref cameraController);
        }
    }


    /// <summary>
    /// Update this instance.
    /// </summary>
    void Update()
    {
        // Recenter pose
        UpdateRecenterPose();

        // Turn On/Off Vision Mode
        UpdateVisionMode();

        // Update Speed and Rotation Scale
        if (playerController != null)
            UpdateSpeedAndRotationScaleMultiplier();

		//todo: enable for Unity Input System
#if ENABLE_LEGACY_INPUT_MANAGER
		// Toggle Fullscreen
		if (Input.GetKeyDown(KeyCode.F11))
            Screen.fullScreen = !Screen.fullScreen;

        if (Input.GetKeyDown(KeyCode.M))
			UnityEngine.XR.XRSettings.showDeviceView = !UnityEngine.XR.XRSettings.showDeviceView;

#if !UNITY_ANDROID || UNITY_EDITOR
        // Escape Application
        if (Input.GetKeyDown(quitKey))
            Application.Quit();
#endif
#endif
    }
#endregion

    /// <summary>
    /// Updates the vision mode.
    /// </summary>
    void UpdateVisionMode()
    {
		//todo: enable for Unity Input System
#if ENABLE_LEGACY_INPUT_MANAGER
		if (Input.GetKeyDown(KeyCode.F2))
        {
            visionMode ^= visionMode;
            OVRManager.tracker.isEnabled = visionMode;
        }
#endif
    }

    /// <summary>
    /// Updates the speed and rotation scale multiplier.
    /// </summary>
    void UpdateSpeedAndRotationScaleMultiplier()
    {
        float moveScaleMultiplier = 0.0f;
        playerController.GetMoveScaleMultiplier(ref moveScaleMultiplier);

		//todo: enable for Unity Input System
#if ENABLE_LEGACY_INPUT_MANAGER
		if (Input.GetKeyDown(KeyCode.Alpha7))
        {
            moveScaleMultiplier -= speedRotationIncrement;
        }
        else if (Input.GetKeyDown(KeyCode.Alpha8))
        {
            moveScaleMultiplier += speedRotationIncrement;
        }
#endif

        playerController.SetMoveScaleMultiplier(moveScaleMultiplier);

        float rotationScaleMultiplier = 0.0f;
        playerController.GetRotationScaleMultiplier(ref rotationScaleMultiplier);

		//todo: enable for Unity Input System
#if ENABLE_LEGACY_INPUT_MANAGER
		if (Input.GetKeyDown(KeyCode.Alpha9))
        {
            rotationScaleMultiplier -= speedRotationIncrement;
        }
        else if (Input.GetKeyDown(KeyCode.Alpha0))
        {
            rotationScaleMultiplier += speedRotationIncrement;
        }
#endif

        playerController.SetRotationScaleMultiplier(rotationScaleMultiplier);
    }

    /// <summary>
    /// Recenter pose
    /// </summary>
    void UpdateRecenterPose()
    {
		//todo: enable for Unity Input System
#if ENABLE_LEGACY_INPUT_MANAGER
		if (Input.GetKeyDown(KeyCode.R))
            OVRManager.display.RecenterPose();
#endif
    }
}
