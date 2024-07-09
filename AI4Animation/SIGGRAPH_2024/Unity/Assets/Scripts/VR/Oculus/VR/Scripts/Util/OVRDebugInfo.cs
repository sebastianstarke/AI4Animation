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
using UnityEngine.UI;

//-------------------------------------------------------------------------------------
/// <summary>
/// Shows debug information on a heads-up display.
/// </summary>
public class OVRDebugInfo : MonoBehaviour
{
    #region GameObjects for Debug Information UIs
    GameObject debugUIManager;
    GameObject debugUIObject;
    GameObject riftPresent;
    GameObject fps;
    GameObject ipd;
    GameObject fov;
    GameObject height;
	GameObject depth;
	GameObject resolutionEyeTexture;
    GameObject latencies;
    GameObject texts;
    #endregion

    #region Debug strings
	string strRiftPresent            = null; // "VR DISABLED"
    string strFPS                    = null; // "FPS: 0";
    string strIPD                    = null; // "IPD: 0.000";
    string strFOV                    = null; // "FOV: 0.0f";
    string strHeight                 = null; // "Height: 0.0f";
	string strDepth                  = null; // "Depth: 0.0f";
	string strResolutionEyeTexture   = null; // "Resolution : {0} x {1}"
    string strLatencies              = null; // "R: {0:F3} TW: {1:F3} PP: {2:F3} RE: {3:F3} TWE: {4:F3}"
    #endregion

    /// <summary>
    /// Variables for FPS
    /// </summary>
    float updateInterval = 0.5f;
    float accum          = 0.0f;
    int   frames         = 0;
    float timeLeft       = 0.0f;

    /// <summary>
    /// Managing for UI initialization
    /// </summary>
    bool  initUIComponent = false;
    bool  isInited        = false;

    /// <summary>
    /// UIs Y offset
    /// </summary>
    float offsetY = 55.0f;

    /// <summary>
    /// Managing for rift detection UI
    /// </summary>
    float riftPresentTimeout = 0.0f;

    /// <summary>
    /// Turn on / off VR variables
    /// </summary>
    bool showVRVars = false;

    #region MonoBehaviour handler

    /// <summary>
    /// Initialization
    /// </summary>
    void Awake()
    {
        // Create canvas for using new GUI
        debugUIManager = new GameObject();
        debugUIManager.name = "DebugUIManager";
        debugUIManager.transform.parent = GameObject.Find("LeftEyeAnchor").transform;

        RectTransform rectTransform = debugUIManager.AddComponent<RectTransform>();
        rectTransform.sizeDelta = new Vector2(100f, 100f);
        rectTransform.localScale = new Vector3(0.001f, 0.001f, 0.001f);
        rectTransform.localPosition = new Vector3(0.01f, 0.17f, 0.53f);
        rectTransform.localEulerAngles = Vector3.zero;

        Canvas canvas = debugUIManager.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        canvas.pixelPerfect = false;
    }

    /// <summary>
    /// Updating VR variables and managing UI present
    /// </summary>
    void Update()
    {
        if (initUIComponent && !isInited)
        {
            InitUIComponents();
        }

		//todo: enable for Unity Input System
#if ENABLE_LEGACY_INPUT_MANAGER
        if (Input.GetKeyDown(KeyCode.Space) && riftPresentTimeout < 0.0f)
        {
            initUIComponent = true;
            showVRVars ^= true;
        }
#endif

		UpdateDeviceDetection();

        // Presenting VR variables
        if (showVRVars)
        {
            debugUIManager.SetActive(true);
            UpdateVariable();
            UpdateStrings();
        }
        else
        {
            debugUIManager.SetActive(false);
        }
    }

    /// <summary>
    /// Initialize isInited value on OnDestroy
    /// </summary>
    void OnDestroy()
    {
        isInited = false;
    }
#endregion

#region Private Functions
    /// <summary>
    /// Initialize UI GameObjects
    /// </summary>
    void InitUIComponents()
    {
        float posY = 0.0f;
        int fontSize = 20;

        debugUIObject = new GameObject();
        debugUIObject.name = "DebugInfo";
        debugUIObject.transform.parent = GameObject.Find("DebugUIManager").transform;
        debugUIObject.transform.localPosition = new Vector3(0.0f, 100.0f, 0.0f);
        debugUIObject.transform.localEulerAngles = Vector3.zero;
        debugUIObject.transform.localScale = new Vector3(1.0f, 1.0f, 1.0f);

        // Print out for FPS
        if (!string.IsNullOrEmpty(strFPS))
        {
            fps = VariableObjectManager(fps, "FPS", posY -= offsetY, strFPS, fontSize);
        }

        // Print out for IPD
        if (!string.IsNullOrEmpty(strIPD))
        {
            ipd = VariableObjectManager(ipd, "IPD", posY -= offsetY, strIPD, fontSize);
        }

        // Print out for FOV
        if (!string.IsNullOrEmpty(strFOV))
        {
            fov = VariableObjectManager(fov, "FOV", posY -= offsetY, strFOV, fontSize);
        }

        // Print out for Height
        if (!string.IsNullOrEmpty(strHeight))
        {
            height = VariableObjectManager(height, "Height", posY -= offsetY, strHeight, fontSize);
        }

		// Print out for Depth
		if (!string.IsNullOrEmpty(strDepth))
		{
			depth = VariableObjectManager(depth, "Depth", posY -= offsetY, strDepth, fontSize);
		}

		// Print out for Resoulution of Eye Texture
        if (!string.IsNullOrEmpty(strResolutionEyeTexture))
        {
            resolutionEyeTexture = VariableObjectManager(resolutionEyeTexture, "Resolution", posY -= offsetY, strResolutionEyeTexture, fontSize);
        }

        // Print out for Latency
        if (!string.IsNullOrEmpty(strLatencies))
        {
            latencies = VariableObjectManager(latencies, "Latency", posY -= offsetY, strLatencies, 17);
            posY = 0.0f;
        }

        initUIComponent = false;
        isInited = true;

    }

    /// <summary>
    /// Update VR Variables
    /// </summary>
    void UpdateVariable()
    {
        UpdateIPD();
        UpdateEyeHeightOffset();
		UpdateEyeDepthOffset();
		UpdateFOV();
        UpdateResolutionEyeTexture();
        UpdateLatencyValues();
        UpdateFPS();
    }

    /// <summary>
    /// Update Strings
    /// </summary>
    void UpdateStrings()
    {
        if (debugUIObject == null)
            return;

        if (!string.IsNullOrEmpty(strFPS))
            fps.GetComponentInChildren<Text>().text = strFPS;
        if (!string.IsNullOrEmpty(strIPD))
            ipd.GetComponentInChildren<Text>().text = strIPD;
        if (!string.IsNullOrEmpty(strFOV))
            fov.GetComponentInChildren<Text>().text = strFOV;
        if (!string.IsNullOrEmpty(strResolutionEyeTexture))
            resolutionEyeTexture.GetComponentInChildren<Text>().text = strResolutionEyeTexture;
        if (!string.IsNullOrEmpty(strLatencies))
		{
            latencies.GetComponentInChildren<Text>().text = strLatencies;
			latencies.GetComponentInChildren<Text>().fontSize = 14;
		}
        if (!string.IsNullOrEmpty(strHeight))
            height.GetComponentInChildren<Text>().text = strHeight;
		if (!string.IsNullOrEmpty(strDepth))
			depth.GetComponentInChildren<Text>().text = strDepth;
	}

	/// <summary>
    /// It's for rift present GUI
    /// </summary>
    void RiftPresentGUI(GameObject guiMainOBj)
    {
        riftPresent = ComponentComposition(riftPresent);
        riftPresent.transform.SetParent(guiMainOBj.transform);
        riftPresent.name = "RiftPresent";
        RectTransform rectTransform = riftPresent.GetComponent<RectTransform>();
        rectTransform.localPosition = new Vector3(0.0f, 0.0f, 0.0f);
        rectTransform.localScale = new Vector3(1.0f, 1.0f, 1.0f);
        rectTransform.localEulerAngles = Vector3.zero;

        Text text = riftPresent.GetComponentInChildren<Text>();
        text.text = strRiftPresent;
        text.fontSize = 20;
    }

    /// <summary>
    /// Updates the device detection.
    /// </summary>
    void UpdateDeviceDetection()
    {
        if (riftPresentTimeout >= 0.0f)
        {
            riftPresentTimeout -= Time.deltaTime;
        }
    }

    /// <summary>
    /// Object Manager for Variables
    /// </summary>
    /// <returns> gameobject for each Variable </returns>
    GameObject VariableObjectManager(GameObject gameObject, string name, float posY, string str, int fontSize)
    {
        gameObject = ComponentComposition(gameObject);
        gameObject.name = name;
        gameObject.transform.SetParent(debugUIObject.transform);

        RectTransform rectTransform = gameObject.GetComponent<RectTransform>();
        rectTransform.localPosition = new Vector3(0.0f, posY -= offsetY, 0.0f);

        Text text = gameObject.GetComponentInChildren<Text>();
        text.text = str;
        text.fontSize = fontSize;
        gameObject.transform.localEulerAngles = Vector3.zero;

        rectTransform.localScale = new Vector3(1.0f, 1.0f, 1.0f);

        return gameObject;
    }

    /// <summary>
    /// Component composition
    /// </summary>
    /// <returns> Composed gameobject. </returns>
    GameObject ComponentComposition(GameObject GO)
    {
        GO = new GameObject();
        GO.AddComponent<RectTransform>();
        GO.AddComponent<CanvasRenderer>();
        GO.AddComponent<Image>();
        GO.GetComponent<RectTransform>().sizeDelta = new Vector2(350f, 50f);
        GO.GetComponent<Image>().color = new Color(7f / 255f, 45f / 255f, 71f / 255f, 200f / 255f);

        texts = new GameObject();
        texts.AddComponent<RectTransform>();
        texts.AddComponent<CanvasRenderer>();
        texts.AddComponent<Text>();
        texts.GetComponent<RectTransform>().sizeDelta = new Vector2(350f, 50f);
		texts.GetComponent<Text>().font = Resources.GetBuiltinResource(typeof(Font), "Arial.ttf") as Font;
        texts.GetComponent<Text>().alignment = TextAnchor.MiddleCenter;

        texts.transform.SetParent(GO.transform);
        texts.name = "TextBox";

        return GO;
    }
#endregion

#region Debugging variables handler
    /// <summary>
    /// Updates the IPD.
    /// </summary>
    void UpdateIPD()
    {
        strIPD = System.String.Format("IPD (mm): {0:F4}", OVRManager.profile.ipd * 1000.0f);
    }

    /// <summary>
    /// Updates the eye height offset.
    /// </summary>
    void UpdateEyeHeightOffset()
    {
        float eyeHeight = OVRManager.profile.eyeHeight;
        strHeight = System.String.Format("Eye Height (m): {0:F3}", eyeHeight);
	}

	/// <summary>
	/// Updates the eye depth offset.
	/// </summary>
	void UpdateEyeDepthOffset()
	{
		float eyeDepth = OVRManager.profile.eyeDepth;
		strDepth = System.String.Format("Eye Depth (m): {0:F3}", eyeDepth);
	}

	/// <summary>
	/// Updates the FOV.
    /// </summary>
    void UpdateFOV()
    {
        OVRDisplay.EyeRenderDesc eyeDesc = OVRManager.display.GetEyeRenderDesc(UnityEngine.XR.XRNode.LeftEye);
        strFOV = System.String.Format("FOV (deg): {0:F3}", eyeDesc.fov.y);
    }

    /// <summary>
    /// Updates resolution of eye texture
    /// </summary>
    void UpdateResolutionEyeTexture()
    {
		OVRDisplay.EyeRenderDesc leftEyeDesc = OVRManager.display.GetEyeRenderDesc(UnityEngine.XR.XRNode.LeftEye);
		OVRDisplay.EyeRenderDesc rightEyeDesc = OVRManager.display.GetEyeRenderDesc(UnityEngine.XR.XRNode.RightEye);

		float scale = UnityEngine.XR.XRSettings.renderViewportScale;
        float w = (int)(scale * (float)(leftEyeDesc.resolution.x + rightEyeDesc.resolution.x));
        float h = (int)(scale * (float)Mathf.Max(leftEyeDesc.resolution.y, rightEyeDesc.resolution.y));

        strResolutionEyeTexture = System.String.Format("Resolution : {0} x {1}", w, h);
    }

    /// <summary>
    /// Updates latency values
    /// </summary>
    void UpdateLatencyValues()
    {
#if !UNITY_ANDROID || UNITY_EDITOR
            OVRDisplay.LatencyData latency = OVRManager.display.latency;
            if (latency.render < 0.000001f && latency.timeWarp < 0.000001f && latency.postPresent < 0.000001f)
                strLatencies = System.String.Format("Latency values are not available.");
            else
                strLatencies = System.String.Format("Render: {0:F3} TimeWarp: {1:F3} Post-Present: {2:F3}\nRender Error: {3:F3} TimeWarp Error: {4:F3}",
                    latency.render,
                    latency.timeWarp,
                    latency.postPresent,
                    latency.renderError,
                    latency.timeWarpError);
#endif
    }

    /// <summary>
    /// Updates the FPS.
    /// </summary>
    void UpdateFPS()
    {
        timeLeft -= Time.unscaledDeltaTime;
        accum += Time.unscaledDeltaTime;
        ++frames;

        // Interval ended - update GUI text and start new interval
        if (timeLeft <= 0.0)
        {
            // display two fractional digits (f2 format)
            float fps = frames / accum;

            strFPS = System.String.Format("FPS: {0:F2}", fps);

            timeLeft += updateInterval;
            accum = 0.0f;
            frames = 0;
        }
    }
#endregion
}
