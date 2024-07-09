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

using System.Collections;
using UnityEngine;
using UnityEngine.UI;

public class OVRTrackedKeyboardSampleControls : MonoBehaviour
{
    public OVRTrackedKeyboard trackedKeyboard;
    public InputField StartingFocusField;
    public Text NameValue;
    public Text ConnectedValue;
    public Text StateValue;
    public Text SelectKeyboardValue;
    public Text TypeValue;
    public Color GoodStateColor = new Color(0.25f, 1, 0.25f, 1);
    public Color BadStateColor = new Color(1, 0.25f, 0.25f, 1);
    public Toggle TrackingToggle;
    public Toggle ConnectionToggle;
    public Toggle RemoteKeyboardToggle;
    public Button[] ShaderButtons;

    void Start()
    {
        StartingFocusField.Select();
        StartingFocusField.ActivateInputField();
        if(TrackingToggle.isOn != trackedKeyboard.TrackingEnabled){
            TrackingToggle.isOn = trackedKeyboard.TrackingEnabled;
        }
        if (ConnectionToggle.isOn != trackedKeyboard.ConnectionRequired)
        {
            ConnectionToggle.isOn = trackedKeyboard.ConnectionRequired;
        }

        if (RemoteKeyboardToggle.isOn != trackedKeyboard.RemoteKeyboard) {
            RemoteKeyboardToggle.isOn = trackedKeyboard.RemoteKeyboard;
        }

    }

    void Update()
    {
        NameValue.text = trackedKeyboard.SystemKeyboardInfo.Name;
        ConnectedValue.text = ((bool)((trackedKeyboard.SystemKeyboardInfo.KeyboardFlags & OVRPlugin.TrackedKeyboardFlags.Connected) > 0)).ToString();
        StateValue.text = trackedKeyboard.TrackingState.ToString();
        SelectKeyboardValue.text = "Select " + trackedKeyboard.KeyboardQueryFlags.ToString() + " Keyboard";
        TypeValue.text = trackedKeyboard.KeyboardQueryFlags.ToString();
        switch (trackedKeyboard.TrackingState)
        {
            case OVRTrackedKeyboard.TrackedKeyboardState.Uninitialized:
            case OVRTrackedKeyboard.TrackedKeyboardState.Error:
            case OVRTrackedKeyboard.TrackedKeyboardState.ErrorExtensionFailed:
            case OVRTrackedKeyboard.TrackedKeyboardState.StartedNotTracked:
            case OVRTrackedKeyboard.TrackedKeyboardState.Stale:
                StateValue.color = BadStateColor;
                break;
            default:
                StateValue.color = GoodStateColor;
                break;
        }

    }

    public void SetPresentationOpaque()
    {
        trackedKeyboard.Presentation = OVRTrackedKeyboard.KeyboardPresentation.PreferOpaque;
    }

    public void SetPresentationKeyLabels()
    {
        trackedKeyboard.Presentation = OVRTrackedKeyboard.KeyboardPresentation.PreferKeyLabels;
    }

    public void SetUnlitShader()
    {
        StartCoroutine(SetShaderCoroutine("Unlit/Texture"));
    }

    public void SetDiffuseShader()
    {
        StartCoroutine(SetShaderCoroutine("Mobile/Diffuse"));
    }

    private IEnumerator SetShaderCoroutine(string shaderName)
    {
        bool trackingWasEnabled = trackedKeyboard.TrackingEnabled;
        trackedKeyboard.TrackingEnabled = false;
        yield return new WaitWhile(() => trackedKeyboard.TrackingState != OVRTrackedKeyboard.TrackedKeyboardState.Offline);
        trackedKeyboard.keyboardModelShader = Shader.Find(shaderName);
        trackedKeyboard.TrackingEnabled = trackingWasEnabled;
    }

    public void LaunchKeyboardSelection()
    {
        if (trackedKeyboard.RemoteKeyboard)
        {
            trackedKeyboard.LaunchRemoteKeyboardSelectionDialog();
        }
        else
        {
            trackedKeyboard.LaunchLocalKeyboardSelectionDialog();
        }
    }

    public void SetTrackingEnabled(bool value)
    {
        trackedKeyboard.TrackingEnabled = value;
    }
}
