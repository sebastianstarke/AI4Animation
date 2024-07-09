// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using UnityEngine;
using TMPro;

public class UiDeviceInspector : MonoBehaviour
{
    [Header("Settings")]
    [SerializeField] private OVRInput.Handedness m_handedness = OVRInput.Handedness.LeftHanded;

    [Header("Left Column Components")]
    [SerializeField] private TextMeshProUGUI m_title;
    [SerializeField] private TextMeshProUGUI m_status;

    [SerializeField] private UiBoolInspector m_thumbRestTouch;
    [SerializeField] private UiAxis1dInspector m_thumbRestForce;
    [SerializeField] private UiAxis1dInspector m_indexTrigger;
    [SerializeField] private UiAxis1dInspector m_gripTrigger;
    [SerializeField] private UiAxis1dInspector m_stylusTipForce;

    [SerializeField] private UiAxis1dInspector m_indexCurl1d;
    [SerializeField] private UiAxis1dInspector m_indexSlider1d;

    [Header("Right Column Components")]
    [SerializeField] private UiBoolInspector m_ax;
    [SerializeField] private UiBoolInspector m_axTouch;
    [SerializeField] private UiBoolInspector m_by;
    [SerializeField] private UiBoolInspector m_byTouch;
    [SerializeField] private UiBoolInspector m_indexTouch;

    [SerializeField] private UiAxis2dInspector m_thumbstick;

    private OVRInput.Controller m_controller;

    private void Start()
    {
        m_controller = m_handedness == OVRInput.Handedness.LeftHanded ? OVRInput.Controller.LTouch : OVRInput.Controller.RTouch;
    }

    private void Update()
    {
        // Set device title
        string deviceTitle = $"{ToDeviceModel()} [{ToHandednessString(m_handedness)}]";
        m_title.SetText(deviceTitle);

        // Set status flags
        string connectionState = OVRInput.IsControllerConnected(m_controller) ? "<color=#66ff87>o</color>" : "<color=#ff8991>x</color>";
        bool isTracked = OVRInput.GetControllerOrientationTracked(m_controller) && OVRInput.GetControllerPositionTracked(m_controller);
        string trackingState = isTracked ? "<color=#66ff87>o</color>" : "<color=#ff8991>x</color>";
        m_status.SetText($"Connected [{connectionState}] Tracked [{trackingState}]");

        // ThumbRest force and triggers
        m_thumbRestTouch.SetValue(OVRInput.Get(OVRInput.Touch.PrimaryThumbRest, m_controller));
        m_indexTrigger.SetValue(OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, m_controller));
        m_gripTrigger.SetValue(OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger, m_controller));

        m_thumbRestForce.SetValue(OVRInput.Get(OVRInput.Axis1D.PrimaryThumbRestForce, m_controller));

        // Stylus tip
        m_stylusTipForce.SetValue(OVRInput.Get(OVRInput.Axis1D.PrimaryStylusForce, m_controller));

        // Index capsense
        m_indexCurl1d.SetValue(OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTriggerCurl, m_controller));
        m_indexSlider1d.SetValue(OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTriggerSlide, m_controller));

        // Face buttons
        m_ax.SetValue(OVRInput.Get(OVRInput.Button.One, m_controller));
        m_axTouch.SetValue(OVRInput.Get(OVRInput.Touch.One, m_controller));
        m_by.SetValue(OVRInput.Get(OVRInput.Button.Two, m_controller));
        m_byTouch.SetValue(OVRInput.Get(OVRInput.Touch.Two, m_controller)); ;

        // Index touch
        m_indexTouch.SetValue(OVRInput.Get(OVRInput.Touch.PrimaryIndexTrigger, m_controller));

        // Thumbstick position & touch
        m_thumbstick.SetValue(OVRInput.Get(OVRInput.Touch.PrimaryThumbstick, m_controller), OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, m_controller));
    }

    private static string ToDeviceModel()
    {
        return "Touch";
    }

    private static string ToHandednessString(OVRInput.Handedness handedness)
    {
        switch (handedness)
        {
            case OVRInput.Handedness.LeftHanded:
                return "L";

            case OVRInput.Handedness.RightHanded:
                return "R";
        }
        return "-";
    }
}
