// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using UnityEngine;

public class LocalizedHaptics : MonoBehaviour
{
    [Header("Settings")]
    [SerializeField] private OVRInput.Handedness m_handedness = OVRInput.Handedness.LeftHanded;

    private OVRInput.Controller m_controller;

    private void Start()
    {
        m_controller = m_handedness == OVRInput.Handedness.LeftHanded ? OVRInput.Controller.LTouch : OVRInput.Controller.RTouch;
    }

    private void Update()
    {
        // Build vibration for the frame based on device inputs
        float thumbAmp = OVRInput.Get(OVRInput.Axis1D.PrimaryThumbRestForce, m_controller) > 0.5f ? 1f : 0f;
        OVRInput.SetControllerLocalizedVibration(OVRInput.HapticsLocation.Thumb, 0f, thumbAmp, m_controller);

        float indexAmp = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, m_controller) > 0.5f ? 1f : 0f;
        OVRInput.SetControllerLocalizedVibration(OVRInput.HapticsLocation.Index, 0f, indexAmp, m_controller);

        float handAmp = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger, m_controller) > 0.5f ? 1f : 0f;
        OVRInput.SetControllerLocalizedVibration(OVRInput.HapticsLocation.Hand, 0f, handAmp, m_controller);
    }
}
