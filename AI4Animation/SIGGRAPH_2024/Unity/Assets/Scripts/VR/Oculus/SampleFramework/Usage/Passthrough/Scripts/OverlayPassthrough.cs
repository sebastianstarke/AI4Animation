using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OverlayPassthrough : MonoBehaviour
{
    OVRPassthroughLayer passthroughLayer;

    void Start()
    {
        GameObject ovrCameraRig = GameObject.Find("OVRCameraRig");
        if (ovrCameraRig == null)
        {
            Debug.LogError("Scene does not contain an OVRCameraRig");
            return;
        }

        passthroughLayer = ovrCameraRig.GetComponent<OVRPassthroughLayer>();
        if (passthroughLayer == null)
        {
            Debug.LogError("OVRCameraRig does not contain an OVRPassthroughLayer component");
        }
    }

    void Update()
    {
        if (OVRInput.GetDown(OVRInput.Button.One, OVRInput.Controller.RTouch))
        {
            passthroughLayer.hidden = !passthroughLayer.hidden;
        }

        float thumbstickX = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, OVRInput.Controller.RTouch).x;
        passthroughLayer.textureOpacity = thumbstickX * 0.5f + 0.5f;
    }
}
