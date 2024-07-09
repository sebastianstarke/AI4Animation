using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(OVRCameraRig))]
public class OvrXRRig : XRRig
{   
    // OVR XR Rig should be added to the OVRCamerRig. We have to manually 
    // add disableEyeAnchorCameras or it will force the camera to be enabled. 
    void OnEnable()
    {
        GetComponent<OVRCameraRig>().disableEyeAnchorCameras = false;
    }

    void OnDisable()
    {
        GetComponent<OVRCameraRig>().disableEyeAnchorCameras = true;
    }
}
