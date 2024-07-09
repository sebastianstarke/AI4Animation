using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PassthroughProjectionSurface : MonoBehaviour
{
    private OVRPassthroughLayer passthroughLayer;
    public MeshFilter projectionObject;
    MeshRenderer quadOutline;

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

        passthroughLayer.AddSurfaceGeometry(projectionObject.gameObject, true);

        // The MeshRenderer component renders the quad as a blue outline
        // we only use this when Passthrough isn't visible
        quadOutline = projectionObject.GetComponent<MeshRenderer>();
        quadOutline.enabled = false;
    }

    void Update()
    {
        // Hide object when A button is held, show it again when button is released, move it while held.
        if (OVRInput.GetDown(OVRInput.Button.One))
        {
            passthroughLayer.RemoveSurfaceGeometry(projectionObject.gameObject);
            quadOutline.enabled = true;
        }
        if (OVRInput.Get(OVRInput.Button.One))
        {
            OVRInput.Controller controllingHand = OVRInput.Controller.RTouch;
            transform.position = OVRInput.GetLocalControllerPosition(controllingHand);
            transform.rotation = OVRInput.GetLocalControllerRotation(controllingHand);
        }
        if (OVRInput.GetUp(OVRInput.Button.One))
        {
            passthroughLayer.AddSurfaceGeometry(projectionObject.gameObject);
            quadOutline.enabled = false;
        }
    }
}
