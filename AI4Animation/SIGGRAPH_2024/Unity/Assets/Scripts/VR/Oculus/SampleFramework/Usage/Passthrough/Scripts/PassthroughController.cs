using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PassthroughController : MonoBehaviour
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
        float colorHSV = Time.time * 0.1f;
        colorHSV %= 1; // make sure value is normalized (0-1) so Color.HSVToRGB functions correctly
        Color edgeColor = Color.HSVToRGB(colorHSV, 1, 1);
        passthroughLayer.edgeColor = edgeColor;

        float contrastRange = Mathf.Sin(Time.time); // returns a value -1...1, ideal range for contrast
        passthroughLayer.SetColorMapControls(contrastRange);

        transform.position = Camera.main.transform.position;
        transform.rotation = Quaternion.LookRotation(new Vector3(Camera.main.transform.forward.x, 0.0f, Camera.main.transform.forward.z).normalized);
    }
}
