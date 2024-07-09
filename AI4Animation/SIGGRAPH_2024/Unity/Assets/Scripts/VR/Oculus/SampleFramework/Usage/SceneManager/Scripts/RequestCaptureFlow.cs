using System;
using UnityEngine;

public class RequestCaptureFlow : MonoBehaviour
{
    public OVRInput.Button RequestCaptureBtn = OVRInput.Button.Two;
    private OVRSceneManager _sceneManager;

    private void Start()
    {
        _sceneManager = FindObjectOfType<OVRSceneManager>();
    }

    // Update is called once per frame
    private void Update()
    {
        if (OVRInput.GetUp(RequestCaptureBtn))
        {
            _sceneManager.RequestSceneCapture();
        }
    }
}
