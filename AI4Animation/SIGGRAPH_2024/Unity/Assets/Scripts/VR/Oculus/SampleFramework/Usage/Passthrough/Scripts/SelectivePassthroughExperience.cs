using UnityEngine;

public class SelectivePassthroughExperience : MonoBehaviour
{
    public GameObject leftMaskObject;
    public GameObject rightMaskObject;

    void Update()
    {
        Camera.main.depthTextureMode = DepthTextureMode.Depth;

        bool controllersActive = (OVRInput.GetActiveController() == OVRInput.Controller.LTouch ||
          OVRInput.GetActiveController() == OVRInput.Controller.RTouch ||
          OVRInput.GetActiveController() == OVRInput.Controller.Touch);

        leftMaskObject.SetActive(controllersActive);
        rightMaskObject.SetActive(controllersActive);

        // controller masks are giant circles attached to controllers
        if (controllersActive)
        {
            Vector3 Lpos = OVRInput.GetLocalControllerPosition(OVRInput.Controller.LTouch) + OVRInput.GetLocalControllerRotation(OVRInput.Controller.LTouch) * Vector3.forward * 0.1f;
            Vector3 Rpos = OVRInput.GetLocalControllerPosition(OVRInput.Controller.RTouch) + OVRInput.GetLocalControllerRotation(OVRInput.Controller.RTouch) * Vector3.forward * 0.1f;

            leftMaskObject.transform.position = Lpos;
            rightMaskObject.transform.position = Rpos;
        }
        // hand masks are an inflated hands shader, with alpha fading at wrists and edges
        else if (OVRInput.GetActiveController() == OVRInput.Controller.LHand ||
          OVRInput.GetActiveController() == OVRInput.Controller.RHand ||
          OVRInput.GetActiveController() == OVRInput.Controller.Hands)
        {

        }
    }
}
