using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GrabObject : MonoBehaviour
{
    [TextArea]
    public string ObjectName;
    [TextArea]
    public string ObjectInstructions;
    public enum ManipulationType
    {
        Default,
        ForcedHand,
        DollyHand,
        DollyAttached,
        HorizontalScaled,
        VerticalScaled,
        Menu
    };
    public ManipulationType objectManipulationType = ManipulationType.Default;
    public bool showLaserWhileGrabbed = false;
    [HideInInspector]
    public Quaternion grabbedRotation = Quaternion.identity;

    // only handle grab/release
    // other button input is handled by another script on the object
    public delegate void GrabbedObject(OVRInput.Controller grabHand);
    public GrabbedObject GrabbedObjectDelegate;

    public delegate void ReleasedObject();
    public ReleasedObject ReleasedObjectDelegate;

    public delegate void SetCursorPosition(Vector3 cursorPosition);
    public SetCursorPosition CursorPositionDelegate;

    public void Grab(OVRInput.Controller grabHand)
    {
        grabbedRotation = transform.rotation;
        GrabbedObjectDelegate?.Invoke(grabHand);
    }

    public void Release()
    {
        ReleasedObjectDelegate?.Invoke();
    }

    public void CursorPos(Vector3 cursorPos)
    {
        CursorPositionDelegate?.Invoke(cursorPos);
    }
}
