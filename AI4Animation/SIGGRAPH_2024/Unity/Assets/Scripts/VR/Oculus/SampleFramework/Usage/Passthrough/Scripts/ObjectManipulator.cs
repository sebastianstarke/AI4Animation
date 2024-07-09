using System.Collections;
using UnityEngine;
using UnityEngine.UI;

// grabs any object that has a collider
// adding a GrabObject script to the object offers more functionality
public class ObjectManipulator : MonoBehaviour
{
    OVRInput.Controller controller = OVRInput.Controller.RTouch;
    GameObject hoverObject = null;
    GameObject grabObject = null;
    // all-purpose timer to use for blending after object is grabbed/released
    float grabTime = 0.0f;
    // the grabbed object's transform relative to the controller
    Vector3 localGrabOffset = Vector3.zero;
    Quaternion localGrabRotation = Quaternion.identity;
    // the camera and grabbing hand's world position when grabbing
    Vector3 camGrabPosition = Vector3.zero;
    Quaternion camGrabRotation = Quaternion.identity;
    Vector3 handGrabPosition = Vector3.zero;
    Quaternion handGrabRotation = Quaternion.identity;
    Vector3 cursorPosition = Vector3.zero;
    float rotationOffset = 0.0f;
    public LineRenderer laser;
    public Transform objectInfo;
    public TextMesh objectNameLabel;
    public TextMesh objectInstructionsLabel;
    public Image objectInfoBG;

    // align these in front of the user's view when starting
    public GameObject demoObjects;

    // only used in this script for fading in from black
    public OVRPassthroughLayer passthrough;

    private void Start()
    {
        if (passthrough)
        {
            passthrough.colorMapEditorBrightness = -1;
            passthrough.colorMapEditorContrast = -1;
        }
        StartCoroutine(StartDemo());
        // render these UI elements after the passthrough "hole punch" shader and the brush ring
        if (objectNameLabel) objectNameLabel.font.material.renderQueue = 4600;
        if (objectInstructionsLabel) objectInstructionsLabel.font.material.renderQueue = 4600;
        if (objectInfoBG) objectInfoBG.materialForRendering.renderQueue = 4599;
    }

    void Update()
    {
        Vector3 controllerPos = OVRInput.GetLocalControllerPosition(controller);
        Quaternion controllerRot = OVRInput.GetLocalControllerRotation(controller);

        FindHoverObject(controllerPos, controllerRot);

        if (hoverObject)
        {
            if (OVRInput.GetDown(OVRInput.Button.PrimaryHandTrigger, controller))
            {
                // grabbing
                grabObject = hoverObject;
                GrabHoverObject(grabObject, controllerPos, controllerRot);
            }
        }

        if (grabObject)
        {
            grabTime += Time.deltaTime * 5;
            grabTime = Mathf.Clamp01(grabTime);
            ManipulateObject(grabObject, controllerPos, controllerRot);

            if (!OVRInput.Get(OVRInput.Button.PrimaryHandTrigger, controller))
            {
                ReleaseObject();
            }
        }
        else
        {
            grabTime -= Time.deltaTime * 5;
            grabTime = Mathf.Clamp01(grabTime);
        }
    }

    void GrabHoverObject(GameObject grbObj, Vector3 controllerPos, Quaternion controllerRot)
    {
        localGrabOffset = Quaternion.Inverse(controllerRot) * (grabObject.transform.position - controllerPos);
        localGrabRotation = Quaternion.Inverse(controllerRot) * grabObject.transform.rotation;
        rotationOffset = 0.0f;
        if (grabObject.GetComponent<GrabObject>())
        {
            grabObject.GetComponent<GrabObject>().Grab(controller);
            grabObject.GetComponent<GrabObject>().grabbedRotation = grabObject.transform.rotation;
            AssignInstructions(grabObject.GetComponent<GrabObject>());
        }
        handGrabPosition = controllerPos;
        handGrabRotation = controllerRot;
        camGrabPosition = Camera.main.transform.position;
        camGrabRotation = Camera.main.transform.rotation;
    }

    void ReleaseObject()
    {
        if (grabObject.GetComponent<GrabObject>())
        {
            if (grabObject.GetComponent<GrabObject>().objectManipulationType == GrabObject.ManipulationType.Menu)
            {
                // restore the menu it to its first resting place
                grabObject.transform.position = handGrabPosition + handGrabRotation * localGrabOffset;
                grabObject.transform.rotation = handGrabRotation * localGrabRotation;
            }
            grabObject.GetComponent<GrabObject>().Release();
        }
        grabObject = null;
    }

    // wait for systems to get situated, then spawn the objects in front of them
    IEnumerator StartDemo()
    {
        demoObjects.SetActive(false);
        // fade from black
        float timer = 0.0f;
        float fadeTime = 1.0f;
        while (timer <= fadeTime)
        {
            timer += Time.deltaTime;
            float normTimer = Mathf.Clamp01(timer / fadeTime);
            if (passthrough)
            {
                passthrough.colorMapEditorBrightness = Mathf.Lerp(-1.0f, 0.0f, normTimer);
                passthrough.colorMapEditorContrast = Mathf.Lerp(-1.0f, 0.0f, normTimer);
            }
            yield return null;
        }
        //yield return new WaitForSeconds(1.0f);
        demoObjects.SetActive(true);
        Vector3 objFwd = new Vector3(Camera.main.transform.forward.x, 0, Camera.main.transform.forward.z).normalized;
        demoObjects.transform.position = Camera.main.transform.position + objFwd;
        demoObjects.transform.rotation = Quaternion.LookRotation(objFwd);
    }

    void FindHoverObject(Vector3 controllerPos, Quaternion controllerRot)
    {
        RaycastHit[] objectsHit = Physics.RaycastAll(controllerPos, controllerRot * Vector3.forward);
        float closestObject = Mathf.Infinity;
        float rayDistance = 2.0f;
        bool showLaser = true;
        Vector3 labelPosition = Vector3.zero;
        foreach (RaycastHit hit in objectsHit)
        {
            float thisHitDistance = Vector3.Distance(hit.point, controllerPos);
            if (thisHitDistance < closestObject)
            {
                hoverObject = hit.collider.gameObject;
                closestObject = thisHitDistance;
                rayDistance = grabObject ? thisHitDistance : thisHitDistance - 0.1f;
                labelPosition = hit.point;
            }
        }
        if (objectsHit.Length == 0)
        {
            hoverObject = null;
        }

        // if intersecting with an object, grab it
        Collider[] hitColliders = Physics.OverlapSphere(controllerPos, 0.05f);
        foreach (var hitCollider in hitColliders)
        {
            // use the last object, if there are multiple hits.
            // If objects overlap, this would require improvements.
            hoverObject = hitCollider.gameObject;
            showLaser = false;
            labelPosition = hitCollider.ClosestPoint(controllerPos);
            labelPosition += (Camera.main.transform.position - labelPosition).normalized * 0.05f;
        }

        if (objectInfo && objectInstructionsLabel)
        {
            bool showObjectInfo = hoverObject || grabObject;
            objectInfo.gameObject.SetActive(showObjectInfo);
            Vector3 toObj = labelPosition - Camera.main.transform.position;
            if (hoverObject && !grabObject)
            {
                Vector3 targetPos = labelPosition - toObj.normalized * 0.05f;
                objectInfo.position = Vector3.Lerp(targetPos, objectInfo.position, grabTime);
                objectInfo.rotation = Quaternion.LookRotation(toObj);
                //objectInstructionsLabel.gameObject.SetActive(false);
                objectInfo.localScale = Vector3.one * toObj.magnitude * 2.0f;
                if (hoverObject.GetComponent<GrabObject>())
                {
                    AssignInstructions(hoverObject.GetComponent<GrabObject>());
                }
            }
            else if (grabObject)
            {
                Vector3 targetPos = controllerPos + (Camera.main.transform.position - controllerPos).normalized * 0.1f;
                objectInfo.position = Vector3.Lerp(objectInfo.position, targetPos, grabTime); ;
                objectInfo.rotation = Quaternion.LookRotation(objectInfo.position - Camera.main.transform.position);
                //objectInstructionsLabel.gameObject.SetActive(true);
                objectInfo.localScale = Vector3.one;
                if (grabObject.GetComponent<GrabObject>()) showLaser = grabObject.GetComponent<GrabObject>().showLaserWhileGrabbed;
            }
        }

        // show/hide laser pointer
        if (laser)
        {
            laser.positionCount = 2;
            Vector3 pos1 = controllerPos + controllerRot * (Vector3.forward * 0.05f);
            cursorPosition = controllerPos + controllerRot * (Vector3.forward * rayDistance);
            laser.SetPosition(0, pos1);
            laser.SetPosition(1, cursorPosition);
            laser.enabled = (showLaser);
            if (grabObject && grabObject.GetComponent<GrabObject>())
            {
                grabObject.GetComponent<GrabObject>().CursorPos(cursorPosition);
            }
        }
    }

    // the heavy lifting code for moving objects
    void ManipulateObject(GameObject obj, Vector3 controllerPos, Quaternion controllerRot)
    {
        bool useDefaultManipulation = true;
        Vector2 thumbstick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, controller);

        if (obj.GetComponent<GrabObject>())
        {
            useDefaultManipulation = false;
            switch (obj.GetComponent<GrabObject>().objectManipulationType)
            {
                case GrabObject.ManipulationType.Default:
                    useDefaultManipulation = true;
                    break;
                case GrabObject.ManipulationType.ForcedHand:
                    obj.transform.position = Vector3.Lerp(obj.transform.position, controllerPos, grabTime);
                    obj.transform.rotation = Quaternion.Lerp(obj.transform.rotation, controllerRot, grabTime);
                    break;
                case GrabObject.ManipulationType.DollyHand:
                    float targetDist = localGrabOffset.z + thumbstick.y * 0.01f;
                    targetDist = Mathf.Clamp(targetDist, 0.1f, 2.0f);
                    localGrabOffset = Vector3.forward * targetDist;
                    obj.transform.position = Vector3.Lerp(obj.transform.position, controllerPos + controllerRot * localGrabOffset, grabTime);
                    obj.transform.rotation = Quaternion.Lerp(obj.transform.rotation, controllerRot, grabTime);
                    break;
                case GrabObject.ManipulationType.DollyAttached:
                    obj.transform.position = controllerPos + controllerRot * localGrabOffset;
                    obj.transform.rotation = controllerRot * localGrabRotation;
                    ClampGrabOffset(ref localGrabOffset, thumbstick.y);
                    break;
                case GrabObject.ManipulationType.HorizontalScaled:
                    obj.transform.position = controllerPos + controllerRot * localGrabOffset;
                    if (!OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger, controller))
                    {
                        obj.transform.localScale = ClampScale(obj.transform.localScale, thumbstick);
                    }
                    else
                    {
                        rotationOffset -= thumbstick.x;
                        ClampGrabOffset(ref localGrabOffset, thumbstick.y);
                    }
                    Vector3 newFwd = obj.GetComponent<GrabObject>().grabbedRotation * Vector3.forward;
                    newFwd = new Vector3(newFwd.x, 0, newFwd.z);
                    obj.transform.rotation = Quaternion.Euler(0, rotationOffset, 0) * Quaternion.LookRotation(newFwd);
                    break;
                case GrabObject.ManipulationType.VerticalScaled:
                    obj.transform.position = controllerPos + controllerRot * localGrabOffset;
                    if (!OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger, controller))
                    {
                        obj.transform.localScale = ClampScale(obj.transform.localScale, thumbstick);
                    }
                    else
                    {
                        rotationOffset -= thumbstick.x;
                        ClampGrabOffset(ref localGrabOffset, thumbstick.y);
                    }
                    Vector3 newUp = obj.GetComponent<GrabObject>().grabbedRotation * Vector3.up;
                    newUp = new Vector3(newUp.x, 0, newUp.z);
                    obj.transform.rotation = Quaternion.LookRotation(Vector3.up, Quaternion.Euler(0, rotationOffset, 0) * newUp);
                    break;
                case GrabObject.ManipulationType.Menu:
                    Vector3 targetPos = handGrabPosition + (handGrabRotation * Vector3.forward * 0.4f);
                    Quaternion targetRotation = Quaternion.LookRotation(targetPos - camGrabPosition);
                    obj.transform.position = Vector3.Lerp(obj.transform.position, targetPos, grabTime);
                    obj.transform.rotation = Quaternion.Lerp(obj.transform.rotation, targetRotation, grabTime);
                    break;
                default:
                    useDefaultManipulation = true;
                    break;
            }
        }

        if (useDefaultManipulation)
        {
            obj.transform.position = controllerPos + controllerRot * localGrabOffset;
            obj.transform.Rotate(Vector3.up * -thumbstick.x);
            ClampGrabOffset(ref localGrabOffset, thumbstick.y);
        }
    }

    void ClampGrabOffset(ref Vector3 localOffset, float thumbY)
    {
        Vector3 projectedGrabOffset = localOffset + Vector3.forward * thumbY * 0.01f;
        if (projectedGrabOffset.z > 0.1f)
        {
            localOffset = projectedGrabOffset;
        }
    }

    Vector3 ClampScale(Vector3 localScale, Vector2 thumb)
    {
        float newXscale = localScale.x + thumb.x * 0.01f;
        if (newXscale <= 0.1f) newXscale = 0.1f;
        float newZscale = localScale.z + thumb.y * 0.01f;
        if (newZscale <= 0.1f) newZscale = 0.1f;
        return new Vector3(newXscale, 0.0f, newZscale);
    }

    void CheckForDominantHand()
    {
        // don't switch if hovering or grabbing
        if (hoverObject || grabObject)
        {
            return;
        }
        if (controller == OVRInput.Controller.RTouch)
        {
            if (OVRInput.Get(OVRInput.RawButton.LHandTrigger))
            {
                controller = OVRInput.Controller.LTouch;
            }
        }
        else
        {
            if (OVRInput.Get(OVRInput.RawButton.RHandTrigger))
            {
                controller = OVRInput.Controller.RTouch;
            }
        }
    }

    void AssignInstructions(GrabObject targetObject)
    {
        if (objectNameLabel)
        {
            objectNameLabel.text = targetObject.ObjectName;
        }
        if (objectInstructionsLabel)
        {
            if (grabObject)
            {
                objectInstructionsLabel.text = targetObject.ObjectInstructions;
            }
            else
            {
                objectInstructionsLabel.text = "Grip Trigger to grab";
            }
        }
    }
}
