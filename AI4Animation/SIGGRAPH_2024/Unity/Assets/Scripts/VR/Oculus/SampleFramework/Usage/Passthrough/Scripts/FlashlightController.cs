using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class FlashlightController : MonoBehaviour
{
    public Light sceneLight;
    public Transform flashlightRoot;
    Vector3 localPosition = Vector3.zero;
    Quaternion localRotation = Quaternion.identity;
    public TextMesh infoText;
    GrabObject externalController = null;

    OVRSkeleton[] skeletons;
    OVRHand[] hands;
    int handIndex = -1;
    bool pinching = false;

    private void Start()
    {
        localRotation = flashlightRoot.localRotation;
        localPosition = flashlightRoot.localPosition;
        skeletons = new OVRSkeleton[2];
        hands = new OVRHand[2];

        externalController = GetComponent<GrabObject>();
        if (externalController)
        {
            externalController.GrabbedObjectDelegate += Grab;
            externalController.ReleasedObjectDelegate += Release;
        }

        if (GetComponent<Flashlight>())
        {
            GetComponent<Flashlight>().EnableFlashlight(false);
        }
    }

    void LateUpdate()
    {
        if (!externalController)
        {
            FindHands();
            bool usingControllers =
              (OVRInput.GetActiveController() == OVRInput.Controller.RTouch ||
              OVRInput.GetActiveController() == OVRInput.Controller.LTouch ||
              OVRInput.GetActiveController() == OVRInput.Controller.Touch);

            if (!usingControllers)
            {
                if (handIndex >= 0)
                {
                    AlignWithHand(hands[handIndex], skeletons[handIndex]);
                }
                if (infoText) infoText.text = "Pinch to toggle flashlight";
            }
            else
            {
                AlignWithController(OVRInput.Controller.RTouch);
                if (OVRInput.GetUp(OVRInput.RawButton.A))
                {
                    if (GetComponent<Flashlight>()) GetComponent<Flashlight>().ToggleFlashlight();
                }
                if (infoText) infoText.text = "Press A to toggle flashlight";
            }
        }
    }

    void FindHands()
    {
        if (skeletons[0] == null || skeletons[1] == null)
        {
            OVRSkeleton[] foundSkeletons = FindObjectsOfType<OVRSkeleton>();
            if (foundSkeletons[0])
            {
                skeletons[0] = foundSkeletons[0];
                hands[0] = skeletons[0].GetComponent<OVRHand>();
                handIndex = 0;
            }
            if (foundSkeletons[1])
            {
                skeletons[1] = foundSkeletons[1];
                hands[1] = skeletons[1].GetComponent<OVRHand>();
                handIndex = 1;
            }
        }
        else
        {
            if (handIndex == 0)
            {
                if (hands[1].GetFingerIsPinching(OVRHand.HandFinger.Index))
                {
                    handIndex = 1;
                }
            }
            else
            {
                if (hands[0].GetFingerIsPinching(OVRHand.HandFinger.Index))
                {
                    handIndex = 0;
                }
            }
        }
    }

    void AlignWithHand(OVRHand hand, OVRSkeleton skeleton)
    {
        if (pinching)
        {
            if (hand.GetFingerPinchStrength(OVRHand.HandFinger.Index) < 0.8f)
            {
                pinching = false;
            }
        }
        else
        {
            if (hand.GetFingerIsPinching(OVRHand.HandFinger.Index))
            {
                if (GetComponent<Flashlight>()) GetComponent<Flashlight>().ToggleFlashlight();
                pinching = true;
            }
        }
        flashlightRoot.position = skeleton.Bones[6].Transform.position;
        flashlightRoot.rotation = Quaternion.LookRotation(skeleton.Bones[6].Transform.position - skeleton.Bones[0].Transform.position);
    }

    void AlignWithController(OVRInput.Controller controller)
    {
        transform.position = OVRInput.GetLocalControllerPosition(controller);
        transform.rotation = OVRInput.GetLocalControllerRotation(controller);

        flashlightRoot.localRotation = localRotation;
        flashlightRoot.localPosition = localPosition;
    }

    public void Grab(OVRInput.Controller grabHand)
    {
        if (GetComponent<Flashlight>())
        {
            GetComponent<Flashlight>().EnableFlashlight(true);
        }
        StopAllCoroutines();
        StartCoroutine(FadeLighting(new Color(0, 0, 0, 0.95f), 0.0f, 0.25f));
    }

    public void Release()
    {
        if (GetComponent<Flashlight>())
        {
            GetComponent<Flashlight>().EnableFlashlight(false);
        }
        StopAllCoroutines();
        StartCoroutine(FadeLighting(Color.clear, 1.0f, 0.25f));
    }

    IEnumerator FadeLighting(Color newColor, float sceneLightIntensity, float fadeTime)
    {
        float timer = 0.0f;
        Color currentColor = Camera.main.backgroundColor;
        float currentLight = sceneLight ? sceneLight.intensity : 0;
        while (timer <= fadeTime)
        {
            timer += Time.deltaTime;
            float normTimer = Mathf.Clamp01(timer / fadeTime);
            Camera.main.backgroundColor = Color.Lerp(currentColor, newColor, normTimer);
            if (sceneLight) sceneLight.intensity = Mathf.Lerp(currentLight, sceneLightIntensity, normTimer);
            yield return null;
        }
    }
}
