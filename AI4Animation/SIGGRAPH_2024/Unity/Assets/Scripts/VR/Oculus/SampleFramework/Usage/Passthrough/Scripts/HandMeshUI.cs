using UnityEngine;
using UnityEngine.UI;

// low-effort way to get a UI
public class HandMeshUI : MonoBehaviour
{
    public SphereCollider[] knobs;
    public TextMesh[] readouts;

    int rightHeldKnob = -1;
    int leftHeldKnob = -1;

    public OVRSkeleton leftHand;
    public OVRSkeleton rightHand;

    public HandMeshMask leftMask;
    public HandMeshMask rightMask;

    void Start()
    {
        SetSliderValue(0, rightMask.radialDivisions, false);
        SetSliderValue(1, rightMask.borderSize, false);
        SetSliderValue(2, rightMask.fingerTaper, false);
        SetSliderValue(3, rightMask.fingerTipLength, false);
        SetSliderValue(4, rightMask.webOffset, false);
    }

    void Update()
    {
        CheckForHands();

        Vector3 RfingerPos = rightHand.Bones[20].Transform.position;
        Vector3 LfingerPos = leftHand.Bones[20].Transform.position;
        if (rightHeldKnob >= 0)
        {
            Vector3 localCursorPos = knobs[rightHeldKnob].transform.parent.InverseTransformPoint(RfingerPos);
            SetSliderValue(rightHeldKnob, Mathf.Clamp01(localCursorPos.x * 10), true);
            if (localCursorPos.z < -0.02f)
            {
                rightHeldKnob = -1;
            }
        }
        else
        {
            for (int i = 0; i < knobs.Length; i++)
            {
                if (Vector3.Distance(RfingerPos, knobs[i].transform.position) <= 0.02f && leftHeldKnob != i)
                {
                    rightHeldKnob = i;
                    break;
                }
            }
        }

        if (leftHeldKnob >= 0)
        {
            Vector3 localCursorPos = knobs[leftHeldKnob].transform.parent.InverseTransformPoint(LfingerPos);
            SetSliderValue(leftHeldKnob, Mathf.Clamp01(localCursorPos.x * 10), true);
            if (localCursorPos.z < -0.02f)
            {
                leftHeldKnob = -1;
            }
        }
        else
        {
            for (int i = 0; i < knobs.Length; i++)
            {
                if (Vector3.Distance(LfingerPos, knobs[i].transform.position) <= 0.02f && rightHeldKnob != i)
                {
                    leftHeldKnob = i;
                    break;
                }
            }
        }
    }

    void SetSliderValue(int sliderID, float value, bool isNormalized)
    {
        float sliderStart = 0.0f;
        float sliderEnd = 1.0f;
        float sliderScale = 0.1f;
        string displayString = "";
        switch (sliderID)
        {
            case 0:
                sliderStart = 2;
                sliderEnd = 16;
                displayString = "{0, 0:0}";
                break;
            case 1:
                sliderStart = 0;
                sliderEnd = 0.05f;
                displayString = "{0, 0:0.000}";
                break;
            case 2:
                sliderStart = 0;
                sliderEnd = 0.3333f;
                displayString = "{0, 0:0.00}";
                break;
            case 3:
                sliderStart = 0.5f;
                sliderEnd = 1.5f;
                displayString = "{0, 0:0.00}";
                break;
            case 4:
                sliderStart = 0.0f;
                sliderEnd = 1.0f;
                displayString = "{0, 0:0.00}";
                break;
        }
        float absoluteValue = isNormalized ? value * (sliderEnd - sliderStart) + sliderStart : value;
        float normalizedValue = isNormalized ? value : (value - sliderStart) / (sliderEnd - sliderStart);
        knobs[sliderID].transform.localPosition = Vector3.right * normalizedValue * sliderScale;
        readouts[sliderID].text = string.Format(displayString, absoluteValue);

        // for both hands, set the properties
        switch (sliderID)
        {
            case 0:
                rightMask.radialDivisions = (int)absoluteValue;
                leftMask.radialDivisions = (int)absoluteValue;
                break;
            case 1:
                rightMask.borderSize = absoluteValue;
                leftMask.borderSize = absoluteValue;
                break;
            case 2:
                rightMask.fingerTaper = absoluteValue;
                leftMask.fingerTaper = absoluteValue;
                break;
            case 3:
                rightMask.fingerTipLength = absoluteValue;
                leftMask.fingerTipLength = absoluteValue;
                break;
            case 4:
                rightMask.webOffset = absoluteValue;
                leftMask.webOffset = absoluteValue;
                break;
        }
    }

    void CheckForHands()
    {
        bool handsActive = (
          OVRInput.GetActiveController() == OVRInput.Controller.Hands ||
          OVRInput.GetActiveController() == OVRInput.Controller.LHand ||
          OVRInput.GetActiveController() == OVRInput.Controller.RHand);

        if (transform.GetChild(0).gameObject.activeSelf)
        {
            if (!handsActive)
            {
                transform.GetChild(0).gameObject.SetActive(false);
                leftHeldKnob = -1;
                rightHeldKnob = -1;
            }
        }
        else
        {
            if (handsActive)
            {
                transform.GetChild(0).gameObject.SetActive(true);
                transform.position = (rightHand.Bones[20].Transform.position + rightHand.Bones[20].Transform.position) * 0.5f;
                transform.position += (transform.position - Camera.main.transform.position).normalized * 0.1f;
                transform.rotation = Quaternion.LookRotation(new Vector3(Camera.main.transform.forward.x, 0, Camera.main.transform.forward.z));
            }
        }
    }
}
