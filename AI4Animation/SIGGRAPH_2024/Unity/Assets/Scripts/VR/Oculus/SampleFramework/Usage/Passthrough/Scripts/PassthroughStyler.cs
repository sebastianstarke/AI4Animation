using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class PassthroughStyler : MonoBehaviour
{
    public OVRInput.Controller controllerHand = OVRInput.Controller.None;
    public OVRPassthroughLayer passthroughLayer;
    IEnumerator fadeIn;
    IEnumerator fadeOut;

    public RectTransform[] menuOptions;
    public RectTransform colorWheel;
    public Texture2D colorTexture;
    Vector3 cursorPosition = Vector3.zero;

    bool settingColor = false;
    Color savedColor = Color.white;
    float savedBrightness = 0.0f;
    float savedContrast = 0.0f;

    public CanvasGroup mainCanvas;

    public GameObject[] compactObjects;

    void Start()
    {
        if (GetComponent<GrabObject>())
        {
            GetComponent<GrabObject>().GrabbedObjectDelegate += Grab;
            GetComponent<GrabObject>().ReleasedObjectDelegate += Release;
            GetComponent<GrabObject>().CursorPositionDelegate += Cursor;
        }
        savedColor = new Color(1, 1, 1, 0);
        ShowFullMenu(false);
        mainCanvas.interactable = false;
        passthroughLayer.colorMapEditorType = OVRPassthroughLayer.ColorMapEditorType.ColorAdjustment;
    }

    void Update()
    {
        if (controllerHand == OVRInput.Controller.None)
        {
            return;
        }
        if (settingColor)
        {
            GetColorFromWheel();
        }
    }

    public void Grab(OVRInput.Controller grabHand)
    {
        controllerHand = grabHand;
        ShowFullMenu(true);
        if (mainCanvas) mainCanvas.interactable = true;

        if (fadeIn != null) StopCoroutine(fadeIn);
        if (fadeOut != null) StopCoroutine(fadeOut);
        fadeIn = FadeToCurrentStyle(0.2f);
        StartCoroutine(fadeIn);
    }

    public void Release()
    {
        controllerHand = OVRInput.Controller.None;
        ShowFullMenu(false);
        if (mainCanvas) mainCanvas.interactable = false;

        if (fadeIn != null) StopCoroutine(fadeIn);
        if (fadeOut != null) StopCoroutine(fadeOut);
        fadeOut = FadeToDefaultPassthrough(0.2f);
        StartCoroutine(fadeOut);
    }

    IEnumerator FadeToCurrentStyle(float fadeTime)
    {
        float timer = 0.0f;
        float brightness = passthroughLayer.colorMapEditorBrightness;
        float contrast = passthroughLayer.colorMapEditorContrast;
        Color edgeCol = new Color(savedColor.r, savedColor.g, savedColor.b, 0.0f);
        passthroughLayer.edgeRenderingEnabled = true;
        while (timer <= fadeTime)
        {
            timer += Time.deltaTime;
            float normTimer = Mathf.Clamp01(timer / fadeTime);
            passthroughLayer.colorMapEditorBrightness = Mathf.Lerp(brightness, savedBrightness, normTimer);
            passthroughLayer.colorMapEditorContrast = Mathf.Lerp(contrast, savedContrast, normTimer);
            passthroughLayer.edgeColor = Color.Lerp(edgeCol, savedColor, normTimer);
            yield return null;
        }
    }

    IEnumerator FadeToDefaultPassthrough(float fadeTime)
    {
        float timer = 0.0f;
        float brightness = passthroughLayer.colorMapEditorBrightness;
        float contrast = passthroughLayer.colorMapEditorContrast;
        Color edgeCol = passthroughLayer.edgeColor;
        while (timer <= fadeTime)
        {
            timer += Time.deltaTime;
            float normTimer = Mathf.Clamp01(timer / fadeTime);
            passthroughLayer.colorMapEditorBrightness = Mathf.Lerp(brightness, 0.0f, normTimer);
            passthroughLayer.colorMapEditorContrast = Mathf.Lerp(contrast, 0.0f, normTimer);
            passthroughLayer.edgeColor = Color.Lerp(edgeCol, new Color(edgeCol.r, edgeCol.g, edgeCol.b, 0.0f), normTimer);
            if (timer > fadeTime)
            {
                passthroughLayer.edgeRenderingEnabled = false;
            }
            yield return null;
        }
    }

    public void OnBrightnessChanged(float newValue)
    {
        savedBrightness = newValue;
        passthroughLayer.colorMapEditorBrightness = savedBrightness;
    }

    public void OnContrastChanged(float newValue)
    {
        savedContrast = newValue;
        passthroughLayer.colorMapEditorContrast = savedContrast;
    }

    public void OnAlphaChanged(float newValue)
    {
        savedColor = new Color(savedColor.r, savedColor.g, savedColor.b, newValue);
        passthroughLayer.edgeColor = savedColor;
    }

    void ShowFullMenu(bool doShow)
    {
        foreach (GameObject go in compactObjects)
        {
            go.SetActive(doShow);
        }
    }

    public void Cursor(Vector3 cP)
    {
        cursorPosition = cP;
    }

    public void DoColorDrag(bool doDrag)
    {
        settingColor = doDrag;
    }

    public void GetColorFromWheel()
    {
        // convert cursor world position to UV
        var localPos = colorWheel.transform.InverseTransformPoint(cursorPosition);
        var toImg = new Vector2(localPos.x / colorWheel.sizeDelta.x + 0.5f, localPos.y / colorWheel.sizeDelta.y + 0.5f);
        Debug.Log("Sanctuary: " + toImg.x.ToString() + ", " + toImg.y.ToString());
        Color sampledColor = Color.black;
        if (toImg.x < 1.0 && toImg.x > 0.0f && toImg.y < 1.0 && toImg.y > 0.0f)
        {
            int Upos = Mathf.RoundToInt(toImg.x * colorTexture.width);
            int Vpos = Mathf.RoundToInt(toImg.y * colorTexture.height);
            sampledColor = colorTexture.GetPixel(Upos, Vpos);
        }
        savedColor = new Color(sampledColor.r, sampledColor.g, sampledColor.b, savedColor.a);
        passthroughLayer.edgeColor = savedColor;
    }
}
