using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BrushController : MonoBehaviour
{
    public PassthroughBrush brush;
    public MeshRenderer backgroundSphere;

    IEnumerator grabRoutine;
    IEnumerator releaseRoutine;

    void Start()
    {
        brush.controllerHand = OVRInput.Controller.None;

        if (!brush.lineContainer)
        {
            brush.lineContainer = new GameObject("LineContainer");
        }

        // the material on the background sphere ignores z-write, so it can overwrite other opaque objects in the scene
        // also renders after transparent objects
        backgroundSphere.material.renderQueue = 3998;
        // the selective Passthrough shader renders at 4000 and higher, to render after other transparent objects
        // (white ring and info text render after)
        backgroundSphere.transform.parent = null;
        backgroundSphere.enabled = false;

        if (GetComponent<GrabObject>())
        {
            GetComponent<GrabObject>().GrabbedObjectDelegate += Grab;
            GetComponent<GrabObject>().ReleasedObjectDelegate += Release;
        }
    }

    void Update()
    {
        backgroundSphere.transform.position = Camera.main.transform.position;
    }

    public void Grab(OVRInput.Controller grabHand)
    {
        brush.controllerHand = grabHand;
        brush.lineContainer.SetActive(true);
        backgroundSphere.enabled = true;

        if (grabRoutine != null) StopCoroutine(grabRoutine);
        if (releaseRoutine != null) StopCoroutine(releaseRoutine);
        grabRoutine = FadeSphere(Color.grey, 0.25f);
        StartCoroutine(grabRoutine);
    }

    public void Release()
    {
        brush.controllerHand = OVRInput.Controller.None;
        brush.lineContainer.SetActive(false);

        if (grabRoutine != null) StopCoroutine(grabRoutine);
        if (releaseRoutine != null) StopCoroutine(releaseRoutine);
        releaseRoutine = FadeSphere(new Color(0.5f, 0.5f, 0.5f, 0.0f), 0.25f, true);
        StartCoroutine(releaseRoutine);
    }

    IEnumerator FadeCameraClearColor(Color newColor, float fadeTime)
    {
        float timer = 0.0f;
        Color currentColor = Camera.main.backgroundColor;
        while (timer <= fadeTime)
        {
            timer += Time.deltaTime;
            float normTimer = Mathf.Clamp01(timer / fadeTime);
            Camera.main.backgroundColor = Color.Lerp(currentColor, newColor, normTimer);
            yield return null;
        }
    }

    IEnumerator FadeSphere(Color newColor, float fadeTime, bool disableOnFinish = false)
    {
        float timer = 0.0f;
        Color currentColor = backgroundSphere.material.GetColor("_Color");
        while (timer <= fadeTime)
        {
            timer += Time.deltaTime;
            float normTimer = Mathf.Clamp01(timer / fadeTime);
            backgroundSphere.material.SetColor("_Color", Color.Lerp(currentColor, newColor, normTimer));
            if (disableOnFinish && timer >= fadeTime)
            {
                backgroundSphere.enabled = false;
            }
            yield return null;
        }
    }
}
