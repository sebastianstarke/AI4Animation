using UnityEngine;

public class Flashlight : MonoBehaviour
{
    public GameObject lightVolume;
    public Light spotlight;
    public GameObject bulbGlow;

    void LateUpdate()
    {
        // ensure all the light volume quads are camera-facing
        for (int i = 0; i < lightVolume.transform.childCount; i++)
        {
            lightVolume.transform.GetChild(i).rotation = Quaternion.LookRotation((lightVolume.transform.GetChild(i).position - Camera.main.transform.position).normalized);
        }
    }

    public void ToggleFlashlight()
    {
        lightVolume.SetActive(!lightVolume.activeSelf);
        spotlight.enabled = !spotlight.enabled;
        bulbGlow.SetActive(lightVolume.activeSelf);
    }

    public void EnableFlashlight(bool doEnable)
    {
        lightVolume.SetActive(doEnable);
        spotlight.enabled = doEnable;
        bulbGlow.SetActive(doEnable);
    }
}
