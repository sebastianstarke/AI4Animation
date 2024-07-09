using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnableUnpremultipliedAlpha : MonoBehaviour
{
    void Start()
    {
        // Since the alpha values for Selective Passthrough are written to the framebuffers after the color pass, we
        // need to ensure that the color values get multiplied by the alpha value during compositing. By default, this is
        // not the case, as framebuffers typically contain premultiplied color values. This step is only needed when
        // Selective Passthrough is non-binary (i.e. alpha values are neither 0 nor 1), and it doesn't work if the
        // framebuffer contains semi-transparent pixels even without Selective Passthrough, as those will have
        // premultiplied colors.
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || UNITY_ANDROID
        OVRManager.eyeFovPremultipliedAlphaModeEnabled = false;
#endif
    }
}
