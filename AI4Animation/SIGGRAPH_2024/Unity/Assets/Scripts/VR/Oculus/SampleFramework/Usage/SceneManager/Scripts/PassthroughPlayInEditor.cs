using UnityEngine;

[RequireComponent(typeof(OVRManager))]
public class PassthroughPlayInEditor : MonoBehaviour
{
  void Awake()
  {
#if UNITY_EDITOR
    // Disable passthrough in editor to avoid errors being printed
    GetComponent<OVRManager>().isInsightPassthroughEnabled = false;
    OVRPassthroughLayer passthroughLayer = GetComponent<OVRPassthroughLayer>();
    if(passthroughLayer)
    {
      passthroughLayer.enabled = false;
    }
#endif
  }
}
