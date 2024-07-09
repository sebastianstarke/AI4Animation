using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PassthroughSurface : MonoBehaviour
{
    public OVRPassthroughLayer passthroughLayer;
    public MeshFilter projectionObject;

    void Start()
    {
        Destroy(projectionObject.GetComponent<MeshRenderer>());
        passthroughLayer.AddSurfaceGeometry(projectionObject.gameObject, true);
    }
}
