using System;
using UnityEngine;

[ExecuteInEditMode]
public class SimpleResizable : MonoBehaviour
{
    public Vector3 PivotPosition => _pivotTransform.position; //Vector3.zero;

    [Space(15)] public Method ScalingX;
    [Range(0, 0.5f)] public float PaddingX;

    [Range(-0.5f, 0)] public float PaddingXMax;

    [Space(15)] public Method ScalingY;
    [Range(0, 0.5f)] public float PaddingY;

    [Range(-0.5f, 0)] public float PaddingYMax;

    [Space(15)] public Method ScalingZ;
    [Range(0, 0.5f)] public float PaddingZ;

    [Range(-0.5f, 0)] public float PaddingZMax;

    public enum Method
    {
        Adapt,
        AdaptWithAsymmetricalPadding,
        Scale,
        None
    }

    public Vector3 NewSize {get; set; }
    public Vector3 DefaultSize {get; private set; }
    public Mesh Mesh { get; private set; }

    private Bounds _bounds;

    [SerializeField] private Transform _pivotTransform;

    private void Awake()
    {
        Mesh = GetComponent<MeshFilter>().sharedMesh;
        DefaultSize = Mesh.bounds.size;
        if(!_pivotTransform)
            _pivotTransform = transform.Find("Pivot");
    }

#if UNITY_EDITOR
    private void OnEnable()
    {
        DefaultSize = Mesh.bounds.size;
        NewSize = DefaultSize;
    }

    private void OnDrawGizmos()
    {
        if (!_pivotTransform) return;

        Gizmos.color = Color.red;
        float lineSize = 0.1f;

        Vector3 startX = _pivotTransform.position + Vector3.left * lineSize * 0.5f;
        Vector3 startY = _pivotTransform.position + Vector3.down * lineSize * 0.5f;
        Vector3 startZ = _pivotTransform.position + Vector3.back * lineSize * 0.5f;

        Gizmos.DrawRay(startX, Vector3.right * lineSize);
        Gizmos.DrawRay(startY, Vector3.up * lineSize);
        Gizmos.DrawRay(startZ, Vector3.forward * lineSize);
    }

    void OnDrawGizmosSelected()
    {
        DefaultSize = Mesh.bounds.size;

        if (GetComponent<MeshFilter>().sharedMesh == null)
        {
            // The furniture piece was not customized yet, nothing to do here
            return;
        }

        _bounds = GetComponent<MeshFilter>().sharedMesh.bounds;
        Gizmos.matrix = transform.localToWorldMatrix;
        Vector3 newCenter = _bounds.center;

        Gizmos.color = new Color(1, 0, 0, 0.5f);
        switch (ScalingX)
        {
            case Method.Adapt:
                Gizmos.DrawWireCube(newCenter, new Vector3(NewSize.x * PaddingX * 2, NewSize.y, NewSize.z));
                break;
            case Method.AdaptWithAsymmetricalPadding:
                Gizmos.DrawWireCube(newCenter + new Vector3(
                    NewSize.x * PaddingX, 0, 0), new Vector3(0, NewSize.y, NewSize.z));
                Gizmos.DrawWireCube(newCenter + new Vector3(
                    NewSize.x * PaddingXMax, 0, 0), new Vector3(0, NewSize.y, NewSize.z));
                break;
            case Method.None:
                Gizmos.DrawWireCube(newCenter, NewSize);
                break;
        }

        Gizmos.color = new Color(0, 1, 0, 0.5f);
        switch (ScalingY)
        {
            case Method.Adapt:
                Gizmos.DrawWireCube(newCenter, new Vector3(NewSize.x, NewSize.y * PaddingY * 2, NewSize.z));
                break;
            case Method.AdaptWithAsymmetricalPadding:
                Gizmos.DrawWireCube(newCenter + new Vector3(0, NewSize.y * PaddingY, 0),
                    new Vector3(NewSize.x, 0, NewSize.z));
                Gizmos.DrawWireCube(newCenter + new Vector3(0, NewSize.y * PaddingYMax, 0),
                    new Vector3(NewSize.x, 0, NewSize.z));
                break;
            case Method.None:
                Gizmos.DrawWireCube(newCenter, NewSize);
                break;
        }

        Gizmos.color = new Color(0, 0, 1, 0.5f);
        switch (ScalingZ)
        {
            case Method.Adapt:
                Gizmos.DrawWireCube(newCenter, new Vector3(NewSize.x, NewSize.y, NewSize.z * PaddingZ * 2));
                break;
            case Method.AdaptWithAsymmetricalPadding:
                Gizmos.DrawWireCube(newCenter + new Vector3(0, 0, NewSize.z * PaddingZ),
                    new Vector3(NewSize.x, NewSize.y, 0));
                Gizmos.DrawWireCube(newCenter + new Vector3(0, 0, NewSize.z * PaddingZMax),
                    new Vector3(NewSize.x, NewSize.y, 0));
                break;
            case Method.None:
                Gizmos.DrawWireCube(newCenter, NewSize);
                break;
        }

        Gizmos.color = new Color(0, 1, 1, 1);
        Gizmos.DrawWireCube(newCenter, NewSize);
    }
#endif
}
