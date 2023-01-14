#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

public class GameCamera : MonoBehaviour {

    public Transform Target = null;
	public Vector3 SelfOffset = Vector3.zero;
	public Vector3 TargetOffset = Vector3.zero;
    
    [Range(0f, 1f)] public float Smoothing = 0f;
    [Range(0f, 10f)] public float FOV = 1.5f;

    private Camera Camera;

    private Vector3 PreviousTarget;

    void Awake() {
        Camera = GetComponent<Camera>();
        PreviousTarget = Target.position;
    }

    void Update() {
        Vector3 target = Vector3.Lerp(PreviousTarget, Target.position, 1f-Smoothing);
        PreviousTarget = target;
        transform.position = target + FOV*SelfOffset;
        transform.LookAt(target + TargetOffset);
    }

}
#endif