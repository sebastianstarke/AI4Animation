#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

public class SceneCamera : MonoBehaviour {

    public Transform Target = null;
	public float FocusHeight = 1f;
	public float FocusDistance = 2f;
	public float FocusAngle = 90f;
	public float FocusSmoothing = 0.05f;

    void OnEnable() {
        Vector3 position =  SceneView.lastActiveSceneView.camera.transform.position;
        Quaternion rotation = Quaternion.Euler(0f, SceneView.lastActiveSceneView.camera.transform.rotation.eulerAngles.y, 0f);
        SceneView.lastActiveSceneView.LookAtDirect(position, rotation, 0f);
    }

    void Update() {
        if(Target != null && SceneView.lastActiveSceneView != null) {
            Vector3 lastPosition = SceneView.lastActiveSceneView.camera.transform.position;
            Quaternion lastRotation = SceneView.lastActiveSceneView.camera.transform.rotation;
            Vector3 reference = Target.position;
            Vector3 position = reference + Quaternion.Euler(0f, FocusAngle, 0f) * (FocusDistance * Vector3.forward) + new Vector3(0f, FocusHeight, 0f);
            Quaternion rotation = Quaternion.LookRotation(Vector3.ProjectOnPlane(reference - position, Vector3.up).normalized, Vector3.up);
            SceneView.lastActiveSceneView.LookAtDirect(Vector3.Lerp(lastPosition, position, 1f-FocusSmoothing), Quaternion.Slerp(lastRotation, rotation, (1f-FocusSmoothing)), FocusDistance*(1f-FocusSmoothing));
        }
    }

}
#endif