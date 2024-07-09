using UnityEngine;

public class Example_PID : MonoBehaviour {

    public PID_Vector3Controller Vector3Controller;
    public PID_QuaternionController QuaternionController;

    public Transform Target;

    void Update() {
        transform.position = Vector3Controller.Update(transform.position, Target.position, Time.deltaTime);
        transform.rotation = QuaternionController.Update(transform.rotation, Target.rotation, Time.deltaTime);
    }

}
