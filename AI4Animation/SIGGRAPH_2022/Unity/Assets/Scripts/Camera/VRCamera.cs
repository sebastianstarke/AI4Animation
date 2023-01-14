using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VRCamera : MonoBehaviour {

    public Transform Character;
    public Transform Headset;
    public Transform LeftController;
    public Transform RightController;
    
	public Vector3 SelfOffset = Vector3.zero;
	public Vector3 TargetOffset = Vector3.zero;
    [Range(0f, 1f)] public float Damping = 0.975f;


    void Update() {
        Vector3 currentPosition = transform.position;
        Quaternion currentRotation = transform.rotation;

        Vector3 newPosition = Character.position + Character.rotation * SelfOffset;
        Quaternion newRotation = Headset.rotation;
        transform.position = newPosition;
        transform.rotation = newRotation;
        transform.LookAt(Character.position + Character.rotation * TargetOffset);

        transform.position = Vector3.Lerp(currentPosition, transform.position, 1f-Damping);
        transform.rotation = Quaternion.Slerp(currentRotation, transform.rotation, 1f-Damping);

    }

}
