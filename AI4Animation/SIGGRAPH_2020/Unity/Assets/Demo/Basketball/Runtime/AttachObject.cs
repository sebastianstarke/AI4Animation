using UnityEngine;

public class AttachObject : MonoBehaviour {

    public Transform Reference;
    
    public bool TranslationX;
    public bool TranslationY;
    public bool TranslationZ;
    public bool RotationX;
    public bool RotationY;
    public bool RotationZ;

    void Update() {
        transform.position = new Vector3(
            TranslationX ? Reference.position.x : transform.position.x,
            TranslationY ? Reference.position.y : transform.position.y,
            TranslationZ ? Reference.position.z : transform.position.z
        );
        transform.rotation = Quaternion.Euler(
            RotationX ? Reference.eulerAngles.x : transform.eulerAngles.x,
            RotationY ? Reference.eulerAngles.y : transform.eulerAngles.y,
            RotationZ ? Reference.eulerAngles.z : transform.eulerAngles.z
        );
    }

}
