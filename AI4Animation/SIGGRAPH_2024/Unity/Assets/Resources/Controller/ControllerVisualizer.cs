using UnityEngine;
using UnityEngine.UI;

public class ControllerVisualizer : MonoBehaviour {

    public int ID = 1;

    public Color ActiveColor = Color.magenta;
    public Color InactiveColor = Color.white;

    private Image LB, RB, X, Y, A, B, LArrow, RArrow;

    void Awake() {
        LB = transform.FindRecursive("LB").GetComponent<Image>();
        RB = transform.FindRecursive("RB").GetComponent<Image>();
        X = transform.FindRecursive("X").GetComponent<Image>();
        Y = transform.FindRecursive("Y").GetComponent<Image>();
        A = transform.FindRecursive("A").GetComponent<Image>();
        B = transform.FindRecursive("B").GetComponent<Image>();
        LArrow = transform.FindRecursive("LArrow").GetComponent<Image>();
        RArrow = transform.FindRecursive("RArrow").GetComponent<Image>();
    }

    void Update() {
        void UpdateButton(Image image, string tag) {
            if(Input.GetButton(ID + tag)) {
                image.color = ActiveColor;
            } else {
                image.color = InactiveColor;
            }
        }
        UpdateButton(X, "ButtonX");
        UpdateButton(Y, "ButtonY");
        UpdateButton(A, "ButtonA");
        UpdateButton(B, "ButtonB");
        UpdateButton(LB, "LB");
        UpdateButton(RB, "RB");

        Vector3 leftArrow = new Vector3(Input.GetAxis(ID+"X"), 0f, Input.GetAxis(ID+"Y")).ClampMagnitudeXZ(1f);
        if(leftArrow.magnitude > 0f) {
            LArrow.color = ActiveColor;
            LArrow.rectTransform.localScale = leftArrow.magnitude * new Vector3(1f, -1f, 1f);
            LArrow.rectTransform.rotation = Quaternion.Euler(0f, 0f, -Vector3.SignedAngle(Vector3.forward, leftArrow, Vector3.up));
        } else {
            LArrow.color = InactiveColor;
        }

        Vector3 rightArrow = new Vector3(Input.GetAxis(ID+"H"), 0f, -Input.GetAxis(ID+"V")).ClampMagnitudeXZ(1f);
        if(rightArrow.magnitude > 0f) {
            RArrow.color = ActiveColor;
            RArrow.rectTransform.localScale = rightArrow.magnitude * new Vector3(1f, -1f, 1f);
            RArrow.rectTransform.rotation = Quaternion.Euler(0f, 0f, -Vector3.SignedAngle(Vector3.forward, rightArrow, Vector3.up));
        } else {
            RArrow.color = InactiveColor;
        }

    }

}
