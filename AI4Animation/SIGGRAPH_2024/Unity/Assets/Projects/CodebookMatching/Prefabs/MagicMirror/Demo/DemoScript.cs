using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;

public class DemoScript : MonoBehaviour
{
    public Transform RotateMirror;
    public float RotateMirrorSpeed = 15.0f;
    [Range(10.0f, 75.0f)]
    public float RotateMirrorAngle = 30.0f;
    public GameObject LightBulb;
    public GameObject Head;
    public GameObject Body;
    public bool ShowFPS = true;

    private float deltaTime;
    private float rotationModifier = -1.0f;
    private float moveModifier = 1.0f;
    private Material lightBulbMaterial;

    private enum RotationAxes { MouseXAndY = 0, MouseX = 1, MouseY = 2 }
    private RotationAxes axes = RotationAxes.MouseXAndY;
    private float sensitivityX = 15F;
    private float sensitivityY = 15F;
    private float minimumX = -360F;
    private float maximumX = 360F;
    private float minimumY = -60F;
    private float maximumY = 60F;
    private float rotationX = 0F;
    private float rotationY = 0F;
    private Quaternion originalRotation;

    // Use this for initialization
    void Start()
    {
        originalRotation = Head.transform.localRotation;
        Renderer r = LightBulb.GetComponent<Renderer>();
        if (Application.isPlaying)
        {
            r.sharedMaterial = r.material;
        }
        lightBulbMaterial = r.sharedMaterial;
    }

    // Update is called once per frame
    void Update()
    {
        deltaTime += (Time.unscaledDeltaTime - deltaTime) * 0.1f;
        RotateMirrorFunc();
        MoveLightBulb();
        UpdateMouseLook();
        UpdateMovement();
    }

    private void OnGUI()
    {
        if (!ShowFPS)
        {
            return;
        }

        int w = Screen.width, h = Screen.height;

        GUIStyle style = new GUIStyle();

        Rect rect = new Rect(8, 8, w, h * 2 / 100);
        style.alignment = TextAnchor.UpperLeft;
        style.fontSize = h * 2 / 100;
        style.normal.textColor = Color.white;
        float msec = deltaTime * 1000.0f;
        float fps = 1.0f / deltaTime;
        string text = string.Format("{0:0.0} ms ({1:0.} fps)", msec, fps);
        GUI.Label(rect, text, style);
    }

    private void UpdateMovement()
    {
        float speed = 4.0f * Time.deltaTime;

        if (Input.GetKey(KeyCode.W))
        {
            transform.Translate(Head.transform.forward * speed);
        }
        else if (Input.GetKey(KeyCode.S))
        {
            transform.Translate(Head.transform.forward * -speed);
        }

        if (Input.GetKey(KeyCode.A))
        {
            transform.Translate(Head.transform.right * -speed);
        }
        else if (Input.GetKey(KeyCode.D))
        {
            transform.Translate(Head.transform.right * speed);
        }
    }

    private void RotateMirrorFunc()
    {
        if (RotateMirror == null)
        {
            return;
        }

        Vector3 angles = RotateMirror.localRotation.eulerAngles;
        float y = angles.y;
        RotateMirror.transform.Rotate(Vector3.up, rotationModifier * Time.deltaTime * RotateMirrorSpeed, Space.Self);
        angles = RotateMirror.localRotation.eulerAngles;
        if (y >= 360.0f - RotateMirrorAngle && angles.y > 180.0f && angles.y < 360.0f - RotateMirrorAngle)
        {
            rotationModifier = -rotationModifier;
            angles.y = 360.0f - RotateMirrorAngle;
            Quaternion rot = RotateMirror.localRotation;
            rot.eulerAngles = angles;
            RotateMirror.localRotation = rot;
        }
        else if (angles.y >= RotateMirrorAngle && angles.y < 180.0f && y < RotateMirrorAngle)
        {
            rotationModifier = -rotationModifier;
            angles.y = RotateMirrorAngle;
            Quaternion rot = RotateMirror.localRotation;
            rot.eulerAngles = angles;
            RotateMirror.localRotation = rot;
        }
    }

    private void MoveLightBulb()
    {
        float x = LightBulb.transform.position.x;
        if (x > 5)
        {
            moveModifier = -moveModifier;
            x = 5;
        }
        else if (x < -5)
        {
            moveModifier = -moveModifier;
            x = -5;
        }
        else
        {
            x += (Time.deltaTime * moveModifier);
        }

        Light l = LightBulb.GetComponent<Light>();
        LightBulb.transform.position = new Vector3(x, LightBulb.transform.position.y, LightBulb.transform.position.z);
        float i = Mathf.Min(1.0f, l.intensity);
        lightBulbMaterial.SetColor("_EmissionColor", new Color(i, i, i));
    }

    private void UpdateMouseLook()
    {
        if (axes == RotationAxes.MouseXAndY)
        {
            // Read the mouse input axis
            rotationX += Input.GetAxis("Mouse X") * sensitivityX;
            rotationY += Input.GetAxis("Mouse Y") * sensitivityY;

            rotationX = ClampAngle(rotationX, minimumX, maximumX);
            rotationY = ClampAngle(rotationY, minimumY, maximumY);

            Quaternion xQuaternion = Quaternion.AngleAxis(rotationX, Vector3.up);
            Quaternion yQuaternion = Quaternion.AngleAxis(rotationY, -Vector3.right);

            Head.transform.localRotation = originalRotation * xQuaternion * yQuaternion;
            Body.transform.localRotation = originalRotation * xQuaternion;
        }
        else if (axes == RotationAxes.MouseX)
        {
            rotationX += Input.GetAxis("Mouse X") * sensitivityX;
            rotationX = ClampAngle(rotationX, minimumX, maximumX);

            Quaternion xQuaternion = Quaternion.AngleAxis(rotationX, Vector3.up);
            Body.transform.localRotation = originalRotation * xQuaternion;
            Head.transform.localRotation = originalRotation * xQuaternion;
        }
        else
        {
            rotationY += Input.GetAxis("Mouse Y") * sensitivityY;
            rotationY = ClampAngle(rotationY, minimumY, maximumY);

            Quaternion yQuaternion = Quaternion.AngleAxis(-rotationY, Vector3.right);
            Head.transform.localRotation = originalRotation * yQuaternion;
        }
    }

    public static float ClampAngle(float angle, float min, float max)
    {
        if (angle < -360F)
        {
            angle += 360F;
        }
        if (angle > 360F)
        {
            angle -= 360F;
        }

        return Mathf.Clamp(angle, min, max);
    }
}
