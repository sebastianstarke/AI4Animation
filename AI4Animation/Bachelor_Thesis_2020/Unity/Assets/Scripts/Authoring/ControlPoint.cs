using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class ControlPoint
{
    public GameObject GameObject;
    public SphereCollider SphereCollider;
    public LayerMask Ground = ~1;
    public Transform Transform;
    public Vector3 Velocity;
    public Style[] Styles;
    public int MotionTime;

    public bool Inspect = false;
    public bool Inspector = true;
    public ControlPoint(AnimationAuthoring tool)
    {
        GameObject = new GameObject("ControlPoint" + (tool.transform.childCount).ToString());
        GameObject.transform.SetParent(tool.transform);
        SphereCollider = GameObject.AddComponent<SphereCollider>();
        SphereCollider.center = Vector3.zero;
        SphereCollider.radius = 0.2f;
        GameObject.AddComponent<DragObject>();
        this.Transform = GameObject.transform;
        this.Velocity = Vector3.zero;
        this.Styles = new Style[]
        {
            CreateStyle("Idle", 0f ), CreateStyle("Move", 1f ), CreateStyle("Jump", 0f ), CreateStyle("Sit", 0f ),
            CreateStyle("Stand", 0f), CreateStyle("Lie", 0f ),  CreateStyle("Sneak", 0f), CreateStyle("Eat", 0f),
            CreateStyle("Hydrate", 0f)
        };

        this.MotionTime = 0;
    }
    public ControlPoint(AnimationAuthoring tool, string[] names, float[] values)
    {
        if (tool.transform.childCount > 0 && tool != null)
        {
            GameObject = new GameObject("ControlPoint" + (tool.transform.childCount).ToString());
        }
        else
        {
            GameObject = new GameObject("ControlPoint0");
        }
        
        GameObject.transform.SetParent(tool.transform);
        SphereCollider = GameObject.AddComponent<SphereCollider>();
        SphereCollider.center = Vector3.zero;
        SphereCollider.radius = 0.2f;
        GameObject.AddComponent<DragObject>();
        this.Transform = GameObject.transform;
        this.Velocity = Vector3.zero;
        SetStyles(names, values);

        this.MotionTime = 0;
    }

    //hidden Points
    public ControlPoint(AnimationAuthoring tool, string text)
    {
        GameObject = new GameObject("ControlPoint(hidden)" + (tool.transform.childCount).ToString());
        GameObject.transform.SetParent(tool.transform);
        this.Transform = GameObject.transform;
        this.Velocity = Vector3.zero;

        this.MotionTime = 0;
    }

    public Transform GetTransform()
    {
        return Transform;
    }

    public void SetTransform(Transform transform)
    {
        this.Transform = transform;
    }

    public Vector3 GetPosition()
    {
        return Transform.position;
    }

    public void SetPosition(Vector3 vel)
    {
        Transform.position = vel;
    }
    public int GetMotionTime()
    {
        return MotionTime;
    }

    public void SetMotionTime(int t)
    {
        MotionTime = t;
    }

    public Vector3 GetVelocity()
    {
        return Velocity;
    }

    public void SetVelocity(Vector3 vel)
    {
        Velocity = vel;
    }
    
    public Style[] GetStyles()
    {
        return Styles;
    }

    public Style CreateStyle(string name, float value)
    {
        return new Style(name, value);
    }

    public void SetStyles(string[] names, float[] values)
    {
        Styles = new Style[names.Length];
        for(int i=0; i < names.Length; i++)
        {
            Styles[i] = new Style(names[i], values[i]);
        }
    }

    [System.Serializable]
    public class Style
    {
        public string Name;
        public float Value;

        public Style(string name, float value)
        {
            Name = name;
            Value = value;
        }
    }
}
