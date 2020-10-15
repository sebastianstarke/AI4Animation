using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Point
{
    private Vector3 Position;
    private Vector3 Velocity;
    private Vector3 Direction;
    private float Speed;
    private float[] Styles;

    public Point()
    {
        this.Position = Vector3.zero;
        this.Velocity = Vector3.zero;
        this.Direction = Vector3.zero;
        this.Speed = 0f;
        this.Styles = new float[0];
    }


    public Vector3 GetPosition()
    {
        return Position;
    }

    public void SetPosition(Vector3 vel)
    {
        Position = vel;
    }

    public Vector3 GetVelocity()
    {
        return Velocity;
    }

    public void SetDirection(Vector3 dir)
    {
        Direction = dir;
    }

    public Vector3 GetDirection()
    {
        return Direction;
    }

    public void SetVelocity(Vector3 vel)
    {
        Velocity = vel;
    }

    public void SetSpeed(float speed)
    {
        Speed = speed;
    }

    public float GetSpeed()
    {
        return Speed;
    }

    public float[] GetStyle()
    {
        return Styles;
    }

    public void SetStyle(float[] styles)
    {
        Styles = styles;
    }

}
