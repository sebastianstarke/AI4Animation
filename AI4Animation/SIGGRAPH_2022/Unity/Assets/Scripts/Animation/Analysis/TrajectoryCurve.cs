using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class TrajectoryCurve : MonoBehaviour {
    
    public int Resolution = 10;
    public float LineWidth = 0.01f;
    public Color LineColor = Color.cyan;
    public Color PointColor = Color.black;
    public int Step = 10;
    
    private KDTree.KDTree<Point> Data;

    private class Point {
        public int Index;
        public Vector3 Position;
        public Point(int index, Vector3 position) {
            Index = index;
            Position = position;
        }
    }

    private Point[] Points;

    void Start() {
        Generate();
    }

    void Update() {
        if(Application.isPlaying) {
            return;
        }
        Generate();
    }

    private void Generate() {
        Transform[] transforms = GetComponentsInChildren<Transform>();
        if(transforms.Length < 4) {
            return;
        }
        List<Point> points = new List<Point>();
        for(int i=1; i<transforms.Length-1; i++) {
            for(int j=0; j<Resolution; j++) {
                float t = j.Ratio(0, Resolution-1);
                Vector3 v0 = transforms[Mathf.Clamp(i-1, 0, transforms.Length-1)].position;
                Vector3 v1 = transforms[Mathf.Clamp(i, 0, transforms.Length-1)].position;
                Vector3 v2 = transforms[Mathf.Clamp(i+1, 0, transforms.Length-1)].position;
                Vector3 v3 = transforms[Mathf.Clamp(i+2, 0, transforms.Length-1)].position;
                points.Add(new Point(points.Count, Vector3Extensions.CatmullRom(t, v0, v1, v2, v3)));
            }
        }
        Points = points.ToArray();
        Data = new KDTree.KDTree<Point>(3);
        foreach(Point point in Points) {
            Data.AddPoint(point.Position.ToArray().ToDouble(), point);
        }
    }

    public Vector3 GetPosition(int index) {
        return Points[index].Position;
    }

    public Vector3[] GetPath(Vector3 pivot, int samples) {
        KDTree.NearestNeighbour<Point> result = Data.NearestNeighbors(pivot.ToArray().ToDouble(), 1);
        foreach(Point point in result) {
            Vector3[] path = new Vector3[samples];
            for(int i=0; i<samples; i++) {
                path[i] = Points[point.Index + i].Position;
            }
            return path;
        }
        return null;
    }

    void OnRenderObject() {
        if(Points == null) {
            return;
        }
        UltiDraw.Begin();
        UltiDraw.SetDepthRendering(true);
        for(int i=0; i<Points.Length-Step; i+=Step) {
            if(i<Points.Length-1) {
                UltiDraw.DrawLine(Points[i].Position, Points[i+Step].Position, LineWidth, LineColor);
            }
            UltiDraw.DrawSphere(Points[i].Position, Quaternion.identity, 0.1f, PointColor);
        }
        UltiDraw.SetDepthRendering(false);
        UltiDraw.End();
    }

}
