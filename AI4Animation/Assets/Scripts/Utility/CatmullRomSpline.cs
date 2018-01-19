using UnityEngine;
using System.Collections;

public class CatmullRomSpline : MonoBehaviour {

	public Trajectory Trajectory;
	public Transform[] ControlPoints;

	void OnRenderObject() {
		if(IsInvalid()) {
			return;
		}

		Create();
		Trajectory.Draw(10);
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			OnRenderObject();
		}
	}

	private bool IsInvalid() {
		if(ControlPoints.Length < 4) {
			return true;
		}
		for(int i=0; i<ControlPoints.Length; i++) {
			if(ControlPoints[i] == null) {
				return true;
			}
		}
		return false;
	}

	private void Create() {
		Trajectory = new Trajectory(ControlPoints.Length * 60, 0);
		//UnityGL.Start();
		for(int pos=0; pos<ControlPoints.Length; pos++) {
			Vector3 p0 = ControlPoints[ClampListPos(pos - 1)].position;
			Vector3 p1 = ControlPoints[pos].position;
			Vector3 p2 = ControlPoints[ClampListPos(pos + 1)].position;
			Vector3 p3 = ControlPoints[ClampListPos(pos + 2)].position;
			Vector3 lastPos = p1;
			int loops = 60;
			for(int i=1; i<=loops; i++) {
				float t = i / (float)loops;
				Vector3 newPos = GetCatmullRomPosition(t, p0, p1, p2, p3);
				Trajectory.Points[pos * 60 + i -1].SetPosition(newPos);
				Trajectory.Points[pos * 60 + i -1].SetDirection(newPos-lastPos);
				//UnityGL.DrawLine(lastPos, newPos, Color.white);
				lastPos = newPos;
				//UnityGL.DrawCircle(newPos, 0.01f, Color.black);
			}
		}
		for(int i=0; i<ControlPoints.Length; i++) {
			//UnityGL.DrawCircle(ControlPoints[i].position, 0.025f, Color.cyan);
		}
		//UnityGL.Finish();
	}

	private int ClampListPos(int pos) {
		if(pos < 0) {
			pos = ControlPoints.Length - 1;
		}
		if(pos > ControlPoints.Length) {
			pos = 1;
		} else if(pos > ControlPoints.Length - 1) {
			pos = 0;
		}
		return pos;
	}

	private Vector3 GetCatmullRomPosition(float t, Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3) {
		Vector3 a = 2f * p1;
		Vector3 b = p2 - p0;
		Vector3 c = 2f * p0 - 5f * p1 + 4f * p2 - p3;
		Vector3 d = -p0 + 3f * p1 - 3f * p2 + p3;
		Vector3 pos = 0.5f * (a + (b * t) + (c * t * t) + (d * t * t * t));
		return pos;
	}
}