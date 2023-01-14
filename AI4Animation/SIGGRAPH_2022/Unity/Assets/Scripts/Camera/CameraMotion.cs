using System.Collections;
using UnityEngine;

public class CameraMotion : MonoBehaviour {
    public Transform[] Targets = new Transform[0];

    public bool Draw = true;

    public float StartDelay = 3f;

    public Keypoint[] Keypoints;

    private Vector3 ZeroPosition;
    private Quaternion ZeroRotation;

    [System.Serializable]
    public class Keypoint {
        public Vector3 Position;
        public float Duration;
        public float Offset;
    }

    void Start() {
        ZeroPosition = transform.position;
        ZeroRotation = transform.rotation;
        StartCoroutine(Movement());
    }

    void Update() {
        if(Input.GetKey(KeyCode.Escape)) {
            StopAllCoroutines();
            transform.position = ZeroPosition;
            transform.rotation = ZeroRotation;
            StartCoroutine(Movement());
            GameObject.FindObjectOfType<MotionRecorder>().Index = 300;
        }
    }

    void OnRenderObject() {
        if(!Draw) {
            return;
        }
        UltiDraw.Begin();
        UltiDraw.DrawSphere(transform.position, Quaternion.identity, 0.5f, Color.red);
        for(int i=0; i<Keypoints.Length; i++) {
            UltiDraw.DrawSphere(Keypoints[i].Position, Quaternion.identity, 0.5f, UltiDraw.Red);
            if(i > 0) {
                UltiDraw.DrawLine(Keypoints[i-1].Position, Keypoints[i].Position, UltiDraw.Red);
            }
        }
        UltiDraw.End();
    }

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			OnRenderObject();
		}
	}

    private Vector3 GetTargetPosition() {
        if(Targets.Length == 0) {
            return Vector3.zero;
        }
        Vector3 position = Vector3.zero;
        foreach(Transform t in Targets) {
            position += t.position;
        }
        return position / Targets.Length;
    }

    private IEnumerator Movement() {
        yield return new WaitForSeconds(StartDelay);

        for(int i=0; i<Keypoints.Length; i++) {
            Quaternion rotation = transform.rotation;
            float elapsed = 0f;
            while(elapsed < Keypoints[i].Duration) {
                float ratio = elapsed / Keypoints[i].Duration;

                if(i==0) {
                    ratio = Mathf.Pow(ratio, 2f);
                }

                Vector3 target = GetTargetPosition() + new Vector3(0f, Keypoints[i].Offset, 0f);

                Vector3 p0 = Keypoints[Mathf.Clamp(i-1, 0, Keypoints.Length-1)].Position;
                Vector3 p1 = Keypoints[Mathf.Clamp(i, 0, Keypoints.Length-1)].Position;
                Vector3 p2 = Keypoints[Mathf.Clamp(i+1, 0, Keypoints.Length-1)].Position;
                Vector3 p3 = Keypoints[Mathf.Clamp(i+2, 0, Keypoints.Length-1)].Position;
                transform.position = GetCatmullRomPosition(ratio, p0, p1, p2, p3);
                transform.rotation = Quaternion.Slerp(rotation, Quaternion.LookRotation((target - transform.position).normalized, Vector3.up), (1.0f - Mathf.Pow(1.0f - ratio, 2f)));
                elapsed += 1f/30f;
                yield return 0;
            }
        }

        transform.GetComponent<CameraController>().enabled = true;

        yield return 0;
    }

	private Vector3 GetCatmullRomPosition(float t, Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3) {
		//The coefficients of the cubic polynomial (except the 0.5f * which I added later for performance)
		Vector3 a = 2f * p1;
		Vector3 b = p2 - p0;
		Vector3 c = 2f * p0 - 5f * p1 + 4f * p2 - p3;
		Vector3 d = -p0 + 3f * p1 - 3f * p2 + p3;

		//The cubic polynomial: a + b * t + c * t^2 + d * t^3
		Vector3 pos = 0.5f * (a + (b * t) + (c * t * t) + (d * t * t * t));

		return pos;
	}
}
