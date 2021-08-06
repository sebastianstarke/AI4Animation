using UnityEngine;

[ExecuteInEditMode]
public class Ball : MonoBehaviour {

    public float Radius = 1f;

    private Rigidbody RB;

    private Rigidbody GetRigidbody() {
        if(RB == null) {
            RB = GetComponent<Rigidbody>();
        }
        return RB;
    }

    public void SetPosition(Vector3 position) {
        transform.position = position;
    }

    public Vector3 GetPosition() {
        return transform.position;
    }

    public void SetRotation(Quaternion rotation) {
        transform.rotation = rotation;
    }

    public Quaternion GetRotation() {
        return transform.rotation;
    }

    public void SetVelocity(Vector3 velocity) {
        GetRigidbody().velocity = velocity;
    }
    
    public Vector3 GetVelocity() {
        return GetRigidbody().velocity;
    }

    void Draw() {
        // UltiDraw.Begin();
        // UltiDraw.DrawWireSphere(GetPosition(), GetRotation(), 2f*Radius, UltiDraw.Purple);
        // // UltiDraw.DrawArrow(GetPosition(), GetPosition() + GetVelocity(), 0.8f, 0.01f, 0.025f, UltiDraw.DarkGreen.Opacity(0.25f));
        // UltiDraw.End();
    }

	void OnRenderObject() {
		Draw();
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			OnRenderObject();
		}
	}

}
