using UnityEngine;

public class ResolveCollision : MonoBehaviour {
    
    private Ball Ball;
    private CapsuleCollider Collider;

    void Awake() {
        Ball = GameObject.FindObjectOfType<Ball>();
        Collider = GetComponent<CapsuleCollider>();
    }

    void ResolveBallCollisions() {
        Vector3 position = Ball.transform.position;
        Vector3 closest = Collider.ClosestPoint(position);
        if(Vector3.Distance(closest, position) < Ball.Radius) {
            Ball.SetPosition(closest + Ball.Radius * (position - closest).normalized);
            Ball.SetVelocity(Ball.GetVelocity() + (Ball.GetPosition() - position));
        }
    }
}
