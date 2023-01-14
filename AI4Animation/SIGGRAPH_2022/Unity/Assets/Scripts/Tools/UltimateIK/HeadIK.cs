using UnityEngine;
using UltimateIK;

public class HeadIK : MonoBehaviour {
    public bool AutoUpdate = true;
    public bool Draw = true;
    public float Distance = 1f;
    public float View = 90f;
    public float Power = 2f;
    public Axis Forward = Axis.ZPositive;
    public Axis Up = Axis.YPositive;
    public bool ForceUpright = false;
    public float HeightFalloff = 2f;
    public Transform Root;
    public Transform Neck;
    public Transform Head;
    public Transform Target;
    
    private IK Solver = null;
    private bool Initialized = false;

    void Update() {
        if(AutoUpdate) {
            Solve();
        }
    }

    public void Solve() {
        Solve(Target.position);
    }

    public void Solve(Vector3 target) {
        if(!Initialized) {
            Solver = IK.Create(Neck, Head);
            Solver.Iterations = 25;
            Initialized = true;
        }

        float horizontalDistanceWeight = 1f - Mathf.Clamp(Vector3.Distance(Root.position.ZeroY(), target.ZeroY()) / Distance, 0f, 1f);

        float horizontalAngleWeight = 1f - Mathf.Min(Vector3.Angle(
            (target.ZeroY() - Root.position.ZeroY()).normalized, 
            Root.rotation.GetForward()
        ), View) / View;
        
        float h = Mathf.Max(Target.position.y - Head.position.y, 0f);
        float falloffWeight = 1f - Mathf.Clamp(h, 0f, 1f);

        float weight = Mathf.Min(
            Mathf.Pow(horizontalDistanceWeight, Power),
            Mathf.Pow(horizontalAngleWeight, Power),
            Mathf.Pow(falloffWeight, HeightFalloff)
        );

        Solver.Objectives.First().TargetPosition = Head.position;
        Quaternion targetHeadRotation =
            Quaternion.FromToRotation(
                Head.rotation * Forward.GetAxis(), 
                (target - Head.position).normalized
            ) * Head.rotation;
        if(ForceUpright) {
            targetHeadRotation = Quaternion.LookRotation(
                targetHeadRotation.GetForward(), 
                Up.GetAxis().DirectionFrom(Root.GetWorldMatrix()));
        }
        Solver.Objectives.First().TargetRotation = 
            Quaternion.Slerp(
                Head.rotation, 
                targetHeadRotation,
                weight
            );
        Solver.Solve();
    }

    public float GetHorizontalAngle() {
        return Vector3.SignedAngle(
            Root.rotation.GetForward().ZeroY(),
            (Target.position-Root.position).ZeroY(),
            Vector3.up
        );
    }

    public float GetVerticalAngle() {
        return Vector3.SignedAngle(
            Vector3.forward,
            new Vector3(0f, Target.position.y-Head.position.y, Vector3.Distance(Head.position.ZeroY(), Target.position.ZeroY())),
            Vector3.right            
        );
    }

    void OnDrawGizmos() {
        if(!Application.isPlaying) {
            OnRenderObject();
        }
    }

    void OnRenderObject() {
        if(!Draw) {
            return;
        }
        UltiDraw.Begin();
        // if(Head != null) {
        //     UltiDraw.DrawSphere(Head.position, Quaternion.identity, 2f*Safety, UltiDraw.Gold.Opacity(0.25f));
        // }
        if(Root != null) {
            UltiDraw.DrawCircle(Root.position.ZeroY(), Quaternion.Euler(90f, 0f, 0f), 2f*Distance, UltiDraw.Magenta.Opacity(0.1f));
        }
        UltiDraw.End();
    }
}
