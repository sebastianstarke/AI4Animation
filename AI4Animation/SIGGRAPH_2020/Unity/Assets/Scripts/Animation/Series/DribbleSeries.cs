using UnityEngine;

public class DribbleSeries : ComponentSeries {

    public float AreaOfInterest; //Area to for control and interaction radius
    
    public Vector4 Target; //Horizontal X, Height H, Horizontal Z, Speed S
    public Vector3[] Pivots; //Local Space Direction
    public Vector3[] Momentums; //Local Space Direction

    public Matrix4x4[] BallTransformations; //World Space Transformations
    public Vector3[] BallVelocities; //World Space Velocities

    public Actor Actor;
    public RootSeries ActorRoot;
    public Actor Rival;
    public RootSeries RivalRoot;

    public Vector2 HeightParameters = new Vector2(0.25f, 2f);
    public Vector2 SpeedParameters = new Vector2(0f, 4f);

    public Color ReferenceColor = UltiDraw.Gold;
    public Color TargetColor = UltiDraw.Yellow;
    public Color CurrentColor = UltiDraw.Red;
    public Color PivotStartColor = UltiDraw.Red;
    public Color PivotEndColor = UltiDraw.Green;
    public Color MomentumStartColor = UltiDraw.Black;
    public Color MomentumEndColor = UltiDraw.Purple;
    public float PivotStartOpacity = 0.5f;
    public float PivotEndOpacity = 1f;
    public float MomentumStartOpacity = 0.25f;
    public float MomentumEndOpacity = 0.75f;

    private UltiDraw.GUIRect rect = new UltiDraw.GUIRect(0.125f, 0.175f, 0.15f, 0.15f);
    private UltiDraw.GUIRect opponent = new UltiDraw.GUIRect(0.875f, 0.575f, 0.2f, 0.1f);

    public DribbleSeries(TimeSeries global, float area, Transform ball, Actor actor, RootSeries actorRoot, Actor rival, RootSeries rivalRoot) : base(global) {
        AreaOfInterest = area;

        BallTransformations = new Matrix4x4[Samples.Length];
        BallVelocities = new Vector3[Samples.Length];
        Pivots = new Vector3[Samples.Length];
        Momentums = new Vector3[Samples.Length];
        for(int i=0; i<Samples.Length; i++) {
            BallTransformations[i] = ball.transform.GetWorldMatrix(true);
            BallVelocities[i] = Vector3.zero;
            Pivots[i] = Vector3.forward;
            Momentums[i] = Vector3.zero;
        }

        Actor = actor;
        ActorRoot = actorRoot;
        Rival = rival;
        RivalRoot = rivalRoot;
    }

    public UltiDraw.GUIRect GetRect() {
        return rect;
    }

    public float GetBallAmplitude() {
        float value = 0f;
        for(int i=0; i<Pivot; i++) {
            value += BallTransformations[i].GetPosition().y;
        }
        value /= Pivot;
        return value;
    }

    public Vector3 InterpolatePivot(Vector3 from, Vector3 to, float vectorWeight, float heightWeight) {
        float magnitude = Mathf.Lerp(from.ZeroY().magnitude, to.ZeroY().magnitude, vectorWeight);
        Vector3 vector = Vector3.Lerp(from.ZeroY(), to.ZeroY(), vectorWeight).normalized;
        float height = Mathf.Lerp(from.y, to.y, heightWeight);
        return (magnitude * vector).SetY(height);
    }

    public Vector3 InterpolateMomentum(Vector3 from, Vector3 to, float vectorWeight, float heightWeight) {
        return Vector3.Lerp(from.ZeroY(), to.ZeroY(), vectorWeight).SetY(Mathf.Lerp(from.y, to.y, heightWeight));
    }

    public bool IsInsideControlRadius(Vector3 position, Vector3 pivot) {
        return Vector3.Distance(pivot.ZeroY(), position.ZeroY()) < GetControlRadius();
    }

    public float GetControlWeight(int index, Vector3 pivot) {
        float weight = 1f - Vector3.Distance(pivot.ZeroY(), BallTransformations[index].GetPosition().ZeroY()) / GetControlRadius();
        return Mathf.Clamp(weight, 0f, 1f);
    }

    public Vector3 GetBallPosition(int index) {
        return BallTransformations[index].GetPosition();
    }

    public Vector3 GetBallForward(int index) {
        return BallTransformations[index].GetForward();
    }

    public Vector3 GetWeightedBallUp(int index) {
        return BallTransformations[index].GetUp();
    }

    public Vector3 GetBallVelocity(int index) {
        return BallVelocities[index];
    }

    public Vector3 GetWeightedBallPosition(int index, Vector3 pivot) {
        return (GetControlWeight(index, pivot) * (BallTransformations[index].GetPosition() - pivot)) + pivot;
    }

    public Vector3 GetWeightedBallForward(int index, Vector3 pivot) {
        return GetControlWeight(index, pivot) * BallTransformations[index].GetForward();
    }

    public Vector3 GetWeightedBallUp(int index, Vector3 pivot) {
        return GetControlWeight(index, pivot) * BallTransformations[index].GetUp();
    }

    public Vector3 GetWeightedBallVelocity(int index, Vector3 pivot) {
        return GetControlWeight(index, pivot) * BallVelocities[index];
    }

    public float GetInteractorWeight(int index) {
        return GetInteractorDistance(index) < GetInteractionRadius() ? 1f : 0f;
    }

    public Vector3 GetInteractorGradient(int index) {
        if(GetInteractorDistance(index) == GetInteractionRadius()) {
            return Vector3.zero;
        }
        float distance = Mathf.Min(GetInteractorDistance(index), GetInteractionRadius());
        float inverseDistance = GetInteractionRadius() - distance;
        return inverseDistance * RivalRoot.GetPosition(index).GetRelativePositionTo(ActorRoot.Transformations[index]);
    }

    public Vector3 GetInteractorDirection(int index) {
        if(GetInteractorDistance(index) == GetInteractionRadius()) {
            return Vector3.zero;
        }
        return RivalRoot.GetDirection(index).GetRelativeDirectionTo(ActorRoot.Transformations[index]);
    }

    public Vector3 GetInteractorVelocity(int index) {
        if(GetInteractorDistance(index) == GetInteractionRadius()) {
            return Vector3.zero;
        }
        return RivalRoot.GetVelocity(index).GetRelativeDirectionTo(ActorRoot.Transformations[index]);
    }

    public float[] GetInteractorBoneDistances() {
        float[] distances = new float[Actor.Bones.Length];
        for(int i=0; i<distances.Length; i++) {
            distances[i] = GetInteractorBoneDistance(i);
        }
        return distances;
    }

    public float GetInteractorBoneDistance(int index) {
        if(GetInteractorDistance(index) == GetInteractionRadius()) {
            return GetInteractionRadius();
        }
        return Mathf.Clamp(Vector3.Distance(Actor.Bones[index].Transform.position, Rival.Bones[index].Transform.position), 0f, GetInteractionRadius());
    }

    private float GetInteractorDistance(int index) {
        if(Rival == null || RivalRoot == null) {
            return GetInteractionRadius();
        }
        return Mathf.Clamp(Vector3.Distance(ActorRoot.GetPosition(index).ZeroY(), RivalRoot.GetPosition(index).ZeroY()), 0f, GetInteractionRadius());
    }

    public void IncrementBall(int start, int end) {
        for(int i=start; i<end; i++) {
            BallTransformations[i] = BallTransformations[i+1];
            BallVelocities[i] = BallVelocities[i+1];
        }
    }

    public float GetControlRadius() {
        return AreaOfInterest / 2f;
    }

    public float GetInteractionRadius() {
        return 2f * AreaOfInterest;
    }

    public override void Increment(int start, int end) {
        for(int i=start; i<end; i++) {
            Pivots[i] = Pivots[i+1];
            Momentums[i] = Momentums[i+1];
        }
    }

    public override void Interpolate(int start, int end) {
        for(int i=start; i<end; i++) {
            float weight = (float)(i % Resolution) / (float)Resolution;
            int prevIndex = GetPreviousKey(i).Index;
            int nextIndex = GetNextKey(i).Index;
            Pivots[i] = InterpolatePivot(Pivots[prevIndex], Pivots[nextIndex], weight, weight); 
            Momentums[i] = InterpolateMomentum(Momentums[prevIndex], Momentums[nextIndex], weight, weight);
        }
    }

    public override void GUI(Camera canvas=null) {
        if(DrawGUI) {
            UltiDraw.Begin(canvas);
            UltiDraw.OnGUILabel(rect.GetCenter() + rect.ToScreen(new Vector2(0f, 1.25f)), rect.ToScreen(new Vector2(1f, 0.25f)), 0.02f, "Ball Control", UltiDraw.Black);
            UltiDraw.OnGUILabel(rect.GetCenter() + rect.ToScreen(new Vector2(-1.175f, 1f)), rect.ToScreen(new Vector2(1f, 0.25f)), 0.015f, "Height", UltiDraw.Black);
            UltiDraw.OnGUILabel(rect.GetCenter() + rect.ToScreen(new Vector2(1.175f, 1f)), rect.ToScreen(new Vector2(1f, 0.25f)), 0.015f, "Speed", UltiDraw.Black);
            UltiDraw.OnGUILabel(opponent.GetCenter() + new Vector2(0f, opponent.H+0.02f), rect.ToScreen(new Vector2(1f, 0.25f)), 0.0175f, "Opponent", UltiDraw.Black);
            UltiDraw.End();
        }
    }

    public override void Draw(Camera canvas=null) {
        Matrix4x4 GetRoot(int index) {
            return ActorRoot.Transformations[index];
        }

        UltiDraw.Begin(canvas);

        // //Weighted Balls
        // for(int i=1; i<=Pivot; i++) {
        //     float weight = Mathf.Pow((float)(i) / (float)Pivot, 0.25f);
        //     UltiDraw.DrawLine(GetWeightedBallPosition(i-1), GetWeightedBallPosition(i), UltiDraw.Magenta.Opacity(weight));
        // }
        // for(int i=0; i<=Pivot; i++) {
        //     float weight = Mathf.Pow((float)(i+1) / (float)(Pivot+1), 0.25f);
        //     UltiDraw.DrawSphere(GetWeightedBallPosition(i), Quaternion.identity, 0.025f, UltiDraw.Black.Opacity(weight));
        //     if(GetControlWeight(i) > 0.001f) {
        //         UltiDraw.DrawTranslateGizmo(GetWeightedBallPosition(i), Quaternion.LookRotation(GetWeightedBallForward(i).normalized, GetWeightedBallUp(i).normalized), 0.25f * GetControlWeight(i));
        //     }
        //     UltiDraw.DrawArrow(GetWeightedBallPosition(i), GetWeightedBallPosition(i) + GetTemporalScale(GetWeightedBallVelocity(i)), 0.8f, 0.025f, 0.05f, UltiDraw.Black.Opacity(weight));
        // }
        // float[] values = new float[Samples.Length];
        // for(int i=0; i<values.Length; i++) {
        //     values[i] = GetControlWeight(i);
        // }
        // UltiDraw.PlotFunction(new Vector2(0.125f, 0.425f), new Vector2(0.225f, 0.1f), values, 0f, 1f);

        //World-Space Balls
        // for(int i=1; i<=Pivot; i++) {
        //     float weight = Mathf.Pow((float)(i) / (float)Pivot, 0.25f);
        //     UltiDraw.DrawLine(BallTransformations[i-1].GetPosition(), BallTransformations[i].GetPosition(), UltiDraw.Black.Opacity(weight));
        // }
        // for(int i=0; i<=Pivot; i++) {
        //     float weight = Mathf.Pow((float)(i+1) / (float)(Pivot+1), 0.25f);
        //     UltiDraw.DrawSphere(BallTransformations[i].GetPosition(), Quaternion.identity, 0.025f, UltiDraw.Magenta.Opacity(weight));
        //     UltiDraw.DrawLine(BallTransformations[i].GetPosition(), BallTransformations[i].GetPosition()+GetTemporalScale(BallVelocities[i]), 0.025f, 0f, UltiDraw.Red.Opacity(weight));
        // }

        //Debug Interaction Sphere
        // UltiDraw.DrawWireHemisphere(GetRoot(Pivot).GetPosition(), GetRoot(Pivot).GetRotation(), 2f*GetInteractionRadius(), UltiDraw.DarkGrey.Opacity(0.25f));

        //Control
        float controlWeight = GetControlWeight(Pivot, GetRoot(Pivot).GetPosition()).SmoothStep(1f, 0.1f);
        Color GetPivotColor(int index) {
            float weight = Mathf.Sqrt((float)(index+1) / (float)Samples.Length);
            return PivotStartColor.Lerp(PivotEndColor, weight).Lerp(UltiDraw.DarkGrey, 1f-controlWeight).Opacity(weight.Normalize(0f, 1f, PivotStartOpacity, PivotEndOpacity));
        }
        Color GetMomentumColor(int index) {
            float weight = Mathf.Sqrt((float)(index+1) / (float)Samples.Length);
            return MomentumStartColor.Lerp(MomentumEndColor, weight).Lerp(UltiDraw.DarkGrey, 1f-controlWeight).Opacity(weight.Normalize(0f, 1f, MomentumStartOpacity, MomentumEndOpacity));
        }
        if(DrawGUI) {
            //Image Space
            UltiDraw.GUICircle(rect.GetCenter(), rect.W, UltiDraw.DarkGrey.Opacity(0.8f));
            UltiDraw.GUICircle(rect.GetCenter() + rect.ToScreen(new Vector2(0f, 1f)), 0.01f, ReferenceColor);
            UltiDraw.GUICircle(rect.GetCenter() + rect.ToScreen(new Vector2(0f, -1f)), 0.01f, ReferenceColor);
            UltiDraw.GUICircle(rect.GetCenter() + rect.ToScreen(new Vector2(1f, 0f)), 0.01f, ReferenceColor);
            UltiDraw.GUICircle(rect.GetCenter() + rect.ToScreen(new Vector2(-1f, 0f)), 0.01f, ReferenceColor);
            int step = Resolution;
            for(int i=0; i<Samples.Length; i+=step) {
                Vector3 current = rect.GetCenter() + rect.ToScreen(new Vector2(Pivots[i].x, Pivots[i].z));
                Vector3 target = rect.GetCenter() + rect.ToScreen(new Vector2(Pivots[i].x, Pivots[i].z) + GetTemporalScale(new Vector2(Momentums[i].x, Momentums[i].z)));
                if(i < Samples.Length-step) {
                    Vector3 next = rect.GetCenter() + rect.ToScreen(new Vector2(Pivots[i+step].x, Pivots[i+step].z));
                    UltiDraw.GUILine(current, next, UltiDraw.Red);
                }
                UltiDraw.GUICircle(current, 0.01f, GetPivotColor(i));
                UltiDraw.GUILine(current, target, GetMomentumColor(i));
            }
            UltiDraw.PlotVerticalPivot(rect.GetCenter() + rect.ToScreen(new Vector2(-1.25f, 0f)), rect.ToScreen(new Vector2(0.125f, 1.5f)), Pivots[Pivot].y.Normalize(HeightParameters.x, HeightParameters.y, 0f, 1f), backgroundColor:UltiDraw.DarkGrey, pivotColor:UltiDraw.Green);
            UltiDraw.PlotVerticalPivot(rect.GetCenter() + rect.ToScreen(new Vector2(1.25f, 0f)), rect.ToScreen(new Vector2(0.125f, 1.5f)), Momentums[Pivot].y.Normalize(SpeedParameters.x, SpeedParameters.y, 0f, 1f), backgroundColor:UltiDraw.DarkGrey, pivotColor:UltiDraw.Magenta);

            UltiDraw.GUILine(rect.GetCenter(), rect.GetCenter() + rect.ToScreen(new Vector2(Target.x, Target.z)), 0.01f*Target.MagnitudeXZ(), 0f, TargetColor);
        }
        if(DrawScene) {
            //World Space
            Color circleColor = UltiDraw.DarkGrey.Opacity(0.25f);
            Color wireColor = Color.Lerp(UltiDraw.DarkGrey, UltiDraw.IndianRed, controlWeight);
            Color referenceColor = Color.Lerp(UltiDraw.DarkGrey, UltiDraw.Mustard, controlWeight).Opacity(0.5f);
            UltiDraw.DrawCircle(GetRoot(Pivot).GetPosition(), Quaternion.Euler(90f, 0f, 0f), 2f*GetControlRadius(), circleColor);
            UltiDraw.DrawWireCircle(GetRoot(Pivot).GetPosition(), Quaternion.Euler(90f, 0f, 0f), 2f*GetControlRadius(), wireColor);
            UltiDraw.DrawCircle(GetRoot(Pivot).GetPosition() + GetControlRadius()*Vector3.forward.GetRelativeDirectionFrom(GetRoot(Pivot)), 0.05f, referenceColor);
            UltiDraw.DrawCircle(GetRoot(Pivot).GetPosition() + GetControlRadius()*Vector3.right.GetRelativeDirectionFrom(GetRoot(Pivot)), 0.05f, referenceColor);
            UltiDraw.DrawCircle(GetRoot(Pivot).GetPosition() + GetControlRadius()*Vector3.left.GetRelativeDirectionFrom(GetRoot(Pivot)), 0.05f, referenceColor);
            UltiDraw.DrawCircle(GetRoot(Pivot).GetPosition() + GetControlRadius()*Vector3.back.GetRelativeDirectionFrom(GetRoot(Pivot)), 0.05f, referenceColor);
            UltiDraw.DrawLine(GetRoot(Pivot).GetPosition(), GetRoot(Pivot).GetPosition() + GetControlRadius()*new Vector3(Target.x, 0f, Target.z).GetRelativeDirectionFrom(GetRoot(Pivot)), GetRoot(Pivot).GetUp(), 0.1f*Target.MagnitudeXZ(), 0f, Color.Lerp(UltiDraw.DarkGrey, UltiDraw.Yellow, controlWeight).Opacity(0.5f));

            //Pivots and Momentums
            for(int i=0; i<Samples.Length; i+=Resolution) {
                float size = Mathf.Sqrt((float)(i+1) / (float)Samples.Length);
                Vector3 location = (GetRoot(Pivot).GetPosition() + GetControlRadius()*Pivots[i].GetRelativeDirectionFrom(GetRoot(Pivot))).ZeroY();
                Vector3 momentum = GetTemporalScale(Momentums[i].GetRelativeDirectionFrom(GetRoot(Pivot)).ZeroY());
                UltiDraw.DrawSphere(location, Quaternion.identity, size * 0.05f, GetPivotColor(i));
                UltiDraw.DrawArrow(location, location + momentum, 0.8f, 0.0125f, 0.025f, GetMomentumColor(i));
            }
            //Connections
            for(int i=0; i<Samples.Length-1; i++) {
                float wPrev = (float)i / (float)(Samples.Length-1);
                float wNext = (float)(i+1) / (float)(Samples.Length-1);
                Vector3 prev = (GetRoot(Pivot).GetPosition() + wPrev*GetControlRadius()*Pivots[i].GetRelativeDirectionFrom(GetRoot(Pivot))).SetY(Pivots[i].y);
                Vector3 next = (GetRoot(Pivot).GetPosition() + wNext*GetControlRadius()*Pivots[i+1].GetRelativeDirectionFrom(GetRoot(Pivot))).SetY(Pivots[i+1].y);
                UltiDraw.DrawLine(prev, next, UltiDraw.DarkGrey);
            }
            //Heights and Speeds
            for(int i=0; i<Samples.Length; i+=Resolution) {
                float weight = (float)i / (float)(Samples.Length-1);
                float size = Mathf.Sqrt((float)(i+1) / (float)Samples.Length);
                Vector3 location = GetRoot(Pivot).GetPosition() + weight*GetControlRadius()*Pivots[i].GetRelativeDirectionFrom(GetRoot(Pivot)).ZeroY();
                Vector3 prev = location.SetY(Pivots[i].y);
                UltiDraw.DrawSphere(prev, Quaternion.identity, size * 0.05f, GetPivotColor(i));
                UltiDraw.DrawArrow(prev, location.SetY(Pivots[i].y - 0.5f*GetTemporalScale(Momentums[i].y)), 0.8f, 0.0125f, 0.025f, GetMomentumColor(i));
                UltiDraw.DrawArrow(prev, location.SetY(Pivots[i].y + 0.5f*GetTemporalScale(Momentums[i].y)), 0.8f, 0.0125f, 0.025f, GetMomentumColor(i));
            }
        }

        //Interaction
        if(DrawGUI) {
            float[] weights = new float[SampleCount];
            for(int i=0; i<Samples.Length; i++) {
                weights[i] = GetInteractorWeight(i);
            }
            UltiDraw.PlotFunction(new Vector2(opponent.X, opponent.Y+2.5f*opponent.H/3f), new Vector2(opponent.W, opponent.H/3f), weights, yMin: 0f, yMax: 1f, thickness:0.0025f);

            float padding = 1.1f;
            Vector3[] gradients = new Vector3[SampleCount];
            for(int i=0; i<SampleCount; i++) {
                gradients[i] = GetInteractorGradient(i);
            }
            UltiDraw.PlotFunctions(new Vector2(opponent.X-opponent.W/3f, opponent.Y+1.5f*opponent.H/3f), new Vector2(opponent.W/3f, opponent.H/3f), gradients, yMin: -padding*GetInteractionRadius(), yMax: padding*GetInteractionRadius(), thickness:0.0025f);
            Vector3[] directions = new Vector3[SampleCount];
            for(int i=0; i<SampleCount; i++) {
                directions[i] = GetInteractorDirection(i);
            }
            UltiDraw.PlotFunctions(new Vector2(opponent.X, opponent.Y+1.5f*opponent.H/3f), new Vector2(opponent.W/3f, opponent.H/3f), directions, yMin: -padding*1f, yMax: padding*1f, thickness:0.0025f);
            Vector3[] velocities = new Vector3[SampleCount];
            for(int i=0; i<SampleCount; i++) {
                velocities[i] = GetInteractorVelocity(i);
            }
            UltiDraw.PlotFunctions(new Vector2(opponent.X+opponent.W/3f, opponent.Y+1.5f*opponent.H/3f), new Vector2(opponent.W/3f, opponent.H/3f), velocities, yMin: -padding*5f, yMax: padding*5f, thickness:0.0025f);

            float[] magnitudes = new float[Actor.Bones.Length];
            for(int i=0; i<Actor.Bones.Length; i++) {
                magnitudes[i] = GetInteractorBoneDistance(i);
            }
            UltiDraw.PlotBars(new Vector2(opponent.X, opponent.Y+0.5f*opponent.H/3f), new Vector2(opponent.W, opponent.H/3f), magnitudes, yMin: 0f, yMax: GetInteractionRadius());
        }
        if(DrawScene) {
            if(Rival != null && RivalRoot != null) {
                //Trajectory
                for(int i=0; i<Samples.Length; i++) {
                    float weight = GetInteractorWeight(i);
                    Vector3 pivot = RivalRoot.GetPosition(i);
                    UltiDraw.DrawLine(GetRoot(i).GetPosition(), pivot, Vector3.up, 0.05f, 0f, UltiDraw.Blue.Opacity(0.25f*weight));
                    UltiDraw.DrawLine(pivot, pivot + 0.25f*RivalRoot.GetDirection(i), Vector3.up, 0.05f, 0f, UltiDraw.Red.Opacity(0.25f*weight));
                    UltiDraw.DrawLine(pivot, pivot + GetTemporalScale(RivalRoot.GetVelocity(i)), Vector3.up, 0.05f, 0f, UltiDraw.Red.Opacity(0.25f*weight));
                    UltiDraw.DrawSphere(pivot, Quaternion.identity, 0.05f*weight, Color.red);
                }
                {
                    //Bone Distances
                    float weight = GetInteractorWeight(Pivot);
                    for(int i=0; i<Actor.Bones.Length; i++) {
                        UltiDraw.DrawLine(Actor.Bones[i].Transform.position, Rival.Bones[i].Transform.position, UltiDraw.Red.Opacity(0.25f*weight), UltiDraw.Black.Opacity(0.25f*weight));
                    }
                }
            }
        }

        UltiDraw.End();
    }
}