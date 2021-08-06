using UnityEngine;

public class RootSeries : ComponentSeries {
    
    public Matrix4x4[] Transformations;
    public Vector3[] Velocities;

    public bool CollisionsResolved;

    public RootSeries(TimeSeries global) : base(global) {
        Transformations = new Matrix4x4[Samples.Length];
        Velocities = new Vector3[Samples.Length];
        for(int i=0; i<Samples.Length; i++) {
            Transformations[i] = Matrix4x4.identity;
            Velocities[i] = Vector3.zero;
        }
    }

    public RootSeries(TimeSeries global, Transform transform) : base(global) {
        Transformations = new Matrix4x4[Samples.Length];
        Velocities = new Vector3[Samples.Length];
        Matrix4x4 root = transform.GetWorldMatrix(true);
        for(int i=0; i<Samples.Length; i++) {
            Transformations[i] = root;
            Velocities[i] = Vector3.zero;
        }
    }

    public void SetTransformation(int index, Matrix4x4 transformation) {
        Transformations[index] = transformation;
    }

    public Matrix4x4 GetTransformation(int index) {
        return Transformations[index];
    }

    public void SetPosition(int index, Vector3 position) {
        Matrix4x4Extensions.SetPosition(ref Transformations[index], position);
    }

    public Vector3 GetPosition(int index) {
        return Transformations[index].GetPosition();
    }

    public void SetRotation(int index, Quaternion rotation) {
        Matrix4x4Extensions.SetRotation(ref Transformations[index], rotation);
    }

    public Quaternion GetRotation(int index) {
        return Transformations[index].GetRotation();
    }

    public void SetDirection(int index, Vector3 direction) {
        Matrix4x4Extensions.SetRotation(ref Transformations[index], Quaternion.LookRotation(direction == Vector3.zero ? Vector3.forward : direction, Vector3.up));
    }

    public Vector3 GetDirection(int index) {
        return Transformations[index].GetForward();
    }

    public void SetVelocity(int index, Vector3 velocity) {
        Velocities[index] = velocity;
    }

    public Vector3 GetVelocity(int index) {
        return Velocities[index];
    }

    public void Translate(int index, Vector3 delta) {
        SetPosition(index, GetPosition(index) + delta);
    }

    public void Rotate(int index, Quaternion delta) {
        SetRotation(index, GetRotation(index) * delta);
    }

    public void Rotate(int index, float angles, Vector3 axis) {
        Rotate(index, Quaternion.AngleAxis(angles, axis));
    }

    public void ResolveCollisions(float safety, LayerMask mask) {
        CollisionsResolved = false;
        for(int i=Pivot; i<Samples.Length; i++) {
            Vector3 previous = GetPosition(i-1);
            Vector3 current = GetPosition(i);
            RaycastHit hit;
            if(Physics.Raycast(previous, (current-previous).normalized, out hit, Vector3.Distance(current, previous), mask)) {
                //This makes sure no point would ever fall into a geometry volume by projecting point i to i-1
                for(int j=i; j<Samples.Length; j++) {
                    SetPosition(j, GetPosition(j-1));
                }
            }
            //This generates a safety-slope around objects over multiple frames in a waterflow-fashion
            Vector3 corrected = SafetyProjection(GetPosition(i));
            if(corrected != current) {
                CollisionsResolved = true;
            }
            SetPosition(i, corrected);
            SetVelocity(i, GetVelocity(i) + (corrected-current) / (Samples[i].Timestamp - Samples[i-1].Timestamp));
        }

        Vector3 SafetyProjection(Vector3 pivot) {
            Vector3 point = Utility.GetClosestPointOverlapSphere(pivot, safety, mask);
            return point + safety * (pivot - point).normalized;
        }
    }

    public override void Increment(int start, int end) {
        for(int i=start; i<end; i++) {
            Transformations[i] = Transformations[i+1];
            Velocities[i] = Velocities[i+1];
        }
    }

    public override void Interpolate(int start, int end) {
        for(int i=start; i<end; i++) {
            float weight = (float)(i % Resolution) / (float)Resolution;
            int prevIndex = GetPreviousKey(i).Index;
            int nextIndex = GetNextKey(i).Index;
            if(prevIndex != nextIndex) {
                SetPosition(i, Vector3.Lerp(GetPosition(prevIndex), GetPosition(nextIndex), weight));
                SetDirection(i, Vector3.Lerp(GetDirection(prevIndex), GetDirection(nextIndex), weight).normalized);
                SetVelocity(i, Vector3.Lerp(GetVelocity(prevIndex), GetVelocity(nextIndex), weight));
            }
        }

        //for(int i=start; i<end; i++) {
        //	float weight = (float)(i % Resolution) / (float)Resolution;
        //	float amount = 1f - GetWeight(i);
        //	int prevIndex = GetPreviousKey(i).Index;
        //	int nextIndex = GetNextKey(i).Index;
        //	if(prevIndex != nextIndex) {
        //		SetPosition(i, Vector3.Lerp(GetPosition(i), Vector3.Lerp(GetPosition(prevIndex), GetPosition(nextIndex), weight), amount));
        //		SetDirection(i, Vector3.Slerp(GetDirection(i), Vector3.Slerp(GetDirection(prevIndex), GetDirection(nextIndex), weight), amount));
        //		SetVelocity(i, Vector3.Lerp(GetVelocity(i), Vector3.Lerp(GetVelocity(prevIndex), GetVelocity(nextIndex), weight), amount));
        //	}
        //}
    }

    public override void GUI(Camera canvas=null) {
        
    }

    public override void Draw(Camera canvas=null) {
        if(DrawScene) {
            UltiDraw.Begin(canvas);

            float size = 2f;
            int step = Resolution;

            //Connections
            for(int i=0; i<Transformations.Length-step; i+=step) {
                UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i+step].GetPosition(), Transformations[i].GetUp(), size*0.01f, UltiDraw.Black);
            }

            //Positions
            for(int i=0; i<Transformations.Length; i+=step) {
                UltiDraw.DrawCircle(Transformations[i].GetPosition(), size*0.025f, i % Resolution == 0 ? UltiDraw.Black : UltiDraw.Red.Opacity(0.5f));
            }

            //Directions
            for(int i=0; i<Transformations.Length; i+=step) {
                UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i].GetPosition() + 0.25f*Transformations[i].GetForward(), Transformations[i].GetUp(), size*0.025f, 0f, UltiDraw.Orange.Opacity(0.75f));
            }

            //Velocities
            for(int i=0; i<Velocities.Length; i+=step) {
                UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i].GetPosition() + GetTemporalScale(Velocities[i]), Transformations[i].GetUp(), size*0.0125f, 0f, UltiDraw.DarkGreen.Opacity(0.25f));
            }
            
            //Target
            // UltiDraw.DrawSphere(TargetPosition, Quaternion.identity, 0.25f, UltiDraw.Black);
            // UltiDraw.DrawLine(TargetPosition, TargetPosition + 0.25f*TargetDirection, Vector3.up, size*0.05f, 0f, UltiDraw.Orange);
            // UltiDraw.DrawLine(TargetPosition, TargetPosition + TargetVelocity, Vector3.up, size*0.025f, 0f, UltiDraw.DarkGreen);

            UltiDraw.End();
        }
    }
}