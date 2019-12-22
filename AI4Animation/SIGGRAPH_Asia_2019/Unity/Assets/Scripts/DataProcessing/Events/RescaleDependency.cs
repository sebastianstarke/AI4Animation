#if UNITY_EDITOR
using UnityEngine;

[RequireComponent(typeof(VoxelCollider))]
public class RescaleDependency : SceneEvent {

    public Transformation Self;
	public Transformation Dependency;
    public bool X, Y, Z;

	public override void Callback(MotionEditor editor) {
        if(Blocked) {
            Identity(editor);
            return;
        }
        if(Self != null && Dependency != null) {
            Vector3 scale = Self.GetTransformation(editor.GetCurrentFrame(), editor.Mirror).GetScale();
            Matrix4x4 matrix = Dependency.GetTransformation(editor.GetCurrentFrame(), editor.Mirror).GetRelativeTransformationTo(Dependency.GetRawTransformation(editor.GetCurrentFrame(), editor.Mirror));
            Vector3 size = Vector3.Scale(Vector3.Scale(transform.parent.lossyScale, scale), GetComponent<VoxelCollider>().GetExtents());
            Vector3 delta = matrix.GetPosition();
            if(X) {
                scale.x *= (size.x + delta.x) / size.x;
            }
            if(Y) {
                scale.y *= (size.y + delta.y) / size.y;
            }
            if(Z) {
                scale.z *= (size.z + delta.z) / size.z;
            }
            transform.localScale = scale;
        }
	}

	public override void Identity(MotionEditor editor) {
		if(Self != null) {
            transform.localScale = Self.GetRawTransformation(editor.GetCurrentFrame(), editor.Mirror).GetScale();
        }
	}

}
#endif