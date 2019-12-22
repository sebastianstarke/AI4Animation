#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

public class Transformation : SceneEvent {

	public Vector3 MinPositionNoise = Vector3.zero;
	public Vector3 MaxPositionNoise = Vector3.zero;
	public Vector3 MinRotationNoise = Vector3.zero;
	public Vector3 MaxRotationNoise = Vector3.zero;
	public Vector3 MinScaleNoise = Vector3.one;
	public Vector3 MaxScaleNoise = Vector3.one;
	public Matrix4x4[] Transformations;

	[ContextMenu("Fix")]
	public void Fix() {
		for(int i=0; i<Transformations.Length; i++) {
			Matrix4x4Extensions.SetScale(ref Transformations[i], new Vector3(1.5f, 1f, 1f));
		}
	}

	public override void Callback(MotionEditor editor) {
        if(Blocked) {
            Identity(editor);
            return;
        }
		Matrix4x4 t = GetTransformation(editor.GetCurrentFrame(), editor.Mirror);
		transform.position = t.GetPosition();
		transform.rotation = t.GetRotation();
		transform.localScale = t.GetScale();
	}

	public override void Identity(MotionEditor editor) {
		Matrix4x4 t = GetRawTransformation(editor.GetCurrentFrame(), editor.Mirror);
		transform.position = t.GetPosition();
		transform.rotation = t.GetRotation();
		transform.localScale = t.GetScale();
	}

	public void Setup(MotionData data) {
		Transformations = new Matrix4x4[data.Frames.Length];
		for(int i=0; i<Transformations.Length; i++) {
			Transformations[i] = Matrix4x4.identity;
		}
	}

	public Matrix4x4 GetRawTransformation(Frame frame, bool mirrored) {
		return (mirrored ? Transformations[frame.Index-1].GetMirror(frame.Data.MirrorAxis) : Transformations[frame.Index-1]);
	}

	public Matrix4x4 GetTransformation(Frame frame, bool mirrored) {
		return (mirrored ? Transformations[frame.Index-1].GetMirror(frame.Data.MirrorAxis) : Transformations[frame.Index-1]) * GetNoiseMatrix();
	}

	public Matrix4x4 GetNoiseMatrix() {
		MotionEditor editor = GameObject.FindObjectOfType<MotionEditor>();
		if(editor == null) {
			Debug.Log("Motion editor could not be found inside Transformation event. This should definitely not have happened.");
		}
		if(editor.Callbacks) {
			Random.InitState(editor.GetCurrentSeed());
			return Matrix4x4.TRS(
				Utility.UniformVector3(MinPositionNoise, MaxPositionNoise), 
				Quaternion.Euler(Utility.UniformVector3(MinRotationNoise, MaxRotationNoise)),
				Utility.UniformVector3(MinScaleNoise, MaxScaleNoise)
			);
		} else {
			return Matrix4x4.identity;
		}
	}

}
#endif