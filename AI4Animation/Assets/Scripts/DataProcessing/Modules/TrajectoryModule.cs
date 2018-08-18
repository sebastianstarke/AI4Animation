#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class TrajectoryModule : Module {

	public override TYPE Type() {
		return TYPE.Trajectory;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		Inspect = true;

		return this;
	}

	public Trajectory GetTrajectory(Frame frame, bool mirrored) {
		StyleModule styleModule = Data.GetModule(Module.TYPE.Style) == null ? null : (StyleModule)Data.GetModule(Module.TYPE.Style);
		PhaseModule phaseModule = Data.GetModule(Module.TYPE.Phase) == null ? null : (PhaseModule)Data.GetModule(Module.TYPE.Phase);
		
		Trajectory trajectory = new Trajectory(12, styleModule == null ? new string[0] : styleModule.GetNames());

		//Current
		trajectory.Points[6].SetTransformation(frame.GetRootTransformation(mirrored));
		trajectory.Points[6].SetVelocity(frame.GetRootVelocity(mirrored));
		trajectory.Points[6].SetSpeed(frame.GetSpeed(mirrored));
		trajectory.Points[6].Styles = styleModule == null ? new float[0] : styleModule.GetStyle(frame);
		trajectory.Points[6].Phase = phaseModule == null ? 0f : phaseModule.GetPhase(frame, mirrored);

		//Past
		for(int i=0; i<6; i++) {
			float delta = -1f + (float)i/6f;
			if(frame.Timestamp + delta < 0f) {
				float pivot = - frame.Timestamp - delta;
				float clamped = Mathf.Clamp(pivot, 0f, Data.GetTotalTime());
				float ratio = pivot == clamped ? 1f : Mathf.Abs(pivot / clamped);
				Frame reference = Data.GetFrame(clamped);
				trajectory.Points[i].SetPosition(Data.GetFirstFrame().GetRootPosition(mirrored) - ratio * (reference.GetRootPosition(mirrored) - Data.GetFirstFrame().GetRootPosition(mirrored)));
				trajectory.Points[i].SetRotation(reference.GetRootRotation(mirrored));
				trajectory.Points[i].SetVelocity(reference.GetRootVelocity(mirrored));
				trajectory.Points[i].SetSpeed(reference.GetSpeed(mirrored));
				trajectory.Points[i].Styles = styleModule == null ? new float[0] : styleModule.GetStyle(reference);
				trajectory.Points[i].Phase = phaseModule == null ? 0f : Mathf.Repeat(phaseModule.GetPhase(Data.GetFirstFrame(), mirrored) - Utility.PhaseUpdate(phaseModule.GetPhase(Data.GetFirstFrame(), mirrored), phaseModule.GetPhase(reference, mirrored)), 1f);
			} else {
				Frame previous = Data.GetFrame(Mathf.Clamp(frame.Timestamp + delta, 0f, Data.GetTotalTime()));
				trajectory.Points[i].SetTransformation(previous.GetRootTransformation(mirrored));
				trajectory.Points[i].SetVelocity(previous.GetRootVelocity(mirrored));
				trajectory.Points[i].SetSpeed(previous.GetSpeed(mirrored));
				trajectory.Points[i].Styles = styleModule == null ? new float[0] : styleModule.GetStyle(previous);
				trajectory.Points[i].Phase = phaseModule == null ? 0f : phaseModule.GetPhase(previous, mirrored);
			}
		}

		//Future
		for(int i=1; i<=5; i++) {
			float delta = (float)i/5f;
			if(frame.Timestamp + delta > Data.GetTotalTime()) {
				float pivot = 2f*Data.GetTotalTime() - frame.Timestamp - delta;
				float clamped = Mathf.Clamp(pivot, 0f, Data.GetTotalTime());
				float ratio = pivot == clamped ?1f : Mathf.Abs((Data.GetTotalTime() - pivot) / (Data.GetTotalTime() - clamped));
				Frame reference = Data.GetFrame(clamped);
				trajectory.Points[6+i].SetPosition(Data.GetLastFrame().GetRootPosition(mirrored) - ratio * (reference.GetRootPosition(mirrored) - Data.GetLastFrame().GetRootPosition(mirrored)));
				trajectory.Points[6+i].SetRotation(reference.GetRootRotation(mirrored));
				trajectory.Points[6+i].SetVelocity(reference.GetRootVelocity(mirrored));
				trajectory.Points[6+i].SetSpeed(reference.GetSpeed(mirrored));
				trajectory.Points[6+i].Styles = styleModule == null ? new float[0] : styleModule.GetStyle(reference);
				trajectory.Points[6+i].Phase = phaseModule == null ? 0f : Mathf.Repeat(phaseModule.GetPhase(Data.GetLastFrame(), mirrored) + Utility.PhaseUpdate(phaseModule.GetPhase(reference, mirrored), phaseModule.GetPhase(Data.GetLastFrame(), mirrored)), 1f);
			} else {
				Frame future = Data.GetFrame(Mathf.Clamp(frame.Timestamp + delta, 0f, Data.GetTotalTime()));
				trajectory.Points[6+i].SetTransformation(future.GetRootTransformation(mirrored));
				trajectory.Points[6+i].SetVelocity(future.GetRootVelocity(mirrored));
				trajectory.Points[6+i].SetSpeed(future.GetSpeed(mirrored));
				trajectory.Points[6+i].Styles = styleModule == null ? new float[0] : styleModule.GetStyle(future);
				trajectory.Points[6+i].Phase = phaseModule == null ? 0f : phaseModule.GetPhase(future, mirrored);
			}
		}

		//Signals
		for(int i=0; i<7; i++) {
			float delta = -1f + (float)i/6f;
			if(frame.Timestamp + delta < 0f) {
				float pivot = - frame.Timestamp - delta;
				float clamped = Mathf.Clamp(pivot, 0f, Data.GetTotalTime());
				Frame reference = Data.GetFrame(clamped);
				trajectory.Points[i].Signals = styleModule == null ? new float[0] : styleModule.GetInverseSignal(reference);
			} else {
				Frame pivot = Data.GetFrame(Mathf.Clamp(frame.Timestamp + delta, 0f, Data.GetTotalTime()));
				trajectory.Points[i].Signals = styleModule == null ? new float[0] : styleModule.GetSignal(pivot);
			}
		}
		for(int i=7; i<12; i++) {
			trajectory.Points[i].Signals = (float[])trajectory.Points[6].Signals.Clone();
		}

		//Finish
		for(int i=0; i<trajectory.GetLast().Signals.Length; i++) {
			for(int j=7; j<12; j++) {
				if(trajectory.GetLast().Signals[i] > 0f && trajectory.GetLast().Signals[i] < 1f) {
					int pivot = j-1;
					trajectory.Points[j].Styles[i] = Mathf.Max(trajectory.Points[pivot].Styles[i], trajectory.Points[j].Styles[i]);
				}
				if(trajectory.GetLast().Signals[i] < 0f && trajectory.GetLast().Signals[i] > -1f) {
					int pivot = j-1;
					trajectory.Points[j].Styles[i] = Mathf.Min(trajectory.Points[pivot].Styles[i], trajectory.Points[j].Styles[i]);
				}
				if(trajectory.GetLast().Signals[i] == 0f) {
					trajectory.Points[j].Styles[i] = trajectory.Points[6].Styles[i];
				}
				if(trajectory.GetLast().Signals[i] == 1f || trajectory.GetLast().Signals[i] == -1f) {
					trajectory.Points[j].Styles[i] = trajectory.Points[6].Styles[i];
				}
			}
		}
		if(trajectory.GetLast().Signals.AbsSum() == 0f) {
			for(int j=7; j<12; j++) {
				trajectory.Points[j].Styles = (float[])trajectory.Points[6].Styles.Clone();
			}
		}
		
		return trajectory;
	}

	public override void Draw(MotionEditor editor) {
		GetTrajectory(editor.GetCurrentFrame(), editor.Mirror).Draw();
	}

	protected override void DerivedInspector(MotionEditor editor) {
		EditorGUILayout.LabelField("No variables available.");
	}

}
#endif
