using UnityEngine;

public class TimeSeries {
	
	public abstract class Component : TimeSeries {
		public bool DrawGUI = true;
		public bool DrawScene = true;
		public Component(TimeSeries global) : base(global) {}
		public abstract void Increment(int start, int end);
		public abstract void GUI(UltiDraw.GUIRect rect=null);
		public abstract void Draw(UltiDraw.GUIRect rect=null);
	}

	public int PastKeys {get; private set;}
	public int FutureKeys {get; private set;}
	public float PastWindow {get; private set;}
	public float FutureWindow {get; private set;}
	public int Resolution {get; private set;}
	public Sample[] Samples {get; private set;}

	public int Pivot {
		get {return PastSamples;}
	}
	public int SampleCount {
		get {return PastSamples + FutureSamples + 1;}
	}
	public int PastSamples {
		get {return PastKeys * Resolution;}
	}
	public int FutureSamples {
		get {return FutureKeys * Resolution;}
	}
	public int PivotKey {
		get {return PastKeys;}
	}
	public int KeyCount {
		get {return PastKeys + FutureKeys + 1;}
	}
	public float Window {
		get {return PastWindow + FutureWindow;}
	}
	public float DeltaTime {
		get {return Window / (SampleCount-1);}
	}
	public float MaximumFrequency {
		get {return 0.5f * KeyCount / Window;} //Shannon-Nyquist Sampling Theorem fMax <= 0.5*fSignal
	}

	public class Sample {
		public int Index;
		public float Timestamp;
		public Sample(int index, float timestamp) {
			Index = index;
			Timestamp = timestamp;
		}
	}

	public TimeSeries(int pastKeys, int futureKeys, float pastWindow, float futureWindow, int resolution) {
		PastKeys = pastKeys;
		FutureKeys = futureKeys;
		PastWindow = pastWindow;
		FutureWindow = futureWindow;
		Resolution = resolution;
		Samples = new Sample[SampleCount];
		for(int i=0; i<Pivot; i++) {
			Samples[i] = new Sample(i, -PastWindow+i*PastWindow/PastSamples);
		}
		Samples[Pivot] = new Sample(Pivot, 0f);
		for(int i=Pivot+1; i<Samples.Length; i++) {
			Samples[i] = new Sample(i, (i-Pivot)*FutureWindow/FutureSamples);
		}
	}

	protected TimeSeries(TimeSeries global) {
		SetTimeSeries(global);
	}

	public void SetTimeSeries(TimeSeries global) {
		PastKeys = global.PastKeys;
		FutureKeys = global.FutureKeys;
		PastWindow = global.FutureWindow;
		FutureWindow = global.FutureWindow;
		Resolution = global.Resolution;
		Samples = global.Samples;
	}
	
	public float[] GetTimestamps() {
		float[] timestamps = new float[Samples.Length];
		for(int i=0; i<timestamps.Length; i++) {
			timestamps[i] = Samples[i].Timestamp;
		}
		return timestamps;
	}

	public float GetTemporalScale(float value) {
		// return value;
		return value / KeyCount;
	}

	public Vector2 GetTemporalScale(Vector2 value) {
		// return value;
		return value / KeyCount;
	}

	public Vector3 GetTemporalScale(Vector3 value) {
		// return value;
		return value / KeyCount;
	}

	public Sample GetPivot() {
		return Samples[Pivot];
	}

	public Sample GetKey(float timestamp) {
		if(timestamp < -PastWindow || timestamp > FutureWindow) {
			Debug.Log("Given timestamp was " + timestamp + " but must be within " + PastWindow + " and " + FutureWindow + ".");
			return null;
		}
		return Samples[Mathf.RoundToInt(timestamp.Normalize(-PastWindow, FutureWindow, 0f, SampleCount-1))];
	}

	public Sample GetKey(int index) {
		if(index < 0 || index >= KeyCount) {
			Debug.Log("Given key was " + index + " but must be within 0 and " + (KeyCount-1) + ".");
			return null;
		}
		return Samples[index*Resolution];
	}

	public Sample GetPreviousKey(int sample) {
		if(sample < 0 || sample >= Samples.Length) {
			Debug.Log("Given index was " + sample + " but must be within 0 and " + (Samples.Length-1) + ".");
			return null;
		}
		return GetKey(sample/Resolution);
	}

	public Sample GetNextKey(int sample) {
		if(sample < 0 || sample >= Samples.Length) {
			Debug.Log("Given index was " + sample + " but must be within 0 and " + (Samples.Length-1) + ".");
			return null;
		}
		if(sample % Resolution == 0) {
			return GetKey(sample/Resolution);
		} else {
			return GetKey(sample/Resolution + 1);
		}
	}
}