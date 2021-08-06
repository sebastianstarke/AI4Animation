using UnityEngine;

public class TimeSeries {
	public enum ID {None, Root, Style, Dribble, Contact, Alignment, Phase};

	public readonly int PastKeys = 0;
	public readonly int FutureKeys = 0;
	public readonly float PastWindow = 0f;
	public readonly float FutureWindow = 0f;
	public readonly int Resolution = 0;

	public readonly Sample[] Samples = new Sample[0];

	public int Pivot {
		get {return PastSampleCount;}
	}
	public int SampleCount {
		get {return PastSampleCount + FutureSampleCount + 1;}
	}
	public int PastSampleCount {
		get {return PastKeys * Resolution;}
	}
	public int FutureSampleCount {
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
		get {return Window / SampleCount;}
	}

	public class Sample {
		public int Index;
		public float Timestamp;
		public Sample(int index, float timestamp) {
			Index = index;
			Timestamp = timestamp;
		}
	}

	//Global Constructor
	public TimeSeries(int pastKeys, int futureKeys, float pastWindow, float futureWindow, int resolution) {
		PastKeys = pastKeys;
		FutureKeys = futureKeys;
		PastWindow = pastWindow;
		FutureWindow = futureWindow;
		Resolution = resolution;
		Samples = new Sample[SampleCount];
		for(int i=0; i<Pivot; i++) {
			Samples[i] = new Sample(i, -PastWindow+i*PastWindow/PastSampleCount);
		}
		Samples[Pivot] = new Sample(Pivot, 0f);
		for(int i=Pivot+1; i<Samples.Length; i++) {
			Samples[i] = new Sample(i, (i-Pivot)*FutureWindow/FutureSampleCount);
		}
	}

	//Derived Constructor
	protected TimeSeries(TimeSeries global) {
		PastKeys = global.PastKeys;
		FutureKeys = global.FutureKeys;
		PastWindow = global.FutureWindow;
		FutureWindow = global.FutureWindow;
		Resolution = global.Resolution;
		Samples = global.Samples;
	}
	
	public float GetTemporalScale(float value) {
		return Window / KeyCount * value;
	}

	public Vector2 GetTemporalScale(Vector2 value) {
		return Window / KeyCount * value;
	}

	public Vector3 GetTemporalScale(Vector3 value) {
		return Window / KeyCount * value;
	}

	public Sample GetPivot() {
		return Samples[Pivot];
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

	public float GetControl(int index, float bias, float min=0f, float max=1f) {
		return index.Ratio(Pivot, Samples.Length-1).ActivateCurve(bias, min, max);
	}

	public float GetCorrection(int index, float bias, float max=1f, float min=0f) {
		return index.Ratio(Pivot, Samples.Length-1).ActivateCurve(bias, max, min);
	}
}