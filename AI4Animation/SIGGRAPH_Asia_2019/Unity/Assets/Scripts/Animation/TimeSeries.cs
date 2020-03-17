using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TimeSeries {

    public Sample[] Samples = new Sample[0];
	public Series[] Data = new Series[0];
    public int Pivot = 0;
    public int Resolution = 0;
	public float PastWindow = 0f;
	public float FutureWindow = 0f;

	public int PastSampleCount {
		get {return Pivot;}
	}
	public int FutureSampleCount {
		get {return Samples.Length-Pivot-1;}
	}
	public int KeyCount {
		get {return PastKeyCount + FutureKeyCount + 1;}
	}
	public int PivotKey {
		get {return Pivot / Resolution;}
	}
	public int PastKeyCount {
		get {return Pivot / Resolution;}
	}
	public int FutureKeyCount {
		get {return (Samples.Length-Pivot-1) / Resolution;}
	}

    public TimeSeries(int pastKeys, int futureKeys, float pastWindow, float futureWindow, int resolution) {
        int samples = pastKeys + futureKeys + 1;
        if(samples == 1 && resolution != 1) {
            resolution = 1;
            Debug.Log("Resolution corrected to 1 because only one sample is available.");
        }

        Samples = new Sample[(samples-1)*resolution+1];
        Pivot = pastKeys*resolution;
        Resolution = resolution;

        for(int i=0; i<Pivot; i++) {
            Samples[i] = new Sample(i, -pastWindow + i*pastWindow/(pastKeys*resolution));
        }
        Samples[Pivot] = new Sample(Pivot, 0f);
        for(int i=Pivot+1; i<Samples.Length; i++) {
            Samples[i] = new Sample(i, (i-Pivot)*futureWindow/(futureKeys*resolution));
        }

		PastWindow = pastWindow;
		FutureWindow = futureWindow;
    }
	
	private void Add(Series series) {
		ArrayExtensions.Add(ref Data, series);
		if(series.TimeSeries != null) {
			Debug.Log("Data is already added to another time series.");
		} else {
			series.TimeSeries = this;
		}
	}

	public Series GetSeries(string type) {
		for(int i=0; i<Data.Length; i++) {
			if(Data[i].GetID().ToString() == type) {
				return Data[i];
			}
		}
		Debug.Log("Series of type " + type + " could not be found.");
		return null;
		//return System.Array.Find(Data, x => x.GetID().ToString() == type);
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

	public float GetWeight01(int index, float power) {
		return Mathf.Pow((float)index / (float)(Samples.Length-1), power);
	}

	public float GetWeight1byN1(int index, float power) {
		return Mathf.Pow((float)(index+1) / (float)Samples.Length, power);
	}

    public class Sample {
        public int Index;
        public float Timestamp;

        public Sample(int index, float timestamp) {
		    Index = index;
            Timestamp = timestamp;
        }
    }

	public abstract class Series {
		public enum ID {Root, Style, Goal, Contact, Phase, Alignment, Length};
		public TimeSeries TimeSeries;
		public abstract ID GetID();
	}

	public class Root : Series {
		public Matrix4x4[] Transformations;
		public Vector3[] Velocities;
		// public float[] Speeds;

		public override ID GetID() {
			return Series.ID.Root;
		}

		public Root(TimeSeries timeSeries) {
			timeSeries.Add(this);
			Transformations = new Matrix4x4[TimeSeries.Samples.Length];
			Velocities = new Vector3[TimeSeries.Samples.Length];
			// Speeds = new float[TimeSeries.Samples.Length];
			for(int i=0; i<Transformations.Length; i++) {
				Transformations[i] = Matrix4x4.identity;
				Velocities[i] = Vector3.zero;
				// Speeds[i] = 0f;
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

		public float ComputeSpeed() {
			float length = 0f;
			for(int i=TimeSeries.Pivot+TimeSeries.Resolution; i<TimeSeries.Samples.Length; i+=TimeSeries.Resolution) {
				length += Vector3.Distance(GetPosition(i-TimeSeries.Resolution), GetPosition(i));
			}
			return length / TimeSeries.FutureWindow;
		}

		public void Draw() {
			int step = TimeSeries.Resolution;
			UltiDraw.Begin();
			//Connections
			for(int i=0; i<Transformations.Length-step; i+=step) {
				UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i+step].GetPosition(), 0.01f, UltiDraw.Black);
			}

			//Positions
			for(int i=0; i<Transformations.Length; i+=step) {
				UltiDraw.DrawCircle(Transformations[i].GetPosition(), 0.025f, UltiDraw.Black);
			}

			//Directions
			for(int i=0; i<Transformations.Length; i+=step) {
				UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i].GetPosition() + 0.25f*Transformations[i].GetForward(), 0.025f, 0f, UltiDraw.Orange.Transparent(0.75f));
			}

			//Velocities
			for(int i=0; i<Velocities.Length; i+=step) {
				UltiDraw.DrawLine(Transformations[i].GetPosition(), Transformations[i].GetPosition() + Velocities[i], 0.025f, 0f, UltiDraw.DarkGreen.Transparent(0.25f));
			}

			/*
			//Velocity Magnitudes
			List<float[]> functions = new List<float[]>();
			float[] magnitudes = new float[Velocities.Length];
			for(int i=0; i<Velocities.Length; i++) {
				magnitudes[i] = Velocities[i].magnitude;
			}
			functions.Add(magnitudes);
			functions.Add(Speeds);
			UltiDraw.DrawGUIFunctions(new Vector2(0.125f, 0.125f), new Vector2(0.2f, 0.1f), functions, 0f, 5f, 0.0025f, UltiDraw.DarkGrey, new Color[2]{UltiDraw.DarkGreen, UltiDraw.DarkRed}, 0.0025f, UltiDraw.Black);
			*/
			
			UltiDraw.End();
		}
	}


	public class Style : Series {
		public string[] Styles;
		public float[][] Values;

		public override ID GetID() {
			return Series.ID.Style;
		}
		
		public Style(TimeSeries timeSeries, params string[] styles) {
			timeSeries.Add(this);
			Styles = styles;
			Values = new float[TimeSeries.Samples.Length][];
			for(int i=0; i<Values.Length; i++) {
				Values[i] = new float[Styles.Length];
			}
		}

		public void Draw() {
			UltiDraw.Begin();
            List<float[]> functions = new List<float[]>();
            for(int i=0; i<Styles.Length; i++) {
                float[] function = new float[TimeSeries.KeyCount];
                for(int j=0; j<function.Length; j++) {
                    function[j] = Values[TimeSeries.GetKey(j).Index][i];
                }
                functions.Add(function);
            }
            UltiDraw.DrawGUIFunctions(new Vector2(0.875f, 0.685f), new Vector2(0.2f, 0.1f), functions, 0f, 1f, 0.0025f, UltiDraw.DarkGrey, UltiDraw.GetRainbowColors(Styles.Length), 0.0025f, UltiDraw.Black);
            UltiDraw.DrawGUIRectangle(new Vector2(0.875f, 0.685f), new Vector2(0.005f, 0.1f), UltiDraw.White.Transparent(0.5f));
			UltiDraw.End();
		}

		public void GUI() {
			UltiDraw.Begin();
			UltiDraw.DrawGUILabel(0.825f, 0.22f, 0.0175f, "Current Action", UltiDraw.Black);
			Color[] colors = UltiDraw.GetRainbowColors(Styles.Length);
			for(int i=0; i<Styles.Length; i++) {
				float value = Values[TimeSeries.Pivot][i];
				UltiDraw.DrawGUILabel(0.9f, 1f - (0.685f - 0.05f + value*0.1f), value*0.025f, Styles[i], colors[i]);
			}
			UltiDraw.End();
		}

		public void SetStyle(int index, string name, float value) {
			int idx = ArrayExtensions.FindIndex(ref Styles, name);
			if(idx == -1) {
				// Debug.Log("Style " + name + " could not be found.");
				return;
			}
			Values[index][idx] = value;
		}

		public float GetStyle(int index, string name) {
			int idx = ArrayExtensions.FindIndex(ref Styles, name);
			if(idx == -1) {
				Debug.Log("Style " + name + " could not be found.");
				return 0f;
			}
			return Values[index][idx];
		}
	}

	public class Goal : Series {
		public Matrix4x4[] Transformations;
		public string[] Actions;
		public float[][] Values;

		public override ID GetID() {
			return Series.ID.Goal;
		}
		
		public Goal(TimeSeries timeSeries, params string[] actions) {
			timeSeries.Add(this);
			Transformations = new Matrix4x4[TimeSeries.Samples.Length];
			for(int i=0; i<Transformations.Length; i++) {
				Transformations[i] = Matrix4x4.identity;
			}
			Actions = actions;
			Values = new float[TimeSeries.Samples.Length][];
			for(int i=0; i<Values.Length; i++) {
				Values[i] = new float[Actions.Length];
			}
		}

		public float GetAction(int index, string name) {
			int idx = ArrayExtensions.FindIndex(ref Actions, name);
			if(idx == -1) {
				Debug.Log("Action " + name + " could not be found.");
				return 0f;
			}
			return Values[index][idx];
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

		public void Draw() {
			UltiDraw.Begin();

			List<float[]> functions = new List<float[]>();
			for(int i=0; i<Actions.Length; i++) {
				float[] function = new float[TimeSeries.Samples.Length];
				for(int j=0; j<function.Length; j++) {
					function[j] = Values[TimeSeries.Samples[j].Index][i];
				}
				functions.Add(function);
			}
			UltiDraw.DrawGUIFunctions(new Vector2(0.875f, 0.875f), new Vector2(0.2f, 0.1f), functions, 0f, 1f, 0.0025f, UltiDraw.DarkGrey, UltiDraw.GetRainbowColors(Actions.Length), 0.0025f, UltiDraw.Black);
			UltiDraw.DrawGUIRectangle(new Vector2(0.875f, 0.875f), new Vector2(0.005f, 0.1f), UltiDraw.White.Transparent(0.5f));

			for(int i=0; i<TimeSeries.KeyCount; i++) {
				float size = Utility.Normalise((float)i / (float)(TimeSeries.KeyCount-1), 0f, 1f, 0.5f, 1f);
				Matrix4x4 transformation = Transformations[TimeSeries.GetKey(i).Index];	
				UltiDraw.DrawWiredSphere(transformation.GetPosition(), Quaternion.LookRotation(transformation.GetForward(), Vector3.up), size*0.2f, UltiDraw.Magenta.Transparent(0.5f), UltiDraw.Black);
				UltiDraw.DrawTranslateGizmo(transformation.GetPosition(), Quaternion.LookRotation(transformation.GetForward(), Vector3.up), size*0.4f);
			}
			UltiDraw.End();
		}

		public void GUI() {
			UltiDraw.Begin();
			UltiDraw.DrawGUILabel(0.83f, 0.03f, 0.0175f, "Goal Action", UltiDraw.Black);
			Color[] colors = UltiDraw.GetRainbowColors(Actions.Length);
			for(int i=0; i<Actions.Length; i++) {
				float value = Values[TimeSeries.Pivot][i];
				UltiDraw.DrawGUILabel(0.9f, 1f - (0.875f - 0.05f + value*0.1f), value*0.025f, Actions[i], colors[i]);
			}
			UltiDraw.End();
		}
	}

	public class Contact : Series {
		public int[] Indices;
		public string[] Bones;
		public float[][] Values;
		
		public override ID GetID() {
			return Series.ID.Contact;
		}

		public Contact(TimeSeries timeSeries, params string[] bones) {
			timeSeries.Add(this);
			Bones = bones;
			Values = new float[TimeSeries.Samples.Length][];
			for(int i=0; i<Values.Length; i++) {
				Values[i] = new float[Bones.Length];
			}
		}

		public float[] GetContacts(int index) {
			return GetContacts(index, Bones);
		}

		public float[] GetContacts(int index, params string[] bones) {
			float[] values = new float[bones.Length];
			for(int i=0; i<bones.Length; i++) {
				values[i] = GetContact(index, bones[i]);
			}
			return values;
		}

		public float GetContact(int index, string bone) {
			int idx = ArrayExtensions.FindIndex(ref Bones, bone);
			if(idx == -1) {
				Debug.Log("Contact " + bone + " could not be found.");
				return 0f;
			}
			return Values[index][idx];
		}

		public void Draw() {
			UltiDraw.Begin();
			float[] function = new float[TimeSeries.Pivot+1];
			Color[] colors = UltiDraw.GetRainbowColors(Bones.Length);
			for(int i=0; i<Bones.Length; i++) {
				for(int j=0; j<function.Length; j++) {
					function[j] = Values[j][i];
				}
				UltiDraw.DrawGUIBars(new Vector2(0.875f, (i+1)*0.06f), new Vector2(0.2f, 0.05f), function, 0f, 1f, 0.1f/function.Length, UltiDraw.White, colors[i], 0.0025f, UltiDraw.Black);
			}
			UltiDraw.End();
		}

		public void GUI() {
			UltiDraw.Begin();
			UltiDraw.DrawGUILabel(0.84f, 0.625f, 0.0175f, "Contacts", UltiDraw.Black);
			UltiDraw.End();
		}
	}

	public class Phase : Series {
		public float[] Values;
		
		public override ID GetID() {
			return Series.ID.Phase;
		}

		public Phase(TimeSeries timeSeries) {
			timeSeries.Add(this);
			Values = new float[TimeSeries.Samples.Length];
		}

		public void Draw() {
			float[] values = new float[TimeSeries.KeyCount];
			for(int i=0; i<TimeSeries.KeyCount; i++) {
				values[i] = Values[TimeSeries.GetKey(i).Index];
			}
			UltiDraw.Begin();
            UltiDraw.DrawGUIBars(new Vector2(0.875f, 0.510f), new Vector2(0.2f, 0.1f), values, 01f, 1f, 0.01f, UltiDraw.DarkGrey, UltiDraw.White);
			UltiDraw.End();
		}

		public void GUI() {
			UltiDraw.Begin();
			UltiDraw.DrawGUILabel(0.85f, 0.4f, 0.0175f, "Phase", UltiDraw.Black);
			UltiDraw.End();
		}
	}

	public class Alignment : Series {
		public string[] Bones;
		public float[][] Magnitudes;
		public float[][] Phases;

		public float[][] _PhaseMagnitudes;
		public float[][] _PhaseUpdateValues;
		public Vector2[][] _PhaseUpdateVectors;
		public Vector2[][] _PhaseStates;

		public override ID GetID() {
			return Series.ID.Alignment;
		}

		public Alignment(TimeSeries timeSeries, params string[] bones) {
			timeSeries.Add(this);
			Bones = bones;
			Magnitudes = new float[timeSeries.Samples.Length][];
			Phases = new float[timeSeries.Samples.Length][];
			for(int i=0; i<timeSeries.Samples.Length; i++) {
				Magnitudes[i] = new float[bones.Length];
				Phases[i] = new float[bones.Length];
			}

			_PhaseMagnitudes = new float[timeSeries.Samples.Length][];
			_PhaseUpdateValues = new float[timeSeries.Samples.Length][];
			_PhaseUpdateVectors = new Vector2[timeSeries.Samples.Length][];
			_PhaseStates = new Vector2[timeSeries.Samples.Length][];
			for(int i=0; i<timeSeries.Samples.Length; i++) {
				_PhaseMagnitudes[i] = new float[bones.Length];
				_PhaseUpdateValues[i] = new float[bones.Length];
				_PhaseUpdateVectors[i] = new Vector2[bones.Length];
				_PhaseStates[i] = new Vector2[bones.Length];
			}
		}

		public float GetAveragePhase(int index) {
			float[] values = new float[Bones.Length];
			for(int i=0; i<Bones.Length; i++) {
				values[i] = Phases[index][i];
			}
			return Utility.FilterPhaseLinear(values);
		}

		public void Draw() {
			UltiDraw.Begin();

			//DEBUG RENDERING
			// {
			// 	float _xMin = 0.025f;
			// 	float _xMax = 0.25f;
			// 	float _yMin = 0.05f;
			// 	float _yMax = 0.25f;
			// 	for(int b=0; b<Bones.Length; b++) {
			// 		float w = (float)b/(float)(Bones.Length-1);
			// 		List<float[]> values = new List<float[]>();
			// 		float max = 0f;
			// 		float[] first = new float[TimeSeries.FutureKeyCount+1];
			// 		float[] second = new float[TimeSeries.FutureKeyCount+1];
			// 		float[] third = new float[TimeSeries.FutureKeyCount+1];
			// 		for(int i=0; i<TimeSeries.FutureKeyCount+1; i++) {
			// 			first[i] = _PhaseMagnitudes[TimeSeries.GetKey(i+TimeSeries.PivotKey).Index][b];
			// 			second[i] = _PhaseStates[TimeSeries.GetKey(i+TimeSeries.PivotKey).Index][b].magnitude;
			// 			third[i] = _PhaseUpdateVectors[TimeSeries.GetKey(i+TimeSeries.PivotKey).Index][b].magnitude;
			// 			max = Mathf.Max(max, first[i], second[i], third[i]);
			// 		}
			// 		values.Add(first);
			// 		values.Add(second);
			// 		values.Add(third);
			// 		float vertical = Utility.Normalise(w, 0f, 1f, _yMin, _yMax);
			// 		float height = 0.95f*(_yMax-_yMin)/(Bones.Length-1);
			// 		UltiDraw.DrawGUIFunctions(new Vector2(0.5f * (_xMin + _xMax), vertical), new Vector2(_xMax-_xMin, height), values, 0f, max, UltiDraw.DarkGrey, new Color[3]{Color.yellow, Color.magenta, Color.cyan});
			// 	}
			// }
			// {
			// 	float _xMin = 0.025f;
			// 	float _xMax = 0.25f;
			// 	float _yMin = 0.3f;
			// 	float _yMax = 0.5f;
			// 	for(int b=0; b<Bones.Length; b++) {
			// 		float w = (float)b/(float)(Bones.Length-1);
			// 		List<float[]> values = new List<float[]>();
			// 		float[] first = new float[TimeSeries.FutureKeyCount+1];
			// 		float[] second = new float[TimeSeries.FutureKeyCount+1];
			// 		for(int i=0; i<TimeSeries.FutureKeyCount+1; i++) {
			// 			first[i] = _PhaseUpdateValues[TimeSeries.GetKey(i+TimeSeries.PivotKey).Index][b];
			// 			second[i] = Utility.PhaseUpdate(0f, Utility.PhaseValue(_PhaseUpdateVectors[TimeSeries.GetKey(i+TimeSeries.PivotKey).Index][b]));
			// 		}
			// 		values.Add(first);
			// 		values.Add(second);
			// 		float vertical = Utility.Normalise(w, 0f, 1f, _yMin, _yMax);
			// 		float height = 0.95f*(_yMax-_yMin)/(Bones.Length-1);
			// 		UltiDraw.DrawGUIFunctions(new Vector2(0.5f * (_xMin + _xMax), vertical), new Vector2(_xMax-_xMin, height), values, 0f, 0.1f, UltiDraw.DarkGrey, new Color[2]{Color.yellow, Color.magenta});
			// 	}
			// }
			//

			float xMin = 0.3f;
			float xMax = 0.7f;
			float yMin = 0.75f;
			float yMax = 0.95f;

			//This is phase vector rendering
			for(int b=0; b<Bones.Length; b++) {
				float w = (float)b/(float)(Bones.Length-1);
				float[] values = new float[TimeSeries.KeyCount];
				Color[] colors = new Color[TimeSeries.KeyCount];
				for(int i=0; i<TimeSeries.KeyCount; i++) {
					values[i] = Phases[TimeSeries.GetKey(i).Index][b];
					colors[i] = UltiDraw.White.Transparent(Magnitudes[TimeSeries.GetKey(i).Index][b]);
				}
				float vertical = Utility.Normalise(w, 0f, 1f, yMin, yMax);
				float height = 0.95f*(yMax-yMin)/(Bones.Length-1);
				UltiDraw.DrawGUIBars(new Vector2(0.5f * (xMin + xMax), vertical), new Vector2(xMax-xMin, height), values, 1f, 1f, 0.01f, UltiDraw.DarkGrey, Color.white);
				// UltiDraw.DrawGUICircularPivot(new Vector2(xMax + height/2f, vertical), height/2f, UltiDraw.DarkGrey, 360f*Phases[TimeSeries.Pivot][b], 1f, colors[TimeSeries.PivotKey]);
				UltiDraw.DrawGUICircularPivot(new Vector2(xMax + height/2f, vertical), height/2f, UltiDraw.DarkGrey, 360f*Phases[TimeSeries.Pivot][b], Magnitudes[TimeSeries.Pivot][b], colors[TimeSeries.PivotKey]);

				for(int i=0; i<TimeSeries.KeyCount; i++) {
					float horizontal = Utility.Normalise((float)i / (float)(TimeSeries.KeyCount-1), 0f, 1f, xMin, xMax);
					// UltiDraw.DrawGUICircularPivot(new Vector2(horizontal, vertical + height/4f), height/4f, UltiDraw.DarkGrey, 360f*Phases[TimeSeries.GetKey(i).Index][b], 1f, colors[i]);
					UltiDraw.DrawGUICircularPivot(new Vector2(horizontal, vertical + height/4f), height/4f, UltiDraw.DarkGrey, 360f*Utility.PhaseUpdate(Phases[TimeSeries.Pivot][b], Phases[TimeSeries.GetKey(i).Index][b]), 1f, colors[i]);

					// UltiDraw.DrawGUICircularPivot(new Vector2(horizontal, vertical + height/4f), height/4f, UltiDraw.DarkGrey, 360f*Values[TimeSeries.GetKey(i).Index][b], 1f, UltiDraw.Cyan);
					// Vector2 phase = Utility.PhaseVectorUpdate(Vectors[TimeSeries.Pivot][b], Vectors[TimeSeries.GetKey(i).Index][b]);
					// UltiDraw.DrawGUICircularPivot(new Vector2(horizontal, vertical + height/4f), height/4f, UltiDraw.DarkGrey, -Vector2.SignedAngle(Vector2.up, phase), 1f, colors[i]);
				}
			}

			// //This is phase rendering
			// for(int b=0; b<Bones.Length; b++) {
			// 	float w = (float)b/(float)(Bones.Length-1);
			// 	float[] values = new float[TimeSeries.KeyCount];
			// 	Color[] colors = new Color[TimeSeries.KeyCount];
			// 	for(int i=0; i<TimeSeries.KeyCount; i++) {
			// 		values[i] = Values[TimeSeries.GetKey(i).Index][b];
			// 		colors[i] = UltiDraw.White.Transparent(Mathf.Clamp(Magnitudes[TimeSeries.GetKey(i).Index][b], 0f, 1f));
			// 	}
			// 	float vertical = Utility.Normalise(w, 0f, 1f, yMin, yMax);
			// 	float height = 0.95f*(yMax-yMin)/(Bones.Length-1);
			// 	UltiDraw.DrawGUIBars(new Vector2(0.5f * (xMin + xMax), vertical), new Vector2(xMax-xMin, height), values, 01f, 1f, 0.01f, UltiDraw.DarkGrey, colors);
			// 	UltiDraw.DrawGUICircularPivot(new Vector2(xMax + height/2f, vertical), height/2f, UltiDraw.DarkGrey, 360f*Values[TimeSeries.Pivot][b], 1f, UltiDraw.Cyan);
			// }

			//This is energy rendering
			// for(int b=0; b<Bones.Length; b++) {
			// 	float w = (float)b/(float)(Bones.Length-1);
			// 	float[] values = new float[TimeSeries.KeyCount];
			// 	Color[] colors = new Color[TimeSeries.KeyCount];
			// 	for(int i=0; i<TimeSeries.KeyCount; i++) {
			// 		values[i] = 1f;
			// 		float weight = Vector2.Angle(Vector2.up, Utility.PhaseVector(Values[TimeSeries.GetKey(i).Index][b])) / 180f;
			// 		colors[i] = Color.Lerp(UltiDraw.Cyan, UltiDraw.Orange, weight).Transparent(Mathf.Clamp(Magnitudes[TimeSeries.GetKey(i).Index][b], 0f, 1f));
			// 	}
			// 	float vertical = Utility.Normalise(w, 0f, 1f, yMin, yMax);
			// 	float height = 0.95f*(yMax-yMin)/(Bones.Length-1);
			// 	UltiDraw.DrawGUIBars(new Vector2(0.5f * (xMin + xMax), vertical), new Vector2(xMax-xMin, height), values, 01f, 1f, 0.01f, UltiDraw.DarkGrey, colors);
			// 	UltiDraw.DrawGUICircle(new Vector2(xMax + height/2f, vertical), height/2f, colors[TimeSeries.PivotKey]);
			// }

			{
				float max = 0f;
				List<float[]> values = new List<float[]>();
				for(int b=0; b<Bones.Length; b++) {
					float[] v = new float[TimeSeries.KeyCount];
					for(int i=0; i<TimeSeries.KeyCount; i++) {
						v[i] = Magnitudes[TimeSeries.GetKey(i).Index][b];
						max = Mathf.Max(max, v[i]);
					}
					values.Add(v);
				}
				float vertical = yMin - 1f*(yMax-yMin)/(Bones.Length-1);
				float height = 0.95f*(yMax-yMin)/(Bones.Length-1);
				UltiDraw.DrawGUIFunctions(new Vector2(0.5f * (xMin + xMax), vertical), new Vector2(xMax-xMin, height), values, 0f, Mathf.Max(1f, max), UltiDraw.DarkGrey, UltiDraw.GetRainbowColors(values.Count));
			}

			//This is vector rendering
			// for(int b=0; b<Bones.Length; b++) {
			// 	float w = (float)b/(float)(Bones.Length-1);
			// 	float[] values = new float[TimeSeries.KeyCount];
			// 	Color[] colors = new Color[TimeSeries.KeyCount];
			// 	for(int i=0; i<TimeSeries.KeyCount; i++) {
			// 		values[i] = Utility.PhaseValue(Vectors[TimeSeries.GetKey(i).Index][b]);
			// 		colors[i] = UltiDraw.White.Transparent(Mathf.Clamp(Vectors[TimeSeries.GetKey(i).Index][b].magnitude, 0f, 1f));
			// 	}
			// 	float vertical = Utility.Normalise(w, 0f, 1f, yMin, yMax);
			// 	float height = 0.95f*(yMax-yMin)/(Bones.Length-1);
			// 	UltiDraw.DrawGUIBars(new Vector2(0.5f * (xMin + xMax), vertical), new Vector2(xMax-xMin, height), values, 01f, 1f, 0.01f, UltiDraw.DarkGrey, colors);
			// 	UltiDraw.DrawGUICircularPivot(new Vector2(xMax + height/2f, vertical), height/2f, UltiDraw.DarkGrey, 360f*Utility.PhaseValue(Vectors[TimeSeries.Pivot][b]), 1f, UltiDraw.Cyan);
			// }
			// {
			// 	float max = 0f;
			// 	List<float[]> values = new List<float[]>();
			// 	for(int b=0; b<Bones.Length; b++) {
			// 		float[] v = new float[TimeSeries.KeyCount];
			// 		for(int i=0; i<TimeSeries.KeyCount; i++) {
			// 			v[i] = Vectors[TimeSeries.GetKey(i).Index][b].magnitude;
			// 			max = Mathf.Max(max, v[i]);
			// 		}
			// 		values.Add(v);
			// 	}
			// 	float vertical = yMin - 1*(yMax-yMin)/(Bones.Length-1);
			// 	float height = 0.95f*(yMax-yMin)/(Bones.Length-1);
			// 	UltiDraw.DrawGUIFunctions(new Vector2(0.5f * (xMin + xMax), vertical), new Vector2(xMax-xMin, height), values, 0f, Mathf.Max(2f, max), UltiDraw.DarkGrey, UltiDraw.GetRainbowColors(values.Count));
			// }
			// {
			// 	List<float[]> values = new List<float[]>();
			// 	for(int b=0; b<Bones.Length; b++) {
			// 		float[] v = new float[TimeSeries.KeyCount];
			// 		for(int i=0; i<TimeSeries.KeyCount; i++) {
			// 			v[i] = Frequencies[TimeSeries.GetKey(i).Index][b];
			// 		}
			// 		values.Add(v);
			// 	}
			// 	float vertical = yMin - 2*(yMax-yMin)/(Bones.Length-1);
			// 	float height = 0.95f*(yMax-yMin)/(Bones.Length-1);
			// 	UltiDraw.DrawGUIFunctions(new Vector2(0.5f * (xMin + xMax), vertical), new Vector2(xMax-xMin, height), values, 0f, Mathf.Max(0.5f, 2.5f), UltiDraw.DarkGrey, UltiDraw.GetRainbowColors(values.Count));
			// }

			// {
			// 	float[] combination = new float[TimeSeries.KeyCount];
			// 	for(int i=0; i<TimeSeries.KeyCount; i++) {
			// 		combination[i] = GetAveragePhase(TimeSeries.GetKey(i).Index);
			// 	}
			// 	float vertical = yMin - 2f*(yMax-yMin)/(Bones.Length-1);
			// 	float height = 0.95f*(yMax-yMin)/(Bones.Length-1);
			// 	UltiDraw.DrawGUIBars(new Vector2(0.5f * (xMin + xMax), vertical), new Vector2(xMax-xMin, height), combination, 01f, 1f, 0.01f, UltiDraw.DarkGrey, UltiDraw.Red);
			// 	UltiDraw.DrawGUICircularPivot(new Vector2(xMax + height/2f, vertical), height/2f, UltiDraw.DarkGrey, 360f*combination[TimeSeries.PivotKey], 1f, UltiDraw.DarkRed);
			// }

			/*
			for(int i=0; i<Values.Length; i++) {
				float min = 0.1f;
				float max = 0.9f;
				float size = 1f * (max-min) / Values.Length;
				float w = (float)i/(float)(Values.Length-1);
				Vector2 center = new Vector2(Utility.Normalise(w, 0f, 1f, min, max), 0.1f);
				UltiDraw.DrawGUICircularPivot(center, size, UltiDraw.DarkGrey, Values[i] * 360f, 1f, UltiDraw.Cyan);
			}

			for(int i=0; i<Values.Length; i++) {
				float min = 0.1f;
				float max = 0.9f;
				float size = 1f * (max-min) / Values.Length;
				float w = (float)i/(float)(Values.Length-1);
				Vector2 center = new Vector2(Utility.Normalise(w, 0f, 1f, min, max), 0.1f);
				//UltiDraw.DrawGUICircularPivot(center, size, UltiDraw.DarkGrey, -Vector2.SignedAngle(Vector2.up, Vectors[i]), 1f, UltiDraw.Cyan);
			}
			*/

            //UltiDraw.DrawGUIBars(new Vector2(0.875f, 0.510f), new Vector2(0.2f, 0.1f), Values, 01f, 1f, 0.01f, UltiDraw.DarkGrey, UltiDraw.White);
			
			UltiDraw.End();
		}
	}


}