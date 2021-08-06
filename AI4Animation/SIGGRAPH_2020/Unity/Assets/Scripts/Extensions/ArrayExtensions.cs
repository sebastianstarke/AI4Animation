using System;
using System.Collections.Generic;
using UnityEngine;

public static class ArrayExtensions {

	public static bool HasType<T>(this object[] objects) {
		foreach(object o in objects) {
			if(o is T) {
				return true;
			}
		}
		return false;
	}

	public static T FindType<T>(this object[] objects) {
		foreach(object o in objects) {
			if(o is T) {
				return (T)o;
			}
		}
		Debug.Log("Object of type " + typeof(T).Name + " could not be found.");
		return default(T);
	}

	public static bool Verify<T>(this T[] array, int length) {
		return array != null && array.Length == length;
	}

	public static T[] Validate<T>(this T[] array, int length) {
		if(array == null || array.Length != length) {
			array = new T[length];
		}
		return array;
	}

	public static int[] CreateRandom(int length, int min, int max) {
		int[] values = new int[length];
        for(int i=0; i<values.Length; i++) {
            values[i] = UnityEngine.Random.Range(min, max+1);
        }
        return values;
	}

	public static float[] CreateRandom(int length, float min, float max) {
		float[] values = new float[length];
        for(int i=0; i<values.Length; i++) {
            values[i] = UnityEngine.Random.Range(min, max);
        }
        return values;
	}

	public static T[] GatherByLength<T>(this T[] array, int start, int length) {
		T[] result = new T[length];
		for(int i=0; i<length; i++) {
			result[i] = array[start+i];
		}
		return result;
	}

	public static T[] GatherByPivots<T>(this T[] array, int start, int end) {
		return array.GatherByLength(start, end-start+1);
	}

	public static T[] GatherByIndices<T>(this T[] array, params int[] indices) {
		T[] result = new T[indices.Length];
		for(int i=0; i<indices.Length; i++) {
			result[i] = array[indices[i]];
		}
		return result;
	}

	public static T[] GatherByWindow<T>(this T[] array, int index, int padding) {
		int start = Mathf.Max(index-padding, 0);
		int end = Mathf.Min(index+padding, array.Length-1);
		T[] result = new T[end-start+1];
		for(int i=0; i<result.Length; i++) {
			result[i] = array[start+i];
		}
		return result;
	}

	public static T[] GatherByOverflowWindow<T>(this T[] array, int index, int padding) {
		int start = Mathf.Max(index-padding, 0);
		int end = Mathf.Min(index+padding, array.Length-1);
		int endOverflow = (index+padding) - end;
		int startOverflow = start - (index-padding);
		start = Mathf.Max(start-endOverflow, 0);
		end = Mathf.Min(end+startOverflow, array.Length-1);
		T[] result = new T[end-start+1];
		for(int i=0; i<result.Length; i++) {
			result[i] = array[start+i];
		}
		return result;
	}

	public static bool Same<T>(T[] a, T[] b) {
		if(a.Length != b.Length) {
			return false;
		}
		for(int i=0; i<a.Length; i++) {
			if(!a[i].Equals(b[i])) {
				return false;
			}
		}
		return true;
	}

	public static T Append<T>(ref T[] array, T element) {
		Expand(ref array);
		array[array.Length-1] = element;
		return element;
	}

	public static void Insert<T>(ref T[] array, T element, int index) {
		if(index >= 0 && index < array.Length) {
			Expand(ref array);
			for(int i=array.Length-1; i>index; i--) {
				array[i] = array[i-1];
			}
			array[index] = element;
		}
	}

	public static bool RemoveAt<T>(ref T[] array, int index) {
		if(index >= 0 && index < array.Length) {
			for(int i=index; i<array.Length-1; i++) {
				array[i] = array[i+1];
			}
			Shrink(ref array);
			return true;
		} else {
			return false;
		}
	}

	public static bool Remove<T>(ref T[] array, T element) {
		return RemoveAt(ref array, FindIndex(ref array, element));
	}

	public static void Expand<T>(ref T[] array) {
		Array.Resize(ref array, array.Length+1);
	}

	public static void Shrink<T>(ref T[] array) {
		if(array.Length > 0) {
			Array.Resize(ref array, array.Length-1);
		}
	}

	public static void Resize<T>(ref T[] array, int size) {
		Array.Resize(ref array, size);
	}

	public static void Clear<T>(ref T[] array) {
		Array.Resize(ref array, 0);
	}

	public static T First<T>(this T[] array) {
		return array[0];
	}

	public static T[] First<T>(this T[,] array) {
		T[] result = new T[array.GetLength(1)];
		for(int i=0; i<result.Length; i++) {
			result[i] = array[0, i];
		}
		return result;
	}

	public static T Last<T>(this T[] array) {
		return array[array.Length-1];
	}

	public static T[] Last<T>(this T[,] array) {
		T[] result = new T[array.GetLength(1)];
		for(int i=0; i<result.Length; i++) {
			result[i] = array[array.GetLength(0)-1, i];
		}
		return result;
	}

	public static int FindIndex<T>(ref T[] array, T element) {
		return Array.FindIndex(array, x => x != null && x.Equals(element));
	}

	public static int FindIndex<T>(this T[] array, T element) {
		return Array.FindIndex(array, x => x != null && x.Equals(element));
	}

	public static T Find<T>(ref T[] array, T element) {
		return Array.Find(array, x => x != null && x.Equals(element));
	}

	public static T Find<T>(this T[] array, T element) {
		return Array.Find(array, x => x != null && x.Equals(element));
	}

	public static bool Contains<T>(this T[] array, T element) {
		return FindIndex(ref array, element) >= 0;
	}

    public static bool Contains<T>(this T[] haystack, params T[] needles) {
        foreach (T needle in needles) {
            if (haystack.Contains(needle)) {
                return true;
            }
        }
        return false;
    }

	public static T[] Concat<T>(T[] lhs, T[] rhs) {
		T[] result = new T[lhs.Length + rhs.Length];
		lhs.CopyTo(result, 0);
		rhs.CopyTo(result, lhs.Length);
		return result;
	}

	public static T[] Concat<T>(T lhs, T[] rhs) {
		return Concat(new T[1]{lhs}, rhs);
	}

	public static T[] Concat<T>(T[] lhs, T rhs) {
		return Concat(lhs, new T[1]{rhs});
	}

	public static float[] Add(float[] lhs, float[] rhs) {
		if(lhs.Length != rhs.Length) {
			Debug.Log("Incompatible array dimensions.");
			return null;
		}
		float[] result = new float[lhs.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = lhs[i] + rhs[i];
		}
		return result;
	}

	public static float[] Add(float[] lhs, float value) {
		float[] result = new float[lhs.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = lhs[i] + value;
		}
		return result;
	}

	public static double[] Add(double[] lhs, double[] rhs) {
		if(lhs.Length != rhs.Length) {
			Debug.Log("Incompatible array dimensions.");
			return null;
		}
		double[] result = new double[lhs.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = lhs[i] + rhs[i];
		}
		return result;
	}

	public static double[] Add(double[] lhs, double value) {
		double[] result = new double[lhs.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = lhs[i] + value;
		}
		return result;
	}

	public static float[] Sub(float[] lhs, float[] rhs) {
		if(lhs.Length != rhs.Length) {
			Debug.Log("Incompatible array dimensions.");
			return null;
		}
		float[] result = new float[lhs.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = lhs[i] - rhs[i];
		}
		return result;
	}

	public static float[] Sub(float[] lhs, float value) {
		float[] result = new float[lhs.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = lhs[i] - value;
		}
		return result;
	}

	public static double[] Sub(double[] lhs, double[] rhs) {
		if(lhs.Length != rhs.Length) {
			Debug.Log("Incompatible array dimensions.");
			return null;
		}
		double[] result = new double[lhs.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = lhs[i] - rhs[i];
		}
		return result;
	}

	public static double[] Sub(double[] lhs, double value) {
		double[] result = new double[lhs.Length];
		for(int i=0; i<result.Length; i++) {
			result[i] = lhs[i] - value;
		}
		return result;
	}

	public static int Sum(this int[] values) {
		int sum = 0;
		for(int i=0; i<values.Length; i++) {
			sum += values[i];
		}
		return sum;
	}

	public static float Sum(this float[] values) {
		float sum = 0f;
		for(int i=0; i<values.Length; i++) {
			sum += values[i];
		}
		return sum;
	}

	public static double Sum(this double[] values) {
		double sum = 0.0;
		for(int i=0; i<values.Length; i++) {
			sum += values[i];
		}
		return sum;
	}

	public static int AbsSum(this int[] values) {
		int sum = 0;
		for(int i=0; i<values.Length; i++) {
			sum += System.Math.Abs(values[i]);
		}
		return sum;
	}

	public static float AbsSum(this float[] values) {
		float sum = 0f;
		for(int i=0; i<values.Length; i++) {
			sum += System.Math.Abs(values[i]);
		}
		return sum;
	}

	public static double AbsSum(this double[] values) {
		double sum = 0.0;
		for(int i=0; i<values.Length; i++) {
			sum += System.Math.Abs(values[i]);
		}
		return sum;
	}

	public static int Min(this int[] values) {
		if(values.Length == 0) {
			return 0;
		}
		int min = int.MaxValue;
		for(int i=0; i<values.Length; i++) {
			min = Mathf.Min(min, values[i]);
		}
		return min;
	}

	public static float Min(this float[] values) {
		if(values.Length == 0) {
			return 0f;
		}
		float min = float.MaxValue;
		for(int i=0; i<values.Length; i++) {
			min = Mathf.Min(min, values[i]);
		}
		return min;
	}

	public static double Min(this double[] values) {
		if(values.Length == 0) {
			return 0.0;
		}
		double min = double.MaxValue;
		for(int i=0; i<values.Length; i++) {
			min = System.Math.Min(min, values[i]);
		}
		return min;
	}

	public static int Max(this int[] values) {
		if(values.Length == 0) {
			return 0;
		}
		int max = int.MinValue;
		for(int i=0; i<values.Length; i++) {
			max = Mathf.Max(max, values[i]);
		}
		return max;
	}

	public static float Max(this float[] values) {
		if(values.Length == 0) {
			return 0f;
		}
		float max = float.MinValue;
		for(int i=0; i<values.Length; i++) {
			max = Mathf.Max(max, values[i]);
		}
		return max;
	}

	public static double Max(this double[] values) {
		if(values.Length == 0) {
			return 0.0;
		}
		double max = double.MinValue;
		for(int i=0; i<values.Length; i++) {
			max = System.Math.Max(max, values[i]);
		}
		return max;
	}

	public static int Amp(this int[] values) {
		if(values.Length == 0) {
			return 0;
		}
		int max = 0;
		for(int i=0; i<values.Length; i++) {
			max = Mathf.Max(max, Mathf.Abs(values[i]));
		}
		return max;
	}

	public static float Amp(this float[] values) {
		if(values.Length == 0) {
			return 0;
		}
		float max = 0f;
		for(int i=0; i<values.Length; i++) {
			max = Mathf.Max(max, Mathf.Abs(values[i]));
		}
		return max;
	}

	public static double Amp(this double[] values) {
		if(values.Length == 0) {
			return 0;
		}
		double max = 0.0;
		for(int i=0; i<values.Length; i++) {
			max = System.Math.Max(max, System.Math.Abs(values[i]));
		}
		return max;
	}

	public static float Gaussian(this int[] values, float power=1f) {
		if(values.Length == 0) {
			return 0f;
		}
		if(values.Length == 1) {
			return values[0];
		}
		float padding = ((float)values.Length - 1f) / 2f;
		float sum = 0f;
		float value = 0f;
		for(int i=0; i<values.Length; i++) {
			float weight = Mathf.Exp(-Mathf.Pow((float)i - padding, 2f) / Mathf.Pow(0.5f * padding, 2f));
			if(power != 1f) {
				weight = Mathf.Pow(weight, power);
			}
			value += weight * (float)values[i];
			sum += weight;
		}
		return value / sum;
	}

	public static float Gaussian(this float[] values, float power=1f) {
		if(values.Length == 0) {
			return 0f;
		}
		if(values.Length == 1) {
			return values[0];
		}
		float padding = ((float)values.Length - 1f) / 2f;
		float sum = 0f;
		float value = 0f;
		for(int i=0; i<values.Length; i++) {
			float weight = Mathf.Exp(-Mathf.Pow((float)i - padding, 2f) / Mathf.Pow(0.5f * padding, 2f));
			if(power != 1f) {
				weight = Mathf.Pow(weight, power);
			}
			value += weight * values[i];
			sum += weight;
		}
		return value / sum;
	}

	public static double Gaussian(this double[] values, double power=1.0) {
		if(values.Length == 0) {
			return 0.0;
		}
		if(values.Length == 1) {
			return values[0];
		}
		double padding = ((double)values.Length - 1.0) / 2.0;
		double sum = 0.0;
		double value = 0.0;
		for(int i=0; i<values.Length; i++) {
			double weight = System.Math.Exp(-System.Math.Pow((double)i - padding, 2.0) / System.Math.Pow(0.5 * padding, 2.0));
			if(power != 1.0) {
				weight = System.Math.Pow(weight, power);
			}
			value += weight * values[i];
			sum += weight;
		}
		return value / sum;
	}

	public static float Gaussian(this int[] values, bool[] mask, float power=1f) {
		if(values.Length == 0) {
			return 0f;
		}
		if(values.Length == 1) {
			return values[0];
		}
		float padding = ((float)values.Length - 1f) / 2f;
		float sum = 0f;
		float value = 0f;
		for(int i=0; i<values.Length; i++) {
			float weight = Mathf.Exp(-Mathf.Pow((float)i - padding, 2f) / Mathf.Pow(0.5f * padding, 2f));
			if(mask[i]) {
				if(power != 1f) {
					weight = Mathf.Pow(weight, power);
				}
				value += weight * (float)values[i];
			}
			sum += weight;
		}
		return value / sum;
	}

	public static float Mean(this int[] values) {
		if(values.Length == 0) {
			return 0;
		}
		float mean = 0f;
		float args = 0f;
		for(int i=0; i<values.Length; i++) {
			mean += (float)values[i];
			args += 1f;
		}
		mean /= args;
		return mean;
	}

	public static float Mean(this float[] values) {
		if(values.Length == 0) {
			return 0f;
		}
		float mean = 0f;
		float args = 0f;
		for(int i=0; i<values.Length; i++) {
			mean += values[i];
			args += 1f;
		}
		mean /= args;
		return mean;
	}

	public static double Mean(this double[] values) {
		if(values.Length == 0) {
			return 0.0;
		}
		double mean = 0.0;
		double args = 0.0;
		for(int i=0; i<values.Length; i++) {
			mean += values[i];
			args += 1.0;
		}
		mean /= args;
		return mean;
	}

	public static float Variance(this int[] values) {
		if(values.Length == 0) {
			return 0f;
		}
		float variance = 0f;
		float mean = values.Mean();
		float args = 0f;
		for(int i=0; i<values.Length; i++) {
			variance += Mathf.Pow(values[i] - mean, 2f);
			args += 1f;
		}
		variance /= args;
		return variance;
	}

	public static float Variance(this float[] values) {
		if(values.Length == 0) {
			return 0f;
		}
		float variance = 0f;
		float mean = values.Mean();
		float args = 0f;
		for(int i=0; i<values.Length; i++) {
			variance += Mathf.Pow(values[i] - mean, 2f);
			args += 1f;
		}
		variance /= args;
		return variance;
	}

	public static double Variance(this double[] values) {
		if(values.Length == 0) {
			return 0.0;
		}
		double variance = 0.0;
		double mean = values.Mean();
		double args = 1.0;
		for(int i=0; i<values.Length; i++) {
			variance += System.Math.Pow(values[i] - mean, 2.0);
			args += 1.0;
		}
		variance /= args;
		return variance;
	}

	public static float Sigma(this int[] values) {
		return Mathf.Sqrt(Variance(values));
	}

	public static float Sigma(this float[] values) {
		return Mathf.Sqrt(Variance(values));
	}

	public static double Sigma(this double[] values) {
		return System.Math.Sqrt(Variance(values));
	}

	public static void Zero(this float[] values) {
		for(int i=0; i<values.Length; i++) {
			values[i] = 0f;
		}
	}

	public static bool Repeating<T>(this T[] values, float percentage) {
		int count = 0;
		for(int i=1; i<values.Length; i++) {
			if(values[i-1].Equals(values[i])) {
				count += 1;
				float ratio = 1f - (float)count / (float)(values.Length-1);
				if(ratio < percentage) {
					return true;
				}
			}
		}
		return false;
	}

	public static bool All<T>(this T[] values, T value, float percentage) {
		int count = 0;
		foreach(T v in values) {
			if(!v.Equals(value)) {
				count += 1;
				float ratio = 1f - (float)count / (float)values.Length;
				if(ratio < percentage) {
					return false;
				}
			}
		}
		return true;
	}

	public static bool All<T>(this T[] values, T value) {
		foreach(T v in values) {
			if(!v.Equals(value)) {
				return false;
			}
		}
		return true;
	}

	public static bool Any<T>(this T[] values, T value) {
		foreach(T v in values) {
			if(v.Equals(value)) {
				return true;
			}
		}
		return false;
	}
	
	public static bool[] Equal<T>(T[] a, T[] b) {
		if(a.Length != b.Length) {
			return new bool[0];
		}
		bool[] c = new bool[a.Length];
		for(int i=0; i<c.Length; i++) {
			c[i] = a[i].Equals(b[i]);
		}
		return c;
	}

	public static int Count<T>(this T[] values, T value) {
		int count = 0;
		for(int i=0; i<values.Length; i++) {
			if(values[i].Equals(value)) {
				count += 1;
			}
		}
		return count;
	}

	public static float Ratio<T>(this T[] values, T value) {
		return (float)Count(values, value) / (float)values.Length;
	}

	public static float AverageSequenceLength<T>(this T[] values, T value) {
		List<int> sequences = new List<int>();
		for(int i=0; i<values.Length; i++) {
			if(values[i].Equals(value)) {
				int count = 0;
				while(i<values.Length && values[i].Equals(value)) {
					count +=1 ;
					i += 1;
				}
				sequences.Add(count);
			}
		}
		return sequences.ToArray().Mean();
	}

	public static void SetAll<T>(this T[] array, T value) {
		for(int i=0; i<array.Length; i++) {
			array[i] = value;
		}
	}

	public static T[] Flatten<T>(this T[][] values) {
		if(values.Length == 0) {
			return new T[0];
		}
		int x = values.Length;
		int y = values.First().Length;
		T[] result = new T[x * y];
		for(int i=0; i<x; i++) {
			for(int j=0; j<y; j++) {
				result[i*y + j] = values[i][j];
			}
		}
		return result;
	}

	public static void Replace<T>(this T[] values, T from, T to) {
		for(int i=0; i<values.Length; i++) {
			if(values[i].Equals(from)) {
				values[i] = to;
			}
		}
	}

	public static void Print<T>(this T[] values, bool inline=false) {
		string output = string.Empty;
		for(int i=0; i<values.Length; i++) {
			output += values[i].ToString() + (inline ? " " : "\n"); 
		}
		Debug.Log(output);
	}

}
