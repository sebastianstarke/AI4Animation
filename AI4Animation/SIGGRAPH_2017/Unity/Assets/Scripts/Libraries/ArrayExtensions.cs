using System;
using UnityEngine;

public static class ArrayExtensions {

	public static void Add<T>(ref T[] array, T element) {
		Expand(ref array);
		array[array.Length-1] = element;
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

	public static void RemoveAt<T>(ref T[] array, int index) {
		if(index >= 0 && index < array.Length) {
			for(int i=index; i<array.Length-1; i++) {
				array[i] = array[i+1];
			}
			Shrink(ref array);
		}
	}

	public static void Remove<T>(ref T[] array, T element) {
		RemoveAt(ref array, FindIndex(ref array, element));
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

	public static int FindIndex<T>(ref T[] array, T element) {
		return Array.FindIndex(array, x => x.Equals(element));
	}

	public static T Find<T>(ref T[] array, T element) {
		return Array.Find(array, x => x.Equals(element));
	}

	public static bool Contains<T>(ref T[] array, T element) {
		return FindIndex(ref array, element) >= 0;
	}

	public static T[] Concat<T>(T[] lhs, T[] rhs) {
		T[] result = new T[lhs.Length + rhs.Length];
		lhs.CopyTo(result, 0);
		rhs.CopyTo(result, lhs.Length);
		return result;
	}

	public static T[] Concat<T>(T lhs, T[] rhs) {
		T[] clone = (T[])rhs.Clone();
		Insert(ref clone, lhs, 0);
		return clone;
	}

	public static T[] Concat<T>(T[] lhs, T rhs) {
		T[] clone = (T[])lhs.Clone();
		ArrayExtensions.Add(ref clone, rhs);
		return clone;
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

	public static float Mean(this int[] values) {
		if(values.Length == 0) {
			return 0;
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

	public static float Sigma(this int[] values) {
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
		return Mathf.Sqrt(variance);
	}

	public static float Sigma(this float[] values) {
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
		return Mathf.Sqrt(variance);
	}

	public static double Sigma(this double[] values) {
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
		return System.Math.Sqrt(variance);
	}

	public static void Zero(this float[] values) {
		for(int i=0; i<values.Length; i++) {
			values[i] = 0f;
		}
	}

	public static void Print(this double[] values) {
		string output = "[";
		for(int i=0; i<values.Length; i++) {
			output += values[i].ToString() + (i==values.Length-1 ? "]" : ", ");
		}
		Debug.Log(output);
	}

	public static void Print(this float[] values) {
		string output = "[";
		for(int i=0; i<values.Length; i++) {
			output += values[i].ToString() + (i==values.Length-1 ? "]" : ", ");
		}
		Debug.Log(output);
	}

	public static void Print(this int[] values) {
		string output = "[";
		for(int i=0; i<values.Length; i++) {
			output += values[i].ToString() + (i==values.Length-1 ? "]" : ", ");
		}
		Debug.Log(output);
	}

}
