using System;
using System.Collections.Generic;
using UnityEngine;

public static class ListExtensions {
	
	public static T First<T>(this List<T> list) {
		if(list.Count == 0) {
			return default(T);
		}
		return list[0];
	}

	public static T Last<T>(this List<T> list) {
		if(list.Count == 0) {
			return default(T);
		}
		return list[list.Count-1];
	}

	public static T Random<T>(this List<T> list) {
		if(list.Count == 0) {
			return default(T);
		}
		return list[UnityEngine.Random.Range(0, list.Count)];
	}

	public static T Random<T>(this List<T> list, params float[] probabilities) {
		if(list.Count == 0) {
			return default(T);
		}
		float total = 0f;
        foreach(float elem in probabilities) {
            total += elem;
        }
		float randomPoint = UnityEngine.Random.value * total;
        for(int i=0; i<probabilities.Length; i++) {
            if(randomPoint < probabilities[i]) {
                return list[i];
            } else {
                randomPoint -= probabilities[i];
            }
        }
		return list.Last();
	}

	public static void MoveBack<T>(this List<T> list, int index) {
		if(index <= list.Count-1 && index > 0) {
			T previous = list[index-1];
			T element = list[index];
			list[index-1] = element;
			list[index] = previous;
		}
	}

	public static void MoveForward<T>(this List<T> list, int index) {
		if(index < list.Count-1 && index >= 0) {
			T next = list[index+1];
			T element = list[index];
			list[index+1] = element;
			list[index] = next;
		}
	}

	public static int Min(this List<int> values) {
		if(values.Count == 0) {
			return 0;
		}
		int min = int.MaxValue;
		for(int i=0; i<values.Count; i++) {
			min = Mathf.Min(min, values[i]);
		}
		return min;
	}

	public static float Min(this List<float> values) {
		if(values.Count == 0) {
			return 0f;
		}
		float min = float.MaxValue;
		for(int i=0; i<values.Count; i++) {
			min = Mathf.Min(min, values[i]);
		}
		return min;
	}

	public static double Min(this List<double> values) {
		if(values.Count == 0) {
			return 0.0;
		}
		double min = double.MaxValue;
		for(int i=0; i<values.Count; i++) {
			min = System.Math.Min(min, values[i]);
		}
		return min;
	}

	public static int Max(this List<int> values) {
		if(values.Count == 0) {
			return 0;
		}
		int max = int.MinValue;
		for(int i=0; i<values.Count; i++) {
			max = Mathf.Max(max, values[i]);
		}
		return max;
	}

	public static float Max(this List<float> values) {
		if(values.Count == 0) {
			return 0f;
		}
		float max = float.MinValue;
		for(int i=0; i<values.Count; i++) {
			max = Mathf.Max(max, values[i]);
		}
		return max;
	}

	public static double Max(this List<double> values) {
		if(values.Count == 0) {
			return 0.0;
		}
		double max = double.MinValue;
		for(int i=0; i<values.Count; i++) {
			max = System.Math.Max(max, values[i]);
		}
		return max;
	}

	public static bool ContainsAny<T>(this List<T> list, List<T> candidates) {
		foreach(T item in candidates) {
			if(list.Contains(item)) {
				return true;
			}
		}
		return false;
	}

	public static bool ContainsAny<T>(this List<T> list, T[] candidates) {
		foreach(T item in candidates) {
			if(list.Contains(item)) {
				return true;
			}
		}
		return false;
	}

}
