using System;
using UnityEngine;

public static class Arrays {
	
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

	public static void Remove<T>(ref T[] array, int index) {
		if(index >= 0 && index < array.Length) {
			for(int i=index; i<array.Length-1; i++) {
				array[i] = array[i+1];
			}
			Shrink(ref array);
		}
	}

	public static void Remove<T>(ref T[] array, T element) {
		Remove(ref array, Array.FindIndex(array, x => x.Equals(element)));
	}

	public static void Expand<T>(ref T[] array) {
		Array.Resize(ref array, array.Length+1);
	}

	public static void Shrink<T>(ref T[] array) {
		if(array.Length > 0) {
			Array.Resize(ref array, array.Length-1);
		}
	}

	public static void Clear<T>(ref T[] array) {
		Array.Resize(ref array, 0);
	}

}
