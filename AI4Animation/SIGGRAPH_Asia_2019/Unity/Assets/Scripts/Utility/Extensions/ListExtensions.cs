using System;
using UnityEngine;
using System.Collections.Generic;

public static class ListExtensions {

	public static T First<T>(this List<T> list) {
		return list[0];
	}

	public static T Last<T>(this List<T> list) {
		return list[list.Count-1];
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

}
