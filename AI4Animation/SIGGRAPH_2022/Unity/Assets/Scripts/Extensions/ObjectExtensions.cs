using System;
using UnityEngine;

public static class ObjectExtensions {
	
	public static T ToType<T>(this object o) {
		try {
			return (T)Convert.ChangeType(o, typeof(T));
		} catch {
			Debug.Log("Conversion from " + o.GetType() + " to " + typeof(T) + " is not supported: " + o.ToString());
		}
		return default(T);
    }

}
