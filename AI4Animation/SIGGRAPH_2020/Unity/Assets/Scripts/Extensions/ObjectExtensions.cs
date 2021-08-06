using System;

public static class ObjectExtensions {
	
	public static T ToType<T>(this object o) {
        return(T)Convert.ChangeType(o, typeof(T));
    }

}
