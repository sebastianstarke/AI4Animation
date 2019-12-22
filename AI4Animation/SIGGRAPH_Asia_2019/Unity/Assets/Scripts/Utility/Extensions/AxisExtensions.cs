using UnityEngine;

public enum Axis {XPositive, YPositive, ZPositive, XNegative, YNegative, ZNegative, None};

public static class AxisExtensions {
	public static float GetValue(this Vector3 vector, Axis axis) {
		switch(axis) {
			case Axis.None:
			return 0f;
			case Axis.XPositive:
			return vector.x;
			case Axis.XNegative:
			return -vector.x;
			case Axis.YPositive:
			return vector.y;
			case Axis.YNegative:
			return -vector.y;
			case Axis.ZPositive:
			return vector.z;
			case Axis.ZNegative:
			return -vector.z;
		}
		return 0f;
	}

	public static Color GetColor(this Axis axis) {
		switch(axis) {
			case Axis.None:
			return Color.white;
			case Axis.XPositive:
			return Color.red;
			case Axis.XNegative:
			return Color.red;
			case Axis.YPositive:
			return Color.green;
			case Axis.YNegative:
			return Color.green;
			case Axis.ZPositive:
			return Color.blue;
			case Axis.ZNegative:
			return Color.blue;
		}
		return Color.white;
	}

	public static string GetName(this Axis axis) {
		switch(axis) {
			case Axis.None:
			return "?";
			case Axis.XPositive:
			return "X";
			case Axis.XNegative:
			return "-X";
			case Axis.YPositive:
			return "Y";
			case Axis.YNegative:
			return "-Y";
			case Axis.ZPositive:
			return "Z";
			case Axis.ZNegative:
			return "-Z";
		}
		return "?";
	}

    public static Vector3 GetAxis(this Axis axis) {
		switch(axis) {
			case Axis.None:
			return Vector3.zero;
			case Axis.XPositive:
			return Vector3.right;
			case Axis.XNegative:
			return Vector3.left;
			case Axis.YPositive:
			return Vector3.up;
			case Axis.YNegative:
			return Vector3.down;
			case Axis.ZPositive:
			return Vector3.forward;
			case Axis.ZNegative:
			return Vector3.back;
		}
		return Vector3.zero;
    }
}