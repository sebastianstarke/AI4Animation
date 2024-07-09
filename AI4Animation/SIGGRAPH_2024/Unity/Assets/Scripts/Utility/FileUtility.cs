using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class FileUtility {

	public static string[] LineToArray(string line, char separator) {
		return line.Split(separator);
	}

	public static int[] LineToInt(string line, char separator) {
		string[] items = line.Split(separator);
		int[] values = new int[items.Length];
		for(int i=0; i<items.Length; i++) {
			items[i] = Filter(items[i]);
			values[i] = ParseInt(items[i]);
		}
		return values;
	}

	public static float[] LineToFloat(string line, char separator) {
		string[] items = line.Split(separator);
		float[] values = new float[items.Length];
		for(int i=0; i<items.Length; i++) {
			items[i] = Filter(items[i]);
			values[i] = ParseFloat(items[i]);
		}
		return values;
	}

	public static double[] LineToDouble(string line, char separator) {
		string[] items = line.Split(separator);
		double[] values = new double[items.Length];
		for(int i=0; i<items.Length; i++) {
			items[i] = Filter(items[i]);
			values[i] = ParseDouble(items[i]);
		}
		return values;
	}

	public static float[][] ReadMatrix(string path) {
		string[] lines = ReadAllLines(path);
		float[][] values = new float[lines.Length][];
		for(int i=0; i<values.Length; i++) {
			values[i] = LineToFloat(lines[i], ' ');
		}
		return values;
	}

	public static string[] ReadAllLines(string path) {
		if(!File.Exists(path)) {
			Debug.Log("File at path '" + path + "' does not exist.");
			return new string[0];
		}
		return File.ReadAllLines(path);
	}

	public static int ReadInt(string value) {
		value = Filter(value);
		return ParseInt(value);
	}

	public static float ReadFloat(string value) {
		value = Filter(value);
		return ParseFloat(value);
	}

	public static float[] ReadArray(string value) {
		value = Filter(value);
		if(value.StartsWith(" ")) {
			value = value.Substring(1);
		}
		if(value.EndsWith(" ")) {
			value = value.Substring(0, value.Length-1);
		}
		string[] values = value.Split(' ');
		float[] array = new float[values.Length];
		for(int i=0; i<array.Length; i++) {
			array[i] = ParseFloat(values[i]);
		}
		return array;
	}

	public static string Filter(string value) {
		while(value.Contains("  ")) {
			value = value.Replace("  "," ");
		}
		while(value.Contains("< ")) {
			value = value.Replace("< ","<");
		}
		while(value.Contains(" >")) {
			value = value.Replace(" >",">");
		}
		while(value.Contains(" .")) {
			value = value.Replace(" ."," 0.");
		}
		while(value.Contains(". ")) {
			value = value.Replace(". ",".0");
		}
		while(value.Contains("<.")) {
			value = value.Replace("<.","<0.");
		}
		return value;
	}

	public static int ParseInt(string value) {
		int parsed = 0;
		if(int.TryParse(value, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out parsed)) {
			return parsed;
		} else {
			Debug.Log("Error parsing " + value + ".");
			return 0;
		}
	}

	public static float ParseFloat(string value) {
		float parsed = 0f;
		if(float.TryParse(value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out parsed)) {
			return parsed;
		} else {
			Debug.Log("Error parsing " + value + ".");
			return 0f;
		}
	}

	public static double ParseDouble(string value) {
		double parsed = 0f;
		if(double.TryParse(value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out parsed)) {
			return parsed;
		} else {
			Debug.Log("Error parsing " + value + ".");
			return 0.0;
		}
	}

}