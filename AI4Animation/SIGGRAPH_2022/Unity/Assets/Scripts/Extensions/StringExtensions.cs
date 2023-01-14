using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class StringExtensions {
    public static string[] Filter(this string[] values, params string[] filters) {
        List<string> filtered = new List<string>();
        foreach(string value in values) {
            if(value.ContainsAny(filters)) {
                filtered.Add(value);
            }
        }
        return filtered.ToArray();
    }

    public static bool ContainsAny(this string haystack, params string[] needles) {
        foreach (string needle in needles) {
            if(haystack.Contains(needle)) {
                return true;
            }
        }
        return false;
    }

    public static bool EqualsAny(this string haystack, params string[] needles) {
        foreach (string needle in needles) {
            if(haystack == needle) {
                return true;
            }
        }
        return false;
    }

    public static bool EndsWithAny(this string haystack, params string[] needles) {
        foreach (string needle in needles) {
            if(haystack.EndsWith(needle)) {
                return true;
            }
        }
        return false;
    }

    public static int Count(this string text, char item) {
        int count = 0;
        for(int i=0; i<text.Length; i++) {
            if(text[i] == item) {
                count += 1;
            }
        }
        return count;
    }

    public static int ToInt(this string text) {
        return int.Parse(text);
    }
}
