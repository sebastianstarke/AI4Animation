using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class StringExtensions {
    public static bool Contains(this string haystack, params string[] needles) {
        foreach (string needle in needles) {
            if (haystack.Contains(needle)) {
                return true;
            }
        }
        return false;
    }
}
