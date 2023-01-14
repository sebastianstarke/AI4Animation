using System;
using System.Reflection;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class GameObjectExtensions {

    public static T AddComponent<T>(this GameObject go, T toAdd) where T : Component {
        return go.AddComponent<T>().GetCopyOf(toAdd) as T;
    }

    public static T GetCopyOf<T>(this Component comp, T other) where T : Component {
        Type type = comp.GetType();
        if (type != other.GetType()) return null; // type mis-match
        BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Default | BindingFlags.DeclaredOnly;
        PropertyInfo[] pinfos = type.GetProperties(flags);
        foreach (var pinfo in pinfos) {
            if (pinfo.CanWrite) {
                try {
                    pinfo.SetValue(comp, pinfo.GetValue(other, null), null);
                }
                catch { } // In case of NotImplementedException being thrown. For some reason specifying that exception didn't seem to catch it, so I didn't catch anything specific.
            }
        }
        FieldInfo[] finfos = type.GetFields(flags);
        foreach (var finfo in finfos) {
            finfo.SetValue(comp, finfo.GetValue(other));
        }
        return comp as T;
    }

    public static GameObject Find(string name) {
        GameObject[] roots = SceneManager.GetActiveScene().GetRootGameObjects();
        GameObject result = null;
        for(int i=0; i<roots.Length; i++) {
            Recursion(roots[i].transform);
        }
        void Recursion(Transform t) {
            if(result != null) {
                return;
            }
            if(t.name == name) {
                result = t.gameObject;
            } else {
                for(int i=0; i<t.childCount; i++) {
                    Recursion(t.GetChild(i));
                }
            }
        }
        return result;
    }

    public static T Find<T>(bool onlyActive=false) where T : MonoBehaviour {
        GameObject[] roots = SceneManager.GetActiveScene().GetRootGameObjects();
        for(int i=0; i<roots.Length; i++) {
            T[] instances = roots[i].GetComponentsInChildren<T>();
            foreach(T result in instances) {
                if(!onlyActive) {
                    return result;
                } else if(result.gameObject.activeSelf) {
                    return result;
                }
            }
        }
        return null;
    }

}
