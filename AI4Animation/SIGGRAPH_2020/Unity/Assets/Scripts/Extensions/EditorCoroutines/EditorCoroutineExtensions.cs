#if UNITY_EDITOR
using System.Collections;
using UnityEditor;

public static class EditorCoroutineExtensions
{
	public static EditorCoroutines.EditorCoroutine StartCoroutine(this EditorWindow thisRef, IEnumerator coroutine)
	{
		return EditorCoroutines.StartCoroutine(coroutine, thisRef);
	}

	public static EditorCoroutines.EditorCoroutine StartCoroutine(this EditorWindow thisRef, string methodName)
	{
		return EditorCoroutines.StartCoroutine(methodName, thisRef);
	}

	public static EditorCoroutines.EditorCoroutine StartCoroutine(this EditorWindow thisRef, string methodName, object value)
	{
		return EditorCoroutines.StartCoroutine(methodName, value, thisRef);
	}

	public static void StopCoroutine(this EditorWindow thisRef, IEnumerator coroutine)
	{
		EditorCoroutines.StopCoroutine(coroutine, thisRef);
	}

	public static void StopCoroutine(this EditorWindow thisRef, string methodName)
	{
		EditorCoroutines.StopCoroutine(methodName, thisRef);
	}

	public static void StopAllCoroutines(this EditorWindow thisRef)
	{
		EditorCoroutines.StopAllCoroutines(thisRef);
	}
}
#endif