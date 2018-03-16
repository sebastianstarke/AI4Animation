using UnityEngine;
using System.Collections;
using UnityEditor;
using System.Collections.Generic;
using System;
using System.Reflection;

namespace EditorCoroutines
{
	public class EditorCoroutines
	{
		public class EditorCoroutine
		{
			public ICoroutineYield currentYield = new YieldDefault();
			public IEnumerator routine;
			public string routineUniqueHash;
			public string ownerUniqueHash;
			public string MethodName = "";

			public int ownerHash;
			public string ownerType;

			public bool finished = false;

			public EditorCoroutine(IEnumerator routine, int ownerHash, string ownerType)
			{
				this.routine = routine;
				this.ownerHash = ownerHash;
				this.ownerType = ownerType;
				ownerUniqueHash = ownerHash + "_" + ownerType;

				if (routine != null)
				{
					string[] split = routine.ToString().Split('<', '>');
					if (split.Length == 3)
					{
						this.MethodName = split[1];
					}
				}

				routineUniqueHash = ownerHash + "_" + ownerType + "_" + MethodName;
			}

			public EditorCoroutine(string methodName, int ownerHash, string ownerType)
			{
				MethodName = methodName;
				this.ownerHash = ownerHash;
				this.ownerType = ownerType;
				ownerUniqueHash = ownerHash + "_" + ownerType;
				routineUniqueHash = ownerHash + "_" + ownerType + "_" + MethodName;
			}
		}

		public interface ICoroutineYield
		{
			bool IsDone(float deltaTime);
		}

		struct YieldDefault : ICoroutineYield
		{
			public bool IsDone(float deltaTime)
			{
				return true;
			}
		}

		struct YieldWaitForSeconds : ICoroutineYield
		{
			public float timeLeft;

			public bool IsDone(float deltaTime)
			{
				timeLeft -= deltaTime;
				return timeLeft < 0;
			}
		}

		struct YieldWWW : ICoroutineYield
		{
			public WWW Www;

			public bool IsDone(float deltaTime)
			{
				return Www.isDone;
			}
		}

		struct YieldAsync : ICoroutineYield
		{
			public AsyncOperation asyncOperation;

			public bool IsDone(float deltaTime)
			{
				return asyncOperation.isDone;
			}
		}

		struct YieldNestedCoroutine : ICoroutineYield
		{
			public EditorCoroutine coroutine;

			public bool IsDone(float deltaTime)
			{
				return coroutine.finished;
			}
		}

		static EditorCoroutines instance = null;

		Dictionary<string, List<EditorCoroutine>> coroutineDict = new Dictionary<string, List<EditorCoroutine>>();
		List<List<EditorCoroutine>> tempCoroutineList = new List<List<EditorCoroutine>>();

		Dictionary<string, Dictionary<string, EditorCoroutine>> coroutineOwnerDict =
			new Dictionary<string, Dictionary<string, EditorCoroutine>>();

		DateTime previousTimeSinceStartup;

		/// <summary>Starts a coroutine.</summary>
		/// <param name="routine">The coroutine to start.</param>
		/// <param name="thisReference">Reference to the instance of the class containing the method.</param>
		public static EditorCoroutine StartCoroutine(IEnumerator routine, object thisReference)
		{
			CreateInstanceIfNeeded();
			return instance.GoStartCoroutine(routine, thisReference);
		}

		/// <summary>Starts a coroutine.</summary>
		/// <param name="methodName">The name of the coroutine method to start.</param>
		/// <param name="thisReference">Reference to the instance of the class containing the method.</param>
		public static EditorCoroutine StartCoroutine(string methodName, object thisReference)
		{
			return StartCoroutine(methodName, null, thisReference);
		}

		/// <summary>Starts a coroutine.</summary>
		/// <param name="methodName">The name of the coroutine method to start.</param>
		/// <param name="value">The parameter to pass to the coroutine.</param>
		/// <param name="thisReference">Reference to the instance of the class containing the method.</param>
		public static EditorCoroutine StartCoroutine(string methodName, object value, object thisReference)
		{
			MethodInfo methodInfo = thisReference.GetType()
				.GetMethod(methodName, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
			if (methodInfo == null)
			{
				Debug.LogError("Coroutine '" + methodName + "' couldn't be started, the method doesn't exist!");
			}
			object returnValue;

			if (value == null)
			{
				returnValue = methodInfo.Invoke(thisReference, null);
			}
			else
			{
				returnValue = methodInfo.Invoke(thisReference, new object[] {value});
			}

			if (returnValue is IEnumerator)
			{
				CreateInstanceIfNeeded();
				return instance.GoStartCoroutine((IEnumerator) returnValue, thisReference);
			}
			else
			{
				Debug.LogError("Coroutine '" + methodName + "' couldn't be started, the method doesn't return an IEnumerator!");
			}

			return null;
		}

		/// <summary>Stops all coroutines being the routine running on the passed instance.</summary>
		/// <param name="routine"> The coroutine to stop.</param>
		/// <param name="thisReference">Reference to the instance of the class containing the method.</param>
		public static void StopCoroutine(IEnumerator routine, object thisReference)
		{
			CreateInstanceIfNeeded();
			instance.GoStopCoroutine(routine, thisReference);
		}

		/// <summary>
		/// Stops all coroutines named methodName running on the passed instance.</summary>
		/// <param name="methodName"> The name of the coroutine method to stop.</param>
		/// <param name="thisReference">Reference to the instance of the class containing the method.</param>
		public static void StopCoroutine(string methodName, object thisReference)
		{
			CreateInstanceIfNeeded();
			instance.GoStopCoroutine(methodName, thisReference);
		}

		/// <summary>
		/// Stops all coroutines running on the passed instance.</summary>
		/// <param name="thisReference">Reference to the instance of the class containing the method.</param>
		public static void StopAllCoroutines(object thisReference)
		{
			CreateInstanceIfNeeded();
			instance.GoStopAllCoroutines(thisReference);
		}

		static void CreateInstanceIfNeeded()
		{
			if (instance == null)
			{
				instance = new EditorCoroutines();
				instance.Initialize();
			}
		}

		void Initialize()
		{
			previousTimeSinceStartup = DateTime.Now;
			EditorApplication.update += OnUpdate;
		}

		void GoStopCoroutine(IEnumerator routine, object thisReference)
		{
			GoStopActualRoutine(CreateCoroutine(routine, thisReference));
		}

		void GoStopCoroutine(string methodName, object thisReference)
		{
			GoStopActualRoutine(CreateCoroutineFromString(methodName, thisReference));
		}

		void GoStopActualRoutine(EditorCoroutine routine)
		{
			if (coroutineDict.ContainsKey(routine.routineUniqueHash))
			{
				coroutineOwnerDict[routine.ownerUniqueHash].Remove(routine.routineUniqueHash);
				coroutineDict.Remove(routine.routineUniqueHash);
			}
		}

		void GoStopAllCoroutines(object thisReference)
		{
			EditorCoroutine coroutine = CreateCoroutine(null, thisReference);
			if (coroutineOwnerDict.ContainsKey(coroutine.ownerUniqueHash))
			{
				foreach (var couple in coroutineOwnerDict[coroutine.ownerUniqueHash])
				{
					coroutineDict.Remove(couple.Value.routineUniqueHash);
				}
				coroutineOwnerDict.Remove(coroutine.ownerUniqueHash);
			}
		}

		EditorCoroutine GoStartCoroutine(IEnumerator routine, object thisReference)
		{
			if (routine == null)
			{
				Debug.LogException(new Exception("IEnumerator is null!"), null);
			}
			EditorCoroutine coroutine = CreateCoroutine(routine, thisReference);
			GoStartCoroutine(coroutine);
			return coroutine;
		}

		void GoStartCoroutine(EditorCoroutine coroutine)
		{
			if (!coroutineDict.ContainsKey(coroutine.routineUniqueHash))
			{
				List<EditorCoroutine> newCoroutineList = new List<EditorCoroutine>();
				coroutineDict.Add(coroutine.routineUniqueHash, newCoroutineList);
			}
			coroutineDict[coroutine.routineUniqueHash].Add(coroutine);

			if (!coroutineOwnerDict.ContainsKey(coroutine.ownerUniqueHash))
			{
				Dictionary<string, EditorCoroutine> newCoroutineDict = new Dictionary<string, EditorCoroutine>();
				coroutineOwnerDict.Add(coroutine.ownerUniqueHash, newCoroutineDict);
			}

			// If the method from the same owner has been stored before, it doesn't have to be stored anymore,
			// One reference is enough in order for "StopAllCoroutines" to work
			if (!coroutineOwnerDict[coroutine.ownerUniqueHash].ContainsKey(coroutine.routineUniqueHash))
			{
				coroutineOwnerDict[coroutine.ownerUniqueHash].Add(coroutine.routineUniqueHash, coroutine);
			}

			MoveNext(coroutine);
		}

		EditorCoroutine CreateCoroutine(IEnumerator routine, object thisReference)
		{
			return new EditorCoroutine(routine, thisReference.GetHashCode(), thisReference.GetType().ToString());
		}

		EditorCoroutine CreateCoroutineFromString(string methodName, object thisReference)
		{
			return new EditorCoroutine(methodName, thisReference.GetHashCode(), thisReference.GetType().ToString());
		}

		void OnUpdate()
		{
			float deltaTime = (float) (DateTime.Now.Subtract(previousTimeSinceStartup).TotalMilliseconds / 1000.0f);

			previousTimeSinceStartup = DateTime.Now;
			if (coroutineDict.Count == 0)
			{
				return;
			}

			tempCoroutineList.Clear();
			foreach (var pair in coroutineDict)
				tempCoroutineList.Add(pair.Value);

			for (var i = tempCoroutineList.Count - 1; i >= 0; i--)
			{
				List<EditorCoroutine> coroutines = tempCoroutineList[i];

				for (int j = coroutines.Count - 1; j >= 0; j--)
				{
					EditorCoroutine coroutine = coroutines[j];

					if (!coroutine.currentYield.IsDone(deltaTime))
					{
						continue;
					}

					if (!MoveNext(coroutine))
					{
						coroutines.RemoveAt(j);
						coroutine.currentYield = null;
						coroutine.finished = true;
					}

					if (coroutines.Count == 0)
					{
						coroutineDict.Remove(coroutine.ownerUniqueHash);
					}
				}
			}
		}

		static bool MoveNext(EditorCoroutine coroutine)
		{
			if (coroutine.routine.MoveNext())
			{
				return Process(coroutine);
			}

			return false;
		}

		// returns false if no next, returns true if OK
		static bool Process(EditorCoroutine coroutine)
		{
			object current = coroutine.routine.Current;
			if (current == null)
			{
				return false;
			}
			else if (current is WaitForSeconds)
			{
				float seconds = float.Parse(GetInstanceField(typeof(WaitForSeconds), current, "m_Seconds").ToString());
				coroutine.currentYield = new YieldWaitForSeconds() {timeLeft = (float) seconds};
			}
			else if (current is WWW)
			{
				coroutine.currentYield = new YieldWWW {Www = (WWW) current};
			}
			else if (current is WaitForFixedUpdate)
			{
				coroutine.currentYield = new YieldDefault();
			}
			else if (current is AsyncOperation)
			{
				coroutine.currentYield = new YieldAsync {asyncOperation = (AsyncOperation) current};
			}
			else if (current is EditorCoroutine)
			{
				coroutine.currentYield = new YieldNestedCoroutine { coroutine= (EditorCoroutine) current};
			}
			else
			{
				Debug.LogException(
					new Exception("<" + coroutine.MethodName + "> yielded an unknown or unsupported type! (" + current.GetType() + ")"),
					null);
				coroutine.currentYield = new YieldDefault();
			}
			return true;
		}

		static object GetInstanceField(Type type, object instance, string fieldName)
		{
			BindingFlags bindFlags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static;
			FieldInfo field = type.GetField(fieldName, bindFlags);
			return field.GetValue(instance);
		}
	}
}
