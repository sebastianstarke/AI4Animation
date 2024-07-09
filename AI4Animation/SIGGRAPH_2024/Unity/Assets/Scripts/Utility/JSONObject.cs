/*
Copyright (c) 2010-2021 Matt Schoen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

//#define JSONOBJECT_DISABLE_PRETTY_PRINT // Use when you no longer need to read JSON to disable pretty Print system-wide
//#define JSONOBJECT_USE_FLOAT //Use floats for numbers instead of doubles (enable if you don't need support for doubles and want to cut down on significant digits in output)
//#define JSONOBJECT_POOLING //Create JSONObjects from a pool and prevent finalization by returning objects to the pool

// ReSharper disable ArrangeAccessorOwnerBody
// ReSharper disable MergeConditionalExpression
// ReSharper disable UseStringInterpolation

#if UNITY_2 || UNITY_3 || UNITY_4 || UNITY_5 || UNITY_5_3_OR_NEWER
#define USING_UNITY
#endif

using System;
using System.Diagnostics;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Globalization;

#if USING_UNITY
using UnityEngine;
using Debug = UnityEngine.Debug;
#endif

namespace Defective.JSON {
	public class JSONObject : IEnumerable {
#if JSONOBJECT_POOLING
		const int MaxPoolSize = 100000;
		static readonly Queue<JSONObject> JSONObjectPool = new Queue<JSONObject>();
		static readonly Queue<List<JSONObject>> JSONObjectListPool = new Queue<List<JSONObject>>();
		static readonly Queue<List<string>> StringListPool = new Queue<List<string>>();
		static bool poolingEnabled = true;
#endif

#if !JSONOBJECT_DISABLE_PRETTY_PRINT
		const string Newline = "\r\n";
		const string Tab = "\t";
#endif

		const string Infinity = "Infinity";
		const string NegativeInfinity = "-Infinity";
		const string NaN = "NaN";
		const string True = "true";
		const string False = "false";
		const string Null = "null";

		const float MaxFrameTime = 0.008f;
		static readonly Stopwatch PrintWatch = new Stopwatch();
		public static readonly char[] Whitespace = { ' ', '\r', '\n', '\t', '\uFEFF', '\u0009' };

		public enum Type {
			Null,
			String,
			Number,
			Object,
			Array,
			Bool,
			Baked
		}

		public struct ParseResult {
			public readonly JSONObject result;
			public readonly int offset;
			public readonly bool pause;

			public ParseResult(JSONObject result, int offset, bool pause) {
				this.result = result;
				this.offset = offset;
				this.pause = pause;
			}
		}

		public delegate void FieldNotFound(string name);
		public delegate void GetFieldResponse(JSONObject jsonObject);

		public Type type = Type.Null;

		public List<JSONObject> list;
		public List<string> keys;
		public string stringValue;
		public bool isInteger;
		public long longValue;
		public bool boolValue;
#if JSONOBJECT_USE_FLOAT
		public float floatValue;
#else
		public double doubleValue;
#endif

		bool isPooled;

		public int count {
			get {
				return list == null ? 0 : list.Count;
			}
		}

		public bool isContainer {
			get { return type == Type.Array || type == Type.Object; }
		}

		public int intValue {
			get {
				return (int) longValue;
			}
			set {
				longValue = value;
			}
		}

#if JSONOBJECT_USE_FLOAT
		public double doubleValue {
			get {
				return floatValue;
			}
			set {
				floatValue = (float) value;
			}
		}
#else
		public float floatValue {
			get {
				return (float) doubleValue;
			}
			set {
				doubleValue = value;
			}
		}
#endif

		public delegate void AddJSONContents(JSONObject self);

		public static JSONObject nullObject {
			get { return Create(Type.Null); }
		}

		public static JSONObject emptyObject {
			get { return Create(Type.Object); }
		}

		public static JSONObject emptyArray {
			get { return Create(Type.Array); }
		}

		public JSONObject(Type type) { this.type = type; }

		public JSONObject(bool value) {
			type = Type.Bool;
			boolValue = value;
		}

		public JSONObject(float value) {
			type = Type.Number;
#if JSONOBJECT_USE_FLOAT
			floatValue = value;
#else
			doubleValue = value;
#endif
		}

		public JSONObject(double value) {
			type = Type.Number;
#if JSONOBJECT_USE_FLOAT
			floatValue = (float)value;
#else
			doubleValue = value;
#endif
		}

		public JSONObject(int value) {
			type = Type.Number;
			longValue = value;
			isInteger = true;
#if JSONOBJECT_USE_FLOAT
			floatValue = value;
#else
			doubleValue = value;
#endif
		}

		public JSONObject(long value) {
			type = Type.Number;
			longValue = value;
			isInteger = true;
#if JSONOBJECT_USE_FLOAT
			floatValue = value;
#else
			doubleValue = value;
#endif
		}

		public JSONObject(Dictionary<string, string> dictionary) {
			type = Type.Object;
			keys = CreateStringList();
			list = CreateJSONObjectList();
			foreach (KeyValuePair<string, string> kvp in dictionary) {
				keys.Add(kvp.Key);
				list.Add(CreateStringObject(kvp.Value));
			}
		}

		public JSONObject(Dictionary<string, JSONObject> dictionary) {
			type = Type.Object;
			keys = CreateStringList();
			list = CreateJSONObjectList();
			foreach (KeyValuePair<string, JSONObject> kvp in dictionary) {
				keys.Add(kvp.Key);
				list.Add(kvp.Value);
			}
		}

		public JSONObject(AddJSONContents content) {
			content.Invoke(this);
		}

		public JSONObject(JSONObject[] objects) {
			type = Type.Array;
			list = CreateJSONObjectList();
			list.AddRange(objects);
		}

		public JSONObject(List<JSONObject> objects) {
			type = Type.Array;
			list = objects;
		}

		/// <summary>
		/// Convenience function for creating a JSONObject containing a string.
		/// This is not part of the constructor so that malformed JSON data doesn't just turn into a string object
		/// </summary>
		/// <param name="value">The string value for the new JSONObject</param>
		/// <returns>Thew new JSONObject</returns>
		public static JSONObject StringObject(string value) {
			return CreateStringObject(value);
		}

		public void Absorb(JSONObject other) {
			var otherList = other.list;
			if (otherList != null) {
				if (list == null)
					list = CreateJSONObjectList();

				list.AddRange(otherList);
			}

			var otherKeys = other.keys;
			if (otherKeys != null) {
				if (keys == null)
					keys = CreateStringList();

				keys.AddRange(otherKeys);
			}

			stringValue = other.stringValue;
#if JSONOBJECT_USE_FLOAT
			floatValue = other.floatValue;
#else
			doubleValue = other.doubleValue;
#endif

			isInteger = other.isInteger;
			longValue = other.longValue;
			boolValue = other.boolValue;
			type = other.type;
		}

		public static JSONObject Create() {
#if JSONOBJECT_POOLING
			lock (JSONObjectPool) {
				if (JSONObjectPool.Count > 0) {
					var result = JSONObjectPool.Dequeue();

					result.isPooled = false;
					return result;
				}
			}
#endif

			return new JSONObject();
		}

		public static JSONObject Create(Type type) {
			var jsonObject = Create();
			jsonObject.type = type;
			return jsonObject;
		}

		public static JSONObject Create(bool value) {
			var jsonObject = Create();
			jsonObject.type = Type.Bool;
			jsonObject.boolValue = value;
			return jsonObject;
		}

		public static JSONObject Create(float value) {
			var jsonObject = Create();
			jsonObject.type = Type.Number;
#if JSONOBJECT_USE_FLOAT
			jsonObject.floatValue = value;
#else
			jsonObject.doubleValue = value;
#endif

			return jsonObject;
		}

		public static JSONObject Create(double value) {
			var jsonObject = Create();
			jsonObject.type = Type.Number;
#if JSONOBJECT_USE_FLOAT
			jsonObject.floatValue = (float)value;
#else
			jsonObject.doubleValue = value;
#endif

			return jsonObject;
		}

		public static JSONObject Create(int value) {
			var jsonObject = Create();
			jsonObject.type = Type.Number;
			jsonObject.isInteger = true;
			jsonObject.longValue = value;
#if JSONOBJECT_USE_FLOAT
			jsonObject.floatValue = value;
#else
			jsonObject.doubleValue = value;
#endif

			return jsonObject;
		}

		public static JSONObject Create(long value) {
			var jsonObject = Create();
			jsonObject.type = Type.Number;
			jsonObject.isInteger = true;
			jsonObject.longValue = value;
#if JSONOBJECT_USE_FLOAT
			jsonObject.floatValue = value;
#else
			jsonObject.doubleValue = value;
#endif

			return jsonObject;
		}

		public static JSONObject CreateStringObject(string value) {
			var jsonObject = Create();
			jsonObject.type = Type.String;
			jsonObject.stringValue = value;
			return jsonObject;
		}

		public static JSONObject CreateBakedObject(string value) {
			var bakedObject = Create();
			bakedObject.type = Type.Baked;
			bakedObject.stringValue = value;
			return bakedObject;
		}

		/// <summary>
		/// Create a JSONObject (using pooling if enabled) using a string containing valid JSON
		/// </summary>
		/// <param name="jsonString">A string containing valid JSON to be parsed into objects</param>
		/// <param name="offset">An offset into the string at which to start parsing</param>
		/// <param name="endOffset">The length of the string after the offset to parse
		/// Specify a length of -1 (default) to use the full string length</param>
		/// <param name="maxDepth">The maximum depth for the parser to search.</param>
		/// <param name="storeExcessLevels">Whether to store levels beyond maxDepth in baked JSONObjects</param>
		/// <returns>A JSONObject containing the parsed data</returns>
		public static JSONObject Create(string jsonString, int offset = 0, int endOffset = -1, int maxDepth = -1, bool storeExcessLevels = false) {
			var jsonObject = Create();
			Parse(jsonString, ref offset, endOffset, jsonObject, maxDepth, storeExcessLevels);
			return jsonObject;
		}

		public static JSONObject Create(AddJSONContents content) {
			var jsonObject = Create();
			content.Invoke(jsonObject);
			return jsonObject;
		}

		public static JSONObject Create(JSONObject[] objects) {
			var jsonObject = Create();
			jsonObject.type = Type.Array;
			var list = CreateJSONObjectList();
			list.AddRange(objects);
			jsonObject.list = list;

			return jsonObject;
		}

		public static JSONObject Create(List<JSONObject> objects) {
			var jsonObject = Create();
			jsonObject.type = Type.Array;
			jsonObject.list = objects;

			return jsonObject;
		}

		public static JSONObject Create(Dictionary<string, string> dictionary) {
			var jsonObject = Create();
			jsonObject.type = Type.Object;
			var keys = CreateStringList();
			jsonObject.keys = keys;
			var list = CreateJSONObjectList();
			jsonObject.list = list;
			foreach (var kvp in dictionary) {
				keys.Add(kvp.Key);
				list.Add(CreateStringObject(kvp.Value));
			}

			return jsonObject;
		}

		public static JSONObject Create(Dictionary<string, JSONObject> dictionary) {
			var jsonObject = Create();
			jsonObject.type = Type.Object;
			var keys = CreateStringList();
			jsonObject.keys = keys;
			var list = CreateJSONObjectList();
			jsonObject.list = list;
			foreach (var kvp in dictionary) {
				keys.Add(kvp.Key);
				list.Add(kvp.Value);
			}

			return jsonObject;
		}

		/// <summary>
		/// Create a JSONObject (using pooling if enabled) using a string containing valid JSON
		/// </summary>
		/// <param name="jsonString">A string containing valid JSON to be parsed into objects</param>
		/// <param name="offset">An offset into the string at which to start parsing</param>
		/// <param name="endOffset">The length of the string after the offset to parse
		/// Specify a length of -1 (default) to use the full string length</param>
		/// <param name="maxDepth">The maximum depth for the parser to search.</param>
		/// <param name="storeExcessLevels">Whether to store levels beyond maxDepth in baked JSONObjects</param>
		/// <returns>A JSONObject containing the parsed data</returns>
		public static IEnumerable<ParseResult> CreateAsync(string jsonString, int offset = 0, int endOffset = -1, int maxDepth = -1, bool storeExcessLevels = false) {
			var jsonObject = Create();
			PrintWatch.Reset();
			PrintWatch.Start();
			foreach (var e in ParseAsync(jsonString, offset, endOffset, jsonObject, maxDepth, storeExcessLevels)) {
				if (e.pause)
					yield return e;

				offset = e.offset;
			}

			yield return new ParseResult(jsonObject, offset, false);
		}

		public JSONObject() { }

		/// <summary>
		/// Construct a new JSONObject using a string containing valid JSON
		/// </summary>
		/// <param name="jsonString">A string containing valid JSON to be parsed into objects</param>
		/// <param name="offset">An offset into the string at which to start parsing</param>
		/// <param name="endOffset">The length of the string after the offset to parse
		/// Specify a length of -1 (default) to use the full string length</param>
		/// <param name="maxDepth">The maximum depth for the parser to search.</param>
		/// <param name="storeExcessLevels">Whether to store levels beyond maxDepth in baked JSONObjects</param>
		public JSONObject(string jsonString, int offset = 0, int endOffset = -1, int maxDepth = -1, bool storeExcessLevels = false) {
			Parse(jsonString, ref offset, endOffset, this, maxDepth, storeExcessLevels);
		}

		// ReSharper disable UseNameofExpression
		static bool BeginParse(string inputString, int offset, ref int endOffset, JSONObject container, int maxDepth, bool storeExcessLevels) {
			if (container == null)
				throw new ArgumentNullException("container");

			if (maxDepth == 0) {
				if (storeExcessLevels) {
					container.stringValue = inputString;
					container.type = Type.Baked;
				} else {
					container.type = Type.Null;
				}

				return false;
			}

			var stringLength = inputString.Length;
			if (endOffset == -1)
				endOffset = stringLength - 1;

			if (string.IsNullOrEmpty(inputString)) {
				return false;
			}

			if (endOffset >= stringLength)
				throw new ArgumentException("Cannot parse if end offset is greater than or equal to string length", "endOffset");

			if (offset > endOffset)
				throw new ArgumentException("Cannot parse if offset is greater than end offset", "offset");

			return true;
		}
		// ReSharper restore UseNameofExpression

		static void Parse(string inputString, ref int offset, int endOffset, JSONObject container, int maxDepth,
			bool storeExcessLevels, int depth = 0, bool isRoot = true) {
			if (!BeginParse(inputString, offset, ref endOffset, container, maxDepth, storeExcessLevels))
				return;

			var startOffset = offset;
			var quoteStart = 0;
			var quoteEnd = 0;
			var lastValidOffset = offset;
			var openQuote = false;
			var bakeDepth = 0;

			while (offset <= endOffset) {
				var currentCharacter = inputString[offset++];
				if (Array.IndexOf(Whitespace, currentCharacter) > -1)
					continue;

				JSONObject newContainer;
				switch (currentCharacter) {
					case '\\':
						offset++;
						break;
					case '{':
						if (openQuote)
							break;

						if (maxDepth >= 0 && depth >= maxDepth) {
							bakeDepth++;
							break;
						}

						newContainer = container;
						if (!isRoot) {
							newContainer = Create();
							SafeAddChild(container, newContainer);
						}

						newContainer.type = Type.Object;
						Parse(inputString, ref offset, endOffset, newContainer, maxDepth, storeExcessLevels, depth + 1, false);

						break;
					case '[':
						if (openQuote)
							break;

						if (maxDepth >= 0 && depth >= maxDepth) {
							bakeDepth++;
							break;
						}

						newContainer = container;
						if (!isRoot) {
							newContainer = Create();
							SafeAddChild(container, newContainer);
						}

						newContainer.type = Type.Array;
						Parse(inputString, ref offset, endOffset, newContainer, maxDepth, storeExcessLevels, depth + 1, false);

						break;
					case '}':
						if (!ParseObjectEnd(inputString, offset, openQuote, container, startOffset, lastValidOffset, maxDepth, storeExcessLevels, depth, ref bakeDepth))
							return;

						break;
					case ']':
						if (!ParseArrayEnd(inputString, offset, openQuote, container, startOffset, lastValidOffset, maxDepth, storeExcessLevels, depth, ref bakeDepth))
							return;

						break;
					case '"':
						ParseQuote(ref openQuote, offset, ref quoteStart, ref quoteEnd);
						break;
					case ':':
						if (!ParseColon(inputString, openQuote, container, ref startOffset, offset, quoteStart, quoteEnd, bakeDepth))
							return;

						break;
					case ',':
						if (!ParseComma(inputString, openQuote, container, ref startOffset, offset, lastValidOffset, bakeDepth))
							return;

						break;
				}

				lastValidOffset = offset - 1;
			}
		}

		static IEnumerable<ParseResult> ParseAsync(string inputString, int offset, int endOffset, JSONObject container,
			int maxDepth, bool storeExcessLevels, int depth = 0, bool isRoot = true) {
			if (!BeginParse(inputString, offset, ref endOffset, container, maxDepth, storeExcessLevels))
				yield break;

			var startOffset = offset;
			var quoteStart = 0;
			var quoteEnd = 0;
			var lastValidOffset = offset;
			var openQuote = false;
			var bakeDepth = 0;
			while (offset <= endOffset) {
				if (PrintWatch.Elapsed.TotalSeconds > MaxFrameTime) {
					PrintWatch.Reset();
					yield return new ParseResult(container, offset, true);
					PrintWatch.Start();
				}

				var currentCharacter = inputString[offset++];
				if (Array.IndexOf(Whitespace, currentCharacter) > -1)
					continue;

				JSONObject newContainer;
				switch (currentCharacter) {
					case '\\':
						offset++;
						break;
					case '{':
						if (openQuote)
							break;

						if (maxDepth >= 0 && depth >= maxDepth) {
							bakeDepth++;
							break;
						}

						newContainer = container;
						if (!isRoot) {
							newContainer = Create();
							SafeAddChild(container, newContainer);
						}

						newContainer.type = Type.Object;
						foreach (var e in ParseAsync(inputString, offset, endOffset, newContainer, maxDepth, storeExcessLevels, depth + 1, false)) {
							if (e.pause)
								yield return e;

							offset = e.offset;
						}

						break;
					case '[':
						if (openQuote)
							break;

						if (maxDepth >= 0 && depth >= maxDepth) {
							bakeDepth++;
							break;
						}

						newContainer = container;
						if (!isRoot) {
							newContainer = Create();
							SafeAddChild(container, newContainer);
						}

						newContainer.type = Type.Array;
						foreach (var e in ParseAsync(inputString, offset, endOffset, newContainer, maxDepth, storeExcessLevels, depth + 1, false)) {
							if (e.pause)
								yield return e;

							offset = e.offset;
						}

						break;
					case '}':
						if (!ParseObjectEnd(inputString, offset, openQuote, container, startOffset, lastValidOffset, maxDepth, storeExcessLevels, depth, ref bakeDepth)) {
							yield return new ParseResult(container, offset, false);
							yield break;
						}

						break;
					case ']':
						if (!ParseArrayEnd(inputString, offset, openQuote, container, startOffset, lastValidOffset, maxDepth, storeExcessLevels, depth, ref bakeDepth)) {
							yield return new ParseResult(container, offset, false);
							yield break;
						}

						break;
					case '"':
						ParseQuote(ref openQuote, offset, ref quoteStart, ref quoteEnd);
						break;
					case ':':
						if (!ParseColon(inputString, openQuote, container, ref startOffset, offset, quoteStart, quoteEnd, bakeDepth)) {
							yield return new ParseResult(container, offset, false);
							yield break;
						}

						break;
					case ',':
						if (!ParseComma(inputString, openQuote, container, ref startOffset, offset, lastValidOffset, bakeDepth)) {
							yield return new ParseResult(container, offset, false);
							yield break;
						}

						break;
				}

				lastValidOffset = offset - 1;
			}

			yield return new ParseResult(container, offset, false);
		}

		static void SafeAddChild(JSONObject container, JSONObject child) {
			var list = container.list;
			if (list == null) {
				list = CreateJSONObjectList();
				container.list = list;
			}

			list.Add(child);
		}

		void ParseValue(string inputString, int startOffset, int lastValidOffset) {
			var firstCharacter = inputString[startOffset];
			do {
				if (Array.IndexOf(Whitespace, firstCharacter) > -1) {
					firstCharacter = inputString[++startOffset];
					continue;
				}

				break;
			} while (true);

			// Use character comparison instead of string compare as performance optimization
			switch (firstCharacter)
			{
				case '"':
					type = Type.String;

					// Trim quotes from string values
					stringValue = UnEscapeString(inputString.Substring(startOffset + 1, lastValidOffset - startOffset - 1));
					return;
				case 't':
					type = Type.Bool;
					boolValue = true;
					return;
				case 'f':
					type = Type.Bool;
					boolValue = false;
					return;
				case 'n':
					type = Type.Null;
					return;
				case 'I':
					type = Type.Number;

#if JSONOBJECT_USE_FLOAT
					floatValue = float.PositiveInfinity;
#else
					doubleValue = double.PositiveInfinity;
#endif

					return;
				case 'N':
					type = Type.Number;

#if JSONOBJECT_USE_FLOAT
					floatValue = float.NaN;
#else
					doubleValue = double.NaN;
#endif
					return;
				case '-':
					if (inputString[startOffset + 1] == 'I') {
						type = Type.Number;
#if JSONOBJECT_USE_FLOAT
						floatValue = float.NegativeInfinity;
#else
						doubleValue = double.NegativeInfinity;
#endif
						return;
					}

					break;
			}

			var numericString = inputString.Substring(startOffset, lastValidOffset - startOffset + 1);
			try {
				if (numericString.Contains(".")) {
#if JSONOBJECT_USE_FLOAT
					floatValue = Convert.ToSingle(numericString, CultureInfo.InvariantCulture);
#else
					doubleValue = Convert.ToDouble(numericString, CultureInfo.InvariantCulture);
#endif
				} else {
					longValue = Convert.ToInt64(numericString, CultureInfo.InvariantCulture);
					isInteger = true;
#if JSONOBJECT_USE_FLOAT
					floatValue = longValue;
#else
					doubleValue = longValue;
#endif
				}

				type = Type.Number;
			} catch (OverflowException) {
				type = Type.Number;
#if JSONOBJECT_USE_FLOAT
				floatValue = numericString.StartsWith("-") ? float.NegativeInfinity : float.PositiveInfinity;
#else
				doubleValue = numericString.StartsWith("-") ? double.NegativeInfinity : double.PositiveInfinity;
#endif
			} catch (FormatException) {
				type = Type.Null;
#if USING_UNITY
				Debug.LogWarning
#else
				Debug.WriteLine
#endif
					(string.Format("Improper JSON formatting:{0}", numericString));
			}
		}

		static bool ParseObjectEnd(string inputString, int offset, bool openQuote, JSONObject container, int startOffset,
			int lastValidOffset, int maxDepth, bool storeExcessLevels, int depth, ref int bakeDepth) {
			if (openQuote)
				return true;

			if (container == null) {
				Debug.LogError("Parsing error: encountered `}` with no container object");
				return false;
			}

			if (maxDepth >= 0 && depth >= maxDepth) {
				bakeDepth--;
				if (bakeDepth == 0) {
					SafeAddChild(container,
						storeExcessLevels
							? CreateBakedObject(inputString.Substring(startOffset, offset - startOffset))
							: nullObject);
				}

				if (bakeDepth >= 0)
					return true;
			}

			ParseFinalObjectIfNeeded(inputString, container, startOffset, lastValidOffset);
			return false;
		}

		static bool ParseArrayEnd(string inputString, int offset, bool openQuote, JSONObject container,
			int startOffset, int lastValidOffset, int maxDepth, bool storeExcessLevels, int depth, ref int bakeDepth) {
			if (openQuote)
				return true;

			if (container == null) {
				Debug.LogError("Parsing error: encountered `]` with no container object");
				return false;
			}

			if (maxDepth >= 0 && depth >= maxDepth) {
				bakeDepth--;
				if (bakeDepth == 0) {
					SafeAddChild(container,
						storeExcessLevels
							? CreateBakedObject(inputString.Substring(startOffset, offset - startOffset))
							: nullObject);
				}

				if (bakeDepth >= 0)
					return true;
			}

			ParseFinalObjectIfNeeded(inputString, container, startOffset, lastValidOffset);
			return false;
		}

		static void ParseQuote(ref bool openQuote, int offset, ref int quoteStart, ref int quoteEnd) {
			if (openQuote) {
				quoteEnd = offset - 1;
				openQuote = false;
			} else {
				quoteStart = offset;
				openQuote = true;
			}
		}

		static bool ParseColon(string inputString, bool openQuote, JSONObject container,
			ref int startOffset,int offset, int quoteStart, int quoteEnd, int bakeDepth) {
			if (openQuote || bakeDepth > 0)
				return true;

			if (container == null) {
				Debug.LogError("Parsing error: encountered `:` with no container object");
				return false;
			}

			var keys = container.keys;
			if (keys == null) {
				keys = CreateStringList();
				container.keys = keys;
			}

			container.keys.Add(inputString.Substring(quoteStart, quoteEnd - quoteStart));
			startOffset = offset;

			return true;
		}

		static bool ParseComma(string inputString, bool openQuote, JSONObject container,
			ref int startOffset, int offset, int lastValidOffset, int bakeDepth) {
			if (openQuote || bakeDepth > 0)
				return true;

			if (container == null) {
				Debug.LogError("Parsing error: encountered `,` with no container object");
				return false;
			}

			ParseFinalObjectIfNeeded(inputString, container, startOffset, lastValidOffset);

			startOffset = offset;
			return true;
		}

		static void ParseFinalObjectIfNeeded(string inputString, JSONObject container, int startOffset, int lastValidOffset) {
			if (IsClosingCharacter(inputString[lastValidOffset]))
				return;

			var child = Create();
			child.ParseValue(inputString, startOffset, lastValidOffset);
			SafeAddChild(container, child);
		}

		static bool IsClosingCharacter(char character) {
			switch (character) {
				case '}':
				case ']':
					return true;
			}

			return false;
		}

		public bool isNumber {
			get { return type == Type.Number; }
		}

		public bool isNull {
			get { return type == Type.Null; }
		}

		public bool isString {
			get { return type == Type.String; }
		}

		public bool isBool {
			get { return type == Type.Bool; }
		}

		public bool isArray {
			get { return type == Type.Array; }
		}

		public bool isObject {
			get { return type == Type.Object; }
		}

		public bool isBaked {
			get { return type == Type.Baked; }
		}

		public void Add(bool value) {
			Add(Create(value));
		}

		public void Add(float value) {
			Add(Create(value));
		}

		public void Add(double value) {
			Add(Create(value));
		}

		public void Add(long value) {
			Add(Create(value));
		}

		public void Add(int value) {
			Add(Create(value));
		}

		public void Add(string value) {
			Add(CreateStringObject(value));
		}

		public void Add(AddJSONContents content) {
			Add(Create(content));
		}

		public void Add(JSONObject jsonObject) {
			if (jsonObject == null)
				return;

			// Convert to array to support list
			type = Type.Array;
			if (list == null)
				list = CreateJSONObjectList();

			list.Add(jsonObject);
		}

		public void AddField(string name, bool value) {
			AddField(name, Create(value));
		}

		public void AddField(string name, float value) {
			AddField(name, Create(value));
		}

		public void AddField(string name, double value) {
			AddField(name, Create(value));
		}

		public void AddField(string name, int value) {
			AddField(name, Create(value));
		}

		public void AddField(string name, long value) {
			AddField(name, Create(value));
		}

		public void AddField(string name, AddJSONContents content) {
			AddField(name, Create(content));
		}

		public void AddField(string name, string value) {
			AddField(name, CreateStringObject(value));
		}

		public void AddField(string name, JSONObject jsonObject) {
			if (jsonObject == null)
				return;

			// Convert to object if needed to support fields
			type = Type.Object;
			if (list == null)
				list = CreateJSONObjectList();

			if (keys == null)
				keys = CreateStringList();

			while (keys.Count < list.Count) {
				keys.Add(keys.Count.ToString(CultureInfo.InvariantCulture));
			}

			keys.Add(name);
			list.Add(jsonObject);
		}

		public void SetField(string name, string value) {
			SetField(name, CreateStringObject(value));
		}

		public void SetField(string name, bool value) {
			SetField(name, Create(value));
		}

		public void SetField(string name, float value) {
			SetField(name, Create(value));
		}

		public void SetField(string name, double value) {
			SetField(name, Create(value));
		}

		public void SetField(string name, long value) {
			SetField(name, Create(value));
		}

		public void SetField(string name, int value) {
			SetField(name, Create(value));
		}

		public void SetField(string name, JSONObject jsonObject) {
			if (HasField(name)) {
				list.Remove(this[name]);
				keys.Remove(name);
			}

			AddField(name, jsonObject);
		}

		public void RemoveField(string name) {
			if (keys == null || list == null)
				return;

			if (keys.IndexOf(name) > -1) {
				list.RemoveAt(keys.IndexOf(name));
				keys.Remove(name);
			}
		}

		public bool GetField(out bool field, string name, bool fallback) {
			field = fallback;
			return GetField(ref field, name);
		}

		public bool GetField(ref bool field, string name, FieldNotFound fail = null) {
			if (type == Type.Object && keys != null && list != null) {
				var index = keys.IndexOf(name);
				if (index >= 0) {
					field = list[index].boolValue;
					return true;
				}
			}

			if (fail != null) fail.Invoke(name);
			return false;
		}

		public bool GetField(out double field, string name, double fallback) {
			field = fallback;
			return GetField(ref field, name);
		}

		public bool GetField(ref double field, string name, FieldNotFound fail = null) {
			if (type == Type.Object && keys != null && list != null) {
				var index = keys.IndexOf(name);
				if (index >= 0) {
#if JSONOBJECT_USE_FLOAT
					field = list[index].floatValue;
#else
					field = list[index].doubleValue;
#endif
					return true;
				}
			}

			if (fail != null) fail.Invoke(name);
			return false;
		}

		public bool GetField(out float field, string name, float fallback) {
			field = fallback;
			return GetField(ref field, name);
		}

		public bool GetField(ref float field, string name, FieldNotFound fail = null) {
			if (type == Type.Object && keys != null && list != null) {
				var index = keys.IndexOf(name);
				if (index >= 0) {
#if JSONOBJECT_USE_FLOAT
					field = list[index].floatValue;
#else
					field = (float) list[index].doubleValue;
#endif
					return true;
				}
			}

			if (fail != null) fail.Invoke(name);
			return false;
		}

		public bool GetField(out int field, string name, int fallback) {
			field = fallback;
			return GetField(ref field, name);
		}

		public bool GetField(ref int field, string name, FieldNotFound fail = null) {
			if (type == Type.Object && keys != null && list != null) {
				var index = keys.IndexOf(name);
				if (index >= 0) {
					field = (int) list[index].longValue;
					return true;
				}
			}

			if (fail != null) fail.Invoke(name);
			return false;
		}

		public bool GetField(out long field, string name, long fallback) {
			field = fallback;
			return GetField(ref field, name);
		}

		public bool GetField(ref long field, string name, FieldNotFound fail = null) {
			if (type == Type.Object && keys != null && list != null) {
				var index = keys.IndexOf(name);
				if (index >= 0) {
					field = list[index].longValue;
					return true;
				}
			}

			if (fail != null) fail.Invoke(name);
			return false;
		}

		public bool GetField(out uint field, string name, uint fallback) {
			field = fallback;
			return GetField(ref field, name);
		}

		public bool GetField(ref uint field, string name, FieldNotFound fail = null) {
			if (type == Type.Object && keys != null && list != null) {
				var index = keys.IndexOf(name);
				if (index >= 0) {
					field = (uint) list[index].longValue;
					return true;
				}
			}

			if (fail != null) fail.Invoke(name);
			return false;
		}

		public bool GetField(out string field, string name, string fallback) {
			field = fallback;
			return GetField(ref field, name);
		}

		public bool GetField(ref string field, string name, FieldNotFound fail = null) {
			if (type == Type.Object && keys != null && list != null) {
				var index = keys.IndexOf(name);
				if (index >= 0) {
					field = list[index].stringValue;
					return true;
				}
			}

			if (fail != null) fail.Invoke(name);
			return false;
		}

		public void GetField(string name, GetFieldResponse response, FieldNotFound fail = null) {
			if (response != null && type == Type.Object && keys != null && list != null) {
				var index = keys.IndexOf(name);
				if (index >= 0) {
					response.Invoke(list[index]);
					return;
				}
			}

			if (fail != null)
				fail.Invoke(name);
		}

		public JSONObject GetField(string name) {
			if (type == Type.Object && keys != null && list != null) {
				for (var index = 0; index < keys.Count; index++)
					if (keys[index] == name)
						return list[index];
			}

			return null;
		}

		public bool HasFields(string[] names) {
			if (type != Type.Object || keys == null || list == null)
				return false;

			foreach (var name in names)
				if (!keys.Contains(name))
					return false;

			return true;
		}

		public bool HasField(string name) {
			if (type != Type.Object || keys == null || list == null)
				return false;

			if (keys == null || list == null)
				return false;

			foreach (var fieldName in keys)
				if (fieldName == name)
					return true;

			return false;
		}

		public void Clear() {
#if JSONOBJECT_POOLING
			if (list != null) {
				lock (JSONObjectListPool) {
					if (poolingEnabled && JSONObjectListPool.Count < MaxPoolSize) {
						list.Clear();
						JSONObjectListPool.Enqueue(list);
					}
				}
			}

			if (keys != null) {
				lock (StringListPool) {
					if (poolingEnabled && StringListPool.Count < MaxPoolSize) {
						keys.Clear();
						StringListPool.Enqueue(keys);
					}
				}
			}
#endif

			type = Type.Null;
			list = null;
			keys = null;
			stringValue = null;
			longValue = 0;
			boolValue = false;
			isInteger = false;
#if JSONOBJECT_USE_FLOAT
			floatValue = 0;
#else
			doubleValue = 0;
#endif
		}

		/// <summary>
		/// Copy a JSONObject. This could be more efficient
		/// </summary>
		/// <returns></returns>
		public JSONObject Copy() {
			return Create(Print());
		}

		/*
		 * The Merge function is experimental. Use at your own risk.
		 */
		public void Merge(JSONObject jsonObject) {
			MergeRecur(this, jsonObject);
		}

		/// <summary>
		/// Merge object right into left recursively
		/// </summary>
		/// <param name="left">The left (base) object</param>
		/// <param name="right">The right (new) object</param>
		static void MergeRecur(JSONObject left, JSONObject right) {
			if (left.type == Type.Null) {
				left.Absorb(right);
			} else if (left.type == Type.Object && right.type == Type.Object && right.list != null && right.keys != null) {
				for (var i = 0; i < right.list.Count; i++) {
					var key = right.keys[i];
					if (right[i].isContainer) {
						if (left.HasField(key))
							MergeRecur(left[key], right[i]);
						else
							left.AddField(key, right[i]);
					} else {
						if (left.HasField(key))
							left.SetField(key, right[i]);
						else
							left.AddField(key, right[i]);
					}
				}
			} else if (left.type == Type.Array && right.type == Type.Array && right.list != null) {
				if (right.count > left.count) {
#if USING_UNITY
					Debug.LogError
#else
					Debug.WriteLine
#endif
						("Cannot merge arrays when right object has more elements");
					return;
				}

				for (var i = 0; i < right.list.Count; i++) {
					if (left[i].type == right[i].type) {
						//Only overwrite with the same type
						if (left[i].isContainer)
							MergeRecur(left[i], right[i]);
						else {
							left[i] = right[i];
						}
					}
				}
			}
		}

		public void Bake() {
			if (type == Type.Baked)
				return;

			stringValue = Print();
			type = Type.Baked;
		}

		public IEnumerable BakeAsync() {
			if (type == Type.Baked)
				yield break;

			
			var builder = new StringBuilder();
			using (var enumerator = PrintAsync(builder).GetEnumerator()) {
				while (enumerator.MoveNext()) {
					if (enumerator.Current)
						yield return null;
				}

				stringValue = builder.ToString();
				type = Type.Baked;
			}
		}

		public string Print(bool pretty = false) {
			var builder = new StringBuilder();
			Print(builder, pretty);
			return builder.ToString();
		}

		public void Print(StringBuilder builder, bool pretty = false) {
			Stringify(0, builder, pretty);
		}

		static string EscapeString(string input) {
			var escaped = input.Replace("\b", "\\b");
			escaped = escaped.Replace("\f", "\\f");
			escaped = escaped.Replace("\n", "\\n");
			escaped = escaped.Replace("\r", "\\r");
			escaped = escaped.Replace("\t", "\\t");
			escaped = escaped.Replace("\"", "\\\"");
			return escaped;
		}

		static string UnEscapeString(string input) {
			var unescaped = input.Replace("\\\"", "\"");
			unescaped = unescaped.Replace("\\b", "\b");
			unescaped = unescaped.Replace("\\f", "\f");
			unescaped = unescaped.Replace("\\n", "\n");
			unescaped = unescaped.Replace("\\r", "\r");
			unescaped = unescaped.Replace("\\t", "\t");
			return unescaped;
		}

		public IEnumerable<string> PrintAsync(bool pretty = false) {
			var builder = new StringBuilder();
			foreach (var pause in PrintAsync(builder, pretty)) {
				if (pause)
					yield return null;
			}

			yield return builder.ToString();
		}

		public IEnumerable<bool> PrintAsync(StringBuilder builder, bool pretty = false) {
			PrintWatch.Reset();
			PrintWatch.Start();
			using (var enumerator = StringifyAsync(0, builder, pretty).GetEnumerator()) {
				while (enumerator.MoveNext()) {
					if (enumerator.Current)
						yield return true;
				}
			}
		}

		/// <summary>
		/// Convert the JSONObject into a string
		/// </summary>
		/// <param name="depth">How many containers deep this run has reached</param>
		/// <param name="builder">The StringBuilder used to build the string</param>
		/// <param name="pretty">Whether this string should be "pretty" and include whitespace for readability</param>
		/// <returns>An enumerator for this function</returns>
		IEnumerable<bool> StringifyAsync(int depth, StringBuilder builder, bool pretty = false) {
			if (PrintWatch.Elapsed.TotalSeconds > MaxFrameTime) {
				PrintWatch.Reset();
				yield return true;
				PrintWatch.Start();
			}

			switch (type) {
				case Type.Baked:
					builder.Append(stringValue);
					break;
				case Type.String:
					StringifyString(builder);
					break;
				case Type.Number:
					StringifyNumber(builder);
					break;
				case Type.Object:
					var fieldCount = count;
					if (fieldCount <= 0) {
						StringifyEmptyObject(builder);
						break;
					}

					depth++;

					BeginStringifyObjectContainer(builder, pretty);
					for (var index = 0; index < fieldCount; index++) {
						var jsonObject = list[index];
						if (jsonObject == null)
							continue;

						var key = keys[index];
						BeginStringifyObjectField(builder, pretty, depth, key);
						foreach (var pause in jsonObject.StringifyAsync(depth, builder, pretty)) {
							if (pause)
								yield return true;
						}

						EndStringifyObjectField(builder, pretty);
					}

					EndStringifyObjectContainer(builder, pretty, depth);
					break;
				case Type.Array:
					var arraySize = count;
					if (arraySize <= 0) {
						StringifyEmptyArray(builder);
						break;
					}

					BeginStringifyArrayContainer(builder, pretty);
					for (var index = 0; index < arraySize; index++) {
						var jsonObject = list[index];
						if (jsonObject == null)
							continue;

						BeginStringifyArrayElement(builder, pretty, depth);
						foreach (var pause in list[index].StringifyAsync(depth, builder, pretty)) {
							if (pause)
								yield return true;
						}

						EndStringifyArrayElement(builder, pretty);
					}

					EndStringifyArrayContainer(builder, pretty, depth);
					break;
				case Type.Bool:
					StringifyBool(builder);
					break;
				case Type.Null:
					StringifyNull(builder);
					break;
			}
		}

		/// <summary>
		/// Convert the JSONObject into a string
		/// </summary>
		/// <param name="depth">How many containers deep this run has reached</param>
		/// <param name="builder">The StringBuilder used to build the string</param>
		/// <param name="pretty">Whether this string should be "pretty" and include whitespace for readability</param>
		void Stringify(int depth, StringBuilder builder, bool pretty = false) {
			depth++;
			switch (type) {
				case Type.Baked:
					builder.Append(stringValue);
					break;
				case Type.String:
					StringifyString(builder);
					break;
				case Type.Number:
					StringifyNumber(builder);
					break;
				case Type.Object:
					var fieldCount = count;
					if (fieldCount <= 0) {
						StringifyEmptyObject(builder);
						break;
					}

					BeginStringifyObjectContainer(builder, pretty);
					for (var index = 0; index < fieldCount; index++) {
						var jsonObject = list[index];
						if (jsonObject == null)
							continue;

						if (keys == null || index >= keys.Count)
							break;

						var key = keys[index];
						BeginStringifyObjectField(builder, pretty, depth, key);
						jsonObject.Stringify(depth, builder, pretty);
						EndStringifyObjectField(builder, pretty);
					}

					EndStringifyObjectContainer(builder, pretty, depth);
					break;
				case Type.Array:
					if (count <= 0) {
						StringifyEmptyArray(builder);
						break;
					}

					BeginStringifyArrayContainer(builder, pretty);
					foreach (var jsonObject in list) {
						if (jsonObject == null)
							continue;

						BeginStringifyArrayElement(builder, pretty, depth);
						jsonObject.Stringify(depth, builder, pretty);
						EndStringifyArrayElement(builder, pretty);
					}

					EndStringifyArrayContainer(builder, pretty, depth);
					break;
				case Type.Bool:
					StringifyBool(builder);
					break;
				case Type.Null:
					StringifyNull(builder);
					break;
			}
		}

		void StringifyString(StringBuilder builder)
		{
			builder.AppendFormat("\"{0}\"", EscapeString(stringValue));
		}

		void BeginStringifyObjectContainer(StringBuilder builder, bool pretty) {
			builder.Append("{");

#if !JSONOBJECT_DISABLE_PRETTY_PRINT
			if (pretty)
				builder.Append(Newline);
#endif
		}

		static void StringifyEmptyObject(StringBuilder builder) {
			builder.Append("{}");
		}

		void BeginStringifyObjectField(StringBuilder builder, bool pretty, int depth, string key) {
#if !JSONOBJECT_DISABLE_PRETTY_PRINT
			if (pretty)
				for (var j = 0; j < depth; j++)
					builder.Append(Tab); //for a bit more readability
#endif

			builder.AppendFormat("\"{0}\":", key);
		}

		void EndStringifyObjectField(StringBuilder builder, bool pretty) {
			builder.Append(",");
#if !JSONOBJECT_DISABLE_PRETTY_PRINT
			if (pretty)
				builder.Append(Newline);
#endif
		}

		void EndStringifyObjectContainer(StringBuilder builder, bool pretty, int depth) {
#if !JSONOBJECT_DISABLE_PRETTY_PRINT
			if (pretty)
				builder.Length -= 3;
			else
#endif
				builder.Length--;

#if !JSONOBJECT_DISABLE_PRETTY_PRINT
			if (pretty && count > 0) {
				builder.Append(Newline);
				for (var j = 0; j < depth - 1; j++)
					builder.Append(Tab);
			}
#endif

			builder.Append("}");
		}

		static void StringifyEmptyArray(StringBuilder builder) {
			builder.Append("[]");
		}

		void BeginStringifyArrayContainer(StringBuilder builder, bool pretty) {
			builder.Append("[");
#if !JSONOBJECT_DISABLE_PRETTY_PRINT
			if (pretty)
				builder.Append(Newline);
#endif

		}

		void BeginStringifyArrayElement(StringBuilder builder, bool pretty, int depth) {
#if !JSONOBJECT_DISABLE_PRETTY_PRINT
			if (pretty)
				for (var j = 0; j < depth; j++)
					builder.Append(Tab); //for a bit more readability
#endif
		}

		void EndStringifyArrayElement(StringBuilder builder, bool pretty) {
			builder.Append(",");
#if !JSONOBJECT_DISABLE_PRETTY_PRINT
			if (pretty)
				builder.Append(Newline);
#endif
		}

		void EndStringifyArrayContainer(StringBuilder builder, bool pretty, int depth) {
#if !JSONOBJECT_DISABLE_PRETTY_PRINT
			if (pretty)
				builder.Length -= 3;
			else
#endif
				builder.Length--;

#if !JSONOBJECT_DISABLE_PRETTY_PRINT
			if (pretty && count > 0) {
				builder.Append(Newline);
				for (var j = 0; j < depth - 1; j++)
					builder.Append(Tab);
			}
#endif

			builder.Append("]");
		}

		void StringifyNumber(StringBuilder builder) {
			if (isInteger) {
				builder.Append(longValue.ToString(CultureInfo.InvariantCulture));
			} else {
#if JSONOBJECT_USE_FLOAT
				if (float.IsNegativeInfinity(floatValue))
					builder.Append(NegativeInfinity);
				else if (float.IsInfinity(floatValue))
					builder.Append(Infinity);
				else if (float.IsNaN(floatValue))
					builder.Append(NaN);
				else
					builder.Append(floatValue.ToString("R", CultureInfo.InvariantCulture));
#else
				if (double.IsNegativeInfinity(doubleValue))
					builder.Append(NegativeInfinity);
				else if (double.IsInfinity(doubleValue))
					builder.Append(Infinity);
				else if (double.IsNaN(doubleValue))
					builder.Append(NaN);
				else
					builder.Append(doubleValue.ToString("R", CultureInfo.InvariantCulture));
#endif
			}
		}

		void StringifyBool(StringBuilder builder) {
			builder.Append(boolValue ? True : False);
		}

		static void StringifyNull(StringBuilder builder) {
			builder.Append(Null);
		}

#if USING_UNITY
		public static implicit operator WWWForm(JSONObject jsonObject) {
			var form = new WWWForm();
			var count = jsonObject.count;
			var list = jsonObject.list;
			var keys = jsonObject.keys;
			var hasKeys = jsonObject.type == Type.Object && keys != null && keys.Count >= count;

			for (var i = 0; i < count; i++) {
				var key = hasKeys ? keys[i] : i.ToString(CultureInfo.InvariantCulture);
				var element = list[i];
				var val = element.ToString();
				if (element.type == Type.String)
					val = val.Replace("\"", "");

				form.AddField(key, val);
			}

			return form;
		}
#endif
		public JSONObject this[int index] {
			get {
				return count > index ? list[index] : null;
			}
			set {
				if (count > index)
					list[index] = value;
			}
		}

		public JSONObject this[string index] {
			get { return GetField(index); }
			set { SetField(index, value); }
		}

		public override string ToString() {
			return Print();
		}

		public string ToString(bool pretty) {
			return Print(pretty);
		}

		public Dictionary<string, string> ToDictionary() {
			if (type != Type.Object) {
#if USING_UNITY
				Debug.Log
#else
				Debug.WriteLine
#endif
					("Tried to turn non-Object JSONObject into a dictionary");

				return null;
			}

			var result = new Dictionary<string, string>();
			var listCount = count;
			if (keys == null || keys.Count != listCount)
				return result;

			for (var index = 0; index < listCount; index++) {
				var element = list[index];
				switch (element.type) {
					case Type.String:
						result.Add(keys[index], element.stringValue);
						break;
					case Type.Number:
#if JSONOBJECT_USE_FLOAT
						result.Add(keys[index], element.floatValue.ToString(CultureInfo.InvariantCulture));
#else
						result.Add(keys[index], element.doubleValue.ToString(CultureInfo.InvariantCulture));
#endif

						break;
					case Type.Bool:
						result.Add(keys[index], element.boolValue.ToString(CultureInfo.InvariantCulture));
						break;
					default:
#if USING_UNITY
						Debug.LogWarning
#else
						Debug.WriteLine
#endif
							(string.Format("Omitting object: {0} in dictionary conversion", keys[index]));
						break;
				}
			}

			return result;
		}

		public static implicit operator bool(JSONObject jsonObject) {
			return jsonObject != null;
		}

#if JSONOBJECT_POOLING
		public static void ClearPool() {
			poolingEnabled = false;
			poolingEnabled = true;
			lock (JSONObjectPool) {
				JSONObjectPool.Clear();
			}
		}

		~JSONObject() {
			lock (JSONObjectPool) {
				if (!poolingEnabled || isPooled || JSONObjectPool.Count >= MaxPoolSize)
					return;

				Clear();
				isPooled = true;
				JSONObjectPool.Enqueue(this);
				GC.ReRegisterForFinalize(this);
			}
		}

		public static int GetJSONObjectPoolSize() {
			return JSONObjectPool.Count;
		}

		public static int GetJSONObjectListPoolSize() {
			return JSONObjectListPool.Count;
		}

		public static int GetStringListPoolSize() {
			return StringListPool.Count;
		}
#endif

		public static List<JSONObject> CreateJSONObjectList() {
#if JSONOBJECT_POOLING
			lock (JSONObjectListPool) {
				if (JSONObjectListPool.Count > 0)
					return JSONObjectListPool.Dequeue();
			}
#endif

			return new List<JSONObject>();
		}

		public static List<string> CreateStringList() {
#if JSONOBJECT_POOLING
			lock (StringListPool) {
				if (StringListPool.Count > 0)
					return StringListPool.Dequeue();
			}
#endif

			return new List<string>();
		}

		IEnumerator IEnumerable.GetEnumerator() {
			return GetEnumerator();
		}

		public JSONObjectEnumerator GetEnumerator() {
			return new JSONObjectEnumerator(this);
		}
	}

	public class JSONObjectEnumerator : IEnumerator {
		public JSONObject target;

		// Enumerators are positioned before the first element until the first MoveNext() call.
		int position = -1;

		public JSONObjectEnumerator(JSONObject jsonObject) {
			if (!jsonObject.isContainer)
				throw new InvalidOperationException("JSONObject must be an array or object to provide an iterator");

			target = jsonObject;
		}

		public bool MoveNext() {
			position++;
			return position < target.count;
		}

		public void Reset() {
			position = -1;
		}

		object IEnumerator.Current {
			get { return Current; }
		}

		// ReSharper disable once InconsistentNaming
		public JSONObject Current {
			get {
				return target[position];
			}
		}
	}
}