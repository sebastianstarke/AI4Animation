/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Licensed under the Oculus SDK License Agreement (the "License");
 * you may not use the Oculus SDK except in compliance with the License,
 * which is provided at the time of installation or download, or which
 * otherwise accompanies this software in either electronic or hard copy form.
 *
 * You may obtain a copy of the License at
 *
 * https://developer.oculus.com/licenses/oculussdk/
 *
 * Unless required by applicable law or agreed to in writing, the Oculus SDK
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Assets.OVR.Scripts
{
	public class Record
	{
		public int sortOrder;
		public string category;
		public string message;
		public Record(int order, string cat, string msg)
		{
			sortOrder = order;
			category = cat;
			message = msg;
		}
	}

	public class RangedRecord : Record
	{
		public float value;
		public float min;
		public float max;
		public RangedRecord(int order, string cat, string msg, float val, float minVal, float maxVal)
			: base(order, cat, msg)
		{
			value = val;
			min = minVal;
			max = maxVal;
		}
	}

	public delegate void FixMethodDelegate(UnityEngine.Object obj, bool isLastInSet, int selectedIndex);

	public class FixRecord : Record
	{
		public FixMethodDelegate fixMethod;
		public UnityEngine.Object targetObject;
		public string[] buttonNames;
		public bool editModeRequired;
		public bool complete;

		public FixRecord(int order, string cat, string msg, FixMethodDelegate fix, UnityEngine.Object target, bool editRequired, string[] buttons)
			: base(order, cat, msg)
		{
			buttonNames = buttons;
			fixMethod = fix;
			targetObject = target;
			editModeRequired = editRequired;
			complete = false;
		}
	}
}
