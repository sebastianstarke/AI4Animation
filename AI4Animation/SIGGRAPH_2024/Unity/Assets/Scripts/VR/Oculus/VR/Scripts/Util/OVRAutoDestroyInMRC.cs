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

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// If there is a game object under the main camera which should not be cloned under Mixed Reality Capture,
// attaching this component would auto destroy that after the MRC camera get cloned
public class OVRAutoDestroyInMRC : MonoBehaviour {

	// Use this for initialization
	void Start () {
		bool underMrcCamera = false;

		Transform p = transform.parent;
		while (p != null)
		{
			if (p.gameObject.name.StartsWith("OculusMRC_"))
			{
				underMrcCamera = true;
				break;
			}
			p = p.parent;
		}

		if (underMrcCamera)
		{
			Destroy(gameObject);
		}
	}
	
	// Update is called once per frame
	void Update () {
		
	}
}
