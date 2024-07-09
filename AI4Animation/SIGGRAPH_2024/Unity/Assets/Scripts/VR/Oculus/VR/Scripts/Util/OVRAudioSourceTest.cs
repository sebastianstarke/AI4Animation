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

public class OVRAudioSourceTest : MonoBehaviour
{
	public float period = 2.0f;
	private float nextActionTime;

	// Start is called before the first frame update
	void Start()
	{
		Material templateMaterial = GetComponent<Renderer>().material;
		Material newMaterial = Instantiate<Material>(templateMaterial);
		newMaterial.color = Color.green;
		GetComponent<Renderer>().material = newMaterial;

		nextActionTime = Time.time + period;
	}

	// Update is called once per frame
	void Update()
	{
		if (Time.time > nextActionTime)
		{
			nextActionTime = Time.time + period;

			Material mat = GetComponent<Renderer>().material;
			if (mat.color == Color.green)
			{
				mat.color = Color.red;
			}
			else
			{
				mat.color = Color.green;
			}

			AudioSource audioSource = GetComponent<AudioSource>();
			if (audioSource == null)
			{
				Debug.LogError("Unable to find AudioSource");
			}
			else
			{
				audioSource.Play();
			}
		}
	}
}
