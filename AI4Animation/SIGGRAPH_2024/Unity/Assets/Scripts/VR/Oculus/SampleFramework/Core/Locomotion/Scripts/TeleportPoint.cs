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


using UnityEngine;
using System.Collections;

public class TeleportPoint : MonoBehaviour {

    public float dimmingSpeed = 1;
    public float fullIntensity = 1;
    public float lowIntensity = 0.5f;

    public Transform destTransform;

    private float lastLookAtTime = 0;



	// Use this for initialization
	void Start () {

	}

    public Transform GetDestTransform()
    {
        return destTransform;
    }




	// Update is called once per frame
	void Update () {
        float intensity = Mathf.SmoothStep(fullIntensity, lowIntensity, (Time.time - lastLookAtTime) * dimmingSpeed);
        GetComponent<MeshRenderer>().material.SetFloat("_Intensity", intensity);
	}

    public void OnLookAt()
    {
        lastLookAtTime = Time.time;
    }
}
