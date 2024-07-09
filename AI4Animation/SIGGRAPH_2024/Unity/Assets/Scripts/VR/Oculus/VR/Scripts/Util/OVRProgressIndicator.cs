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
using UnityEngine.UI;
using System.Collections;

/// <summary>
/// Visualizes progress for operations such as loading.
/// </summary>
public class OVRProgressIndicator : MonoBehaviour
{
    public MeshRenderer progressImage;

    [Range(0, 1)]
    public float currentProgress = 0.7f;

    void Awake()
    {
        progressImage.sortingOrder = 150;
    }



    // Update is called once per frame
    void Update()
    {
        progressImage.sharedMaterial.SetFloat("_AlphaCutoff", 1-currentProgress);

    }
}
