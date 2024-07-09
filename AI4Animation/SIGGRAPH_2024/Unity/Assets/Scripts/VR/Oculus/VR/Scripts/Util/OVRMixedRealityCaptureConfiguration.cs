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

#if UNITY_ANDROID && !UNITY_EDITOR
#define OVR_ANDROID_MRC
#endif

using System;
using UnityEngine;

public interface OVRMixedRealityCaptureConfiguration
{
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || UNITY_ANDROID
	bool enableMixedReality { get; set; }
	LayerMask extraHiddenLayers { get; set; }
	LayerMask extraVisibleLayers { get; set; }
	bool dynamicCullingMask { get; set; }
	OVRManager.CompositionMethod compositionMethod { get; set; }
	Color externalCompositionBackdropColorRift { get; set; }
	Color externalCompositionBackdropColorQuest { get; set; }
	OVRManager.CameraDevice capturingCameraDevice { get; set; }
	bool flipCameraFrameHorizontally { get; set; }
	bool flipCameraFrameVertically { get; set; }
	float handPoseStateLatency { get; set; }
	float sandwichCompositionRenderLatency { get; set; }
	int sandwichCompositionBufferedFrames { get; set; }
	Color chromaKeyColor { get; set; }
	float chromaKeySimilarity { get; set; }
	float chromaKeySmoothRange { get; set; }
	float chromaKeySpillRange { get; set; }
	bool useDynamicLighting { get; set; }
	OVRManager.DepthQuality depthQuality { get; set; }
	float dynamicLightingSmoothFactor { get; set; }
	float dynamicLightingDepthVariationClampingValue { get; set; }
	OVRManager.VirtualGreenScreenType virtualGreenScreenType { get; set; }
	float virtualGreenScreenTopY { get; set; }
	float virtualGreenScreenBottomY { get; set; }
	bool virtualGreenScreenApplyDepthCulling { get; set; }
	float virtualGreenScreenDepthTolerance { get; set; }
	OVRManager.MrcActivationMode mrcActivationMode { get; set; }
	OVRManager.InstantiateMrcCameraDelegate instantiateMixedRealityCameraGameObject { get; set; }
#endif
}

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || UNITY_ANDROID
public static class OVRMixedRealityCaptureConfigurationExtensions
{
	public static void ApplyTo(this OVRMixedRealityCaptureConfiguration dest, OVRMixedRealityCaptureConfiguration source) 
	{
		dest.ReadFrom(source);
	}

	public static void ReadFrom(this OVRMixedRealityCaptureConfiguration dest, OVRMixedRealityCaptureConfiguration source)
	{
		dest.enableMixedReality = source.enableMixedReality;
		dest.compositionMethod = source.compositionMethod;
		dest.extraHiddenLayers = source.extraHiddenLayers;
		dest.externalCompositionBackdropColorRift = source.externalCompositionBackdropColorRift;
		dest.externalCompositionBackdropColorQuest = source.externalCompositionBackdropColorQuest;
		dest.capturingCameraDevice = source.capturingCameraDevice;
		dest.flipCameraFrameHorizontally = source.flipCameraFrameHorizontally;
		dest.flipCameraFrameVertically = source.flipCameraFrameVertically;
		dest.handPoseStateLatency = source.handPoseStateLatency;
		dest.sandwichCompositionRenderLatency = source.sandwichCompositionRenderLatency;
		dest.sandwichCompositionBufferedFrames = source.sandwichCompositionBufferedFrames;
		dest.chromaKeyColor = source.chromaKeyColor;
		dest.chromaKeySimilarity = source.chromaKeySimilarity;
		dest.chromaKeySmoothRange = source.chromaKeySmoothRange;
		dest.chromaKeySpillRange = source.chromaKeySpillRange;
		dest.useDynamicLighting = source.useDynamicLighting;
		dest.depthQuality = source.depthQuality;
		dest.dynamicLightingSmoothFactor = source.dynamicLightingSmoothFactor;
		dest.dynamicLightingDepthVariationClampingValue = source.dynamicLightingDepthVariationClampingValue;
		dest.virtualGreenScreenType = source.virtualGreenScreenType;
		dest.virtualGreenScreenTopY = source.virtualGreenScreenTopY;
		dest.virtualGreenScreenBottomY = source.virtualGreenScreenBottomY;
		dest.virtualGreenScreenApplyDepthCulling = source.virtualGreenScreenApplyDepthCulling;
		dest.virtualGreenScreenDepthTolerance = source.virtualGreenScreenDepthTolerance;
		dest.mrcActivationMode = source.mrcActivationMode;
		dest.instantiateMixedRealityCameraGameObject = source.instantiateMixedRealityCameraGameObject;
	}

}
#endif
