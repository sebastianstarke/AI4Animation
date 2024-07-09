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
using System;
using System.Collections;

/// <summary>
/// A PCM buffer of data for a haptics effect.
/// </summary>
public class OVRHapticsClip
{
	/// <summary>
	/// The current number of samples in the clip.
	/// </summary>
	public int Count { get; private set; }

	/// <summary>
	/// The maximum number of samples the clip can store.
	/// </summary>
	public int Capacity { get; private set; }

	/// <summary>
	/// The raw haptics data.
	/// </summary>
	public byte[] Samples { get; private set; }

	public OVRHapticsClip()
	{
		Capacity = OVRHaptics.Config.MaximumBufferSamplesCount;
		Samples = new byte[Capacity * OVRHaptics.Config.SampleSizeInBytes];
	}

	/// <summary>
	/// Creates a clip with the specified capacity.
	/// </summary>
	public OVRHapticsClip(int capacity)
	{
		Capacity = (capacity >= 0) ? capacity : 0;
		Samples = new byte[Capacity * OVRHaptics.Config.SampleSizeInBytes];
	}

	/// <summary>
	/// Creates a clip with the specified data.
	/// </summary>
	public OVRHapticsClip(byte[] samples, int samplesCount)
	{
		Samples = samples;
		Capacity = Samples.Length / OVRHaptics.Config.SampleSizeInBytes;
		Count = (samplesCount >= 0) ? samplesCount : 0;
	}

	/// <summary>
	/// Creates a clip by mixing the specified clips.
	/// </summary>
	public OVRHapticsClip(OVRHapticsClip a, OVRHapticsClip b)
	{
		int maxCount = a.Count;
		if (b.Count > maxCount)
			maxCount = b.Count;

		Capacity = maxCount;
		Samples = new byte[Capacity * OVRHaptics.Config.SampleSizeInBytes];

		for (int i = 0; i < a.Count || i < b.Count; i++)
		{
			if (OVRHaptics.Config.SampleSizeInBytes == 1)
			{
				byte sample = 0; // TODO support multi-byte samples
				if ((i < a.Count) && (i < b.Count))
					sample = (byte)(Mathf.Clamp(a.Samples[i] + b.Samples[i], 0, System.Byte.MaxValue)); // TODO support multi-byte samples
				else if (i < a.Count)
					sample = a.Samples[i]; // TODO support multi-byte samples
				else if (i < b.Count)
					sample = b.Samples[i]; // TODO support multi-byte samples

				WriteSample(sample); // TODO support multi-byte samples
			}
		}
	}

	/// <summary>
	/// Creates a haptics clip from the specified audio clip.
	/// </summary>
	public OVRHapticsClip(AudioClip audioClip, int channel = 0)
	{
		float[] audioData = new float[audioClip.samples * audioClip.channels];
		audioClip.GetData(audioData, 0);

		InitializeFromAudioFloatTrack(audioData, audioClip.frequency, audioClip.channels, channel);
	}

	/// <summary>
	/// Adds the specified sample to the end of the clip.
	/// </summary>
	public void WriteSample(byte sample) // TODO support multi-byte samples
	{
		if (Count >= Capacity)
		{
			//Debug.LogError("Attempted to write OVRHapticsClip sample out of range - Count:" + Count + " Capacity:" + Capacity);
			return;
		}

		if (OVRHaptics.Config.SampleSizeInBytes == 1)
		{
			Samples[Count * OVRHaptics.Config.SampleSizeInBytes] = sample; // TODO support multi-byte samples
		}

		Count++;
	}

	/// <summary>
	/// Clears the clip and resets its size to 0.
	/// </summary>
	public void Reset()
	{
		Count = 0;
	}

	private void InitializeFromAudioFloatTrack(float[] sourceData, double sourceFrequency, int sourceChannelCount, int sourceChannel)
	{
		//If SampleRateHz == 0, most likely, controller is powered off.
		//In this case, don't try to fill any sample data.
		if (OVRHaptics.Config.SampleRateHz == 0)
		{
			Debug.Log("Unable to initialize a controller whose SampleRateHz is 0 now.");
			return;
		}

		double stepSizePrecise = (sourceFrequency + 1e-6) / OVRHaptics.Config.SampleRateHz;

		if (stepSizePrecise < 1.0 || stepSizePrecise> Int32.MaxValue)
			return;

		int stepSize = (int)stepSizePrecise;
		double stepSizeError = stepSizePrecise - stepSize;
		double accumulatedStepSizeError = 0.0f;
		int length = sourceData.Length;

		Count = 0;
		Capacity = length / sourceChannelCount / stepSize + 1;
		Samples = new byte[Capacity * OVRHaptics.Config.SampleSizeInBytes];

		int i = sourceChannel % sourceChannelCount;
		while (i < length)
		{
			if (OVRHaptics.Config.SampleSizeInBytes == 1)
			{
				WriteSample((byte)(Mathf.Clamp01(Mathf.Abs(sourceData[i])) * System.Byte.MaxValue)); // TODO support multi-byte samples
			}
			i+= stepSize * sourceChannelCount;
			accumulatedStepSizeError += stepSizeError;
			if ((int)accumulatedStepSizeError > 0)
			{
				i+= (int)accumulatedStepSizeError * sourceChannelCount;
				accumulatedStepSizeError = accumulatedStepSizeError - (int)accumulatedStepSizeError;
			}
		}
	}
}
