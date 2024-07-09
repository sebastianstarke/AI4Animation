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
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

/// <summary>
/// Plays tactile effects on a tracked VR controller.
/// </summary>
public static class OVRHaptics
{
	public readonly static OVRHapticsChannel[] Channels;
	public readonly static OVRHapticsChannel LeftChannel;
	public readonly static OVRHapticsChannel RightChannel;

	private readonly static OVRHapticsOutput[] m_outputs;

	static OVRHaptics()
	{
		Config.Load();

		m_outputs = new OVRHapticsOutput[]
		{
			new OVRHapticsOutput((uint)OVRPlugin.Controller.LTouch),
			new OVRHapticsOutput((uint)OVRPlugin.Controller.RTouch),
		};

		Channels = new OVRHapticsChannel[]
		{
			LeftChannel = new OVRHapticsChannel(0),
			RightChannel = new OVRHapticsChannel(1),
		};
	}

	/// <summary>
	/// Determines the target format for haptics data on a specific device.
	/// </summary>
	public static class Config
	{
		public static int SampleRateHz { get; private set; }
		public static int SampleSizeInBytes { get; private set; }
		public static int MinimumSafeSamplesQueued { get; private set; }
		public static int MinimumBufferSamplesCount { get; private set; }
		public static int OptimalBufferSamplesCount { get; private set; }
		public static int MaximumBufferSamplesCount { get; private set; }

		static Config()
		{
			Load();
		}

		public static void Load()
		{
			OVRPlugin.HapticsDesc desc = OVRPlugin.GetControllerHapticsDesc((uint)OVRPlugin.Controller.RTouch);

			SampleRateHz = desc.SampleRateHz;
			SampleSizeInBytes = desc.SampleSizeInBytes;
			MinimumSafeSamplesQueued = desc.MinimumSafeSamplesQueued;
			MinimumBufferSamplesCount = desc.MinimumBufferSamplesCount;
			OptimalBufferSamplesCount = desc.OptimalBufferSamplesCount;
			MaximumBufferSamplesCount = desc.MaximumBufferSamplesCount;
		}
	}

	/// <summary>
	/// A track of haptics data that can be mixed or sequenced with another track.
	/// </summary>
	public class OVRHapticsChannel
	{
		private OVRHapticsOutput m_output;

		/// <summary>
		/// Constructs a channel targeting the specified output.
		/// </summary>
		public OVRHapticsChannel(uint outputIndex)
		{
			m_output = m_outputs[outputIndex];
		}

		/// <summary>
		/// Cancels any currently-playing clips and immediatly plays the specified clip instead.
		/// </summary>
		public void Preempt(OVRHapticsClip clip)
		{
			m_output.Preempt(clip);
		}

		/// <summary>
		/// Enqueues the specified clip to play after any currently-playing clips finish.
		/// </summary>
		public void Queue(OVRHapticsClip clip)
		{
			m_output.Queue(clip);
		}

		/// <summary>
		/// Adds the specified clip to play simultaneously to the currently-playing clip(s).
		/// </summary>
		public void Mix(OVRHapticsClip clip)
		{
			m_output.Mix(clip);
		}

		/// <summary>
		/// Cancels any currently-playing clips.
		/// </summary>
		public void Clear()
		{
			m_output.Clear();
		}
	}

	private class OVRHapticsOutput
	{
		private class ClipPlaybackTracker
		{
			public int ReadCount { get; set; }
			public OVRHapticsClip Clip { get; set; }

			public ClipPlaybackTracker(OVRHapticsClip clip)
			{
				Clip = clip;
			}
		}

		private bool m_lowLatencyMode = true;
		private bool m_paddingEnabled = true;
		private int m_prevSamplesQueued = 0;
		private float m_prevSamplesQueuedTime = 0;
		private int m_numPredictionHits = 0;
		private int m_numPredictionMisses = 0;
		private int m_numUnderruns = 0;
		private List<ClipPlaybackTracker> m_pendingClips = new List<ClipPlaybackTracker>();
		private uint m_controller = 0;
		private OVRNativeBuffer m_nativeBuffer = new OVRNativeBuffer(OVRHaptics.Config.MaximumBufferSamplesCount * OVRHaptics.Config.SampleSizeInBytes);
		private OVRHapticsClip m_paddingClip = new OVRHapticsClip();
		private int PrevSampleRateHz = -1;

		public OVRHapticsOutput(uint controller)
		{
#if UNITY_ANDROID
			m_paddingEnabled = false;
#endif
			m_controller = controller;
		}

		/// <summary>
		/// The system calls this each frame to update haptics playback.
		/// </summary>
		public void Process()
		{
			if (OVRHaptics.Config.SampleRateHz == 0)
			{
				if(PrevSampleRateHz != 0)
				{
					Debug.Log("Unable to process a controller whose SampleRateHz is 0 now.");
					PrevSampleRateHz = 0;
				}
				return;
			}
			PrevSampleRateHz = OVRHaptics.Config.SampleRateHz;

			// Resize the native buffer if the Config changes
			if (m_nativeBuffer.GetCapacity() != Config.MaximumBufferSamplesCount * Config.SampleSizeInBytes)
			{
				m_nativeBuffer.Reset(Config.MaximumBufferSamplesCount * Config.SampleSizeInBytes);
			}

			var hapticsState = OVRPlugin.GetControllerHapticsState(m_controller);

			float elapsedTime = Time.realtimeSinceStartup - m_prevSamplesQueuedTime;
			if (m_prevSamplesQueued > 0)
			{
				int expectedSamples = m_prevSamplesQueued - (int)(elapsedTime * OVRHaptics.Config.SampleRateHz + 0.5f);
				if (expectedSamples < 0)
					expectedSamples = 0;

				if ((hapticsState.SamplesQueued - expectedSamples) == 0)
					m_numPredictionHits++;
				else
					m_numPredictionMisses++;

				//Debug.Log(hapticsState.SamplesAvailable + "a " + hapticsState.SamplesQueued + "q " + expectedSamples + "e "
				//+ "Prediction Accuracy: " + m_numPredictionHits / (float)(m_numPredictionMisses + m_numPredictionHits));

				if ((expectedSamples > 0) && (hapticsState.SamplesQueued == 0))
				{
					m_numUnderruns++;
					//Debug.LogError("Samples Underrun (" + m_controller + " #" + m_numUnderruns + ") -"
					//        + " Expected: " + expectedSamples
					//        + " Actual: " + hapticsState.SamplesQueued);
				}

				m_prevSamplesQueued = hapticsState.SamplesQueued;
				m_prevSamplesQueuedTime = Time.realtimeSinceStartup;
			}

			int desiredSamplesCount = OVRHaptics.Config.OptimalBufferSamplesCount;
			if (m_lowLatencyMode)
			{
				float sampleRateMs = 1000.0f / (float)OVRHaptics.Config.SampleRateHz;
				float elapsedMs = elapsedTime * 1000.0f;
				int samplesNeededPerFrame = (int)Mathf.Ceil(elapsedMs / sampleRateMs);
				int lowLatencySamplesCount = OVRHaptics.Config.MinimumSafeSamplesQueued + samplesNeededPerFrame;

				if (lowLatencySamplesCount < desiredSamplesCount)
					desiredSamplesCount = lowLatencySamplesCount;
			}

			if (hapticsState.SamplesQueued > desiredSamplesCount)
				return;

			if (desiredSamplesCount > OVRHaptics.Config.MaximumBufferSamplesCount)
				desiredSamplesCount = OVRHaptics.Config.MaximumBufferSamplesCount;
			if (desiredSamplesCount > hapticsState.SamplesAvailable)
				desiredSamplesCount = hapticsState.SamplesAvailable;

			int acquiredSamplesCount = 0;
			int clipIndex = 0;
			while(acquiredSamplesCount < desiredSamplesCount && clipIndex < m_pendingClips.Count)
			{
				int numSamplesToCopy = desiredSamplesCount - acquiredSamplesCount;
				int remainingSamplesInClip = m_pendingClips[clipIndex].Clip.Count - m_pendingClips[clipIndex].ReadCount;
				if (numSamplesToCopy > remainingSamplesInClip)
					numSamplesToCopy = remainingSamplesInClip;

				if (numSamplesToCopy > 0)
				{
					int numBytes = numSamplesToCopy * OVRHaptics.Config.SampleSizeInBytes;
					int dstOffset = acquiredSamplesCount * OVRHaptics.Config.SampleSizeInBytes;
					int srcOffset = m_pendingClips[clipIndex].ReadCount * OVRHaptics.Config.SampleSizeInBytes;
					Marshal.Copy(m_pendingClips[clipIndex].Clip.Samples, srcOffset, m_nativeBuffer.GetPointer(dstOffset), numBytes);

					m_pendingClips[clipIndex].ReadCount += numSamplesToCopy;
					acquiredSamplesCount += numSamplesToCopy;
				}

				clipIndex++;
			}

			for (int i = m_pendingClips.Count - 1; i >= 0 && m_pendingClips.Count > 0; i--)
			{
				if (m_pendingClips[i].ReadCount >= m_pendingClips[i].Clip.Count)
					m_pendingClips.RemoveAt(i);
			}

			if (m_paddingEnabled)
			{
				int desiredPadding = desiredSamplesCount - (hapticsState.SamplesQueued + acquiredSamplesCount);
				if (desiredPadding < (OVRHaptics.Config.MinimumBufferSamplesCount - acquiredSamplesCount))
					desiredPadding = (OVRHaptics.Config.MinimumBufferSamplesCount - acquiredSamplesCount);
				if (desiredPadding > hapticsState.SamplesAvailable)
					desiredPadding = hapticsState.SamplesAvailable;

				if (desiredPadding > 0)
				{
					int numBytes = desiredPadding * OVRHaptics.Config.SampleSizeInBytes;
					int dstOffset = acquiredSamplesCount * OVRHaptics.Config.SampleSizeInBytes;
					int srcOffset = 0;
					Marshal.Copy(m_paddingClip.Samples, srcOffset, m_nativeBuffer.GetPointer(dstOffset), numBytes);

					acquiredSamplesCount += desiredPadding;
				}
			}

			if (acquiredSamplesCount > 0)
			{
				OVRPlugin.HapticsBuffer hapticsBuffer;
				hapticsBuffer.Samples = m_nativeBuffer.GetPointer();
				hapticsBuffer.SamplesCount = acquiredSamplesCount;

				OVRPlugin.SetControllerHaptics(m_controller, hapticsBuffer);

				hapticsState = OVRPlugin.GetControllerHapticsState(m_controller);
				m_prevSamplesQueued = hapticsState.SamplesQueued;
				m_prevSamplesQueuedTime = Time.realtimeSinceStartup;
			}
		}

		/// <summary>
		/// Immediately plays the specified clip without waiting for any currently-playing clip to finish.
		/// </summary>
		public void Preempt(OVRHapticsClip clip)
		{
			m_pendingClips.Clear();
			m_pendingClips.Add(new ClipPlaybackTracker(clip));
		}

		/// <summary>
		/// Enqueues the specified clip to play after any currently-playing clip finishes.
		/// </summary>
		public void Queue(OVRHapticsClip clip)
		{
			m_pendingClips.Add(new ClipPlaybackTracker(clip));
		}

		/// <summary>
		/// Adds the samples from the specified clip to the ones in the currently-playing clip(s).
		/// </summary>
		public void Mix(OVRHapticsClip clip)
		{
			int numClipsToMix = 0;
			int numSamplesToMix = 0;
			int numSamplesRemaining = clip.Count;

			while (numSamplesRemaining > 0 && numClipsToMix < m_pendingClips.Count)
			{
				int numSamplesRemainingInClip = m_pendingClips[numClipsToMix].Clip.Count - m_pendingClips[numClipsToMix].ReadCount;
				numSamplesRemaining -= numSamplesRemainingInClip;
				numSamplesToMix += numSamplesRemainingInClip;
				numClipsToMix++;
			}

			if (numSamplesRemaining > 0)
			{
				numSamplesToMix += numSamplesRemaining;
				numSamplesRemaining = 0;
			}

			if (numClipsToMix > 0)
			{
				OVRHapticsClip mixClip = new OVRHapticsClip(numSamplesToMix);

				OVRHapticsClip a = clip;
				int aReadCount = 0;

				for (int i = 0; i < numClipsToMix; i++)
				{
					OVRHapticsClip b = m_pendingClips[i].Clip;
					for(int bReadCount = m_pendingClips[i].ReadCount; bReadCount < b.Count; bReadCount++)
					{
						if (OVRHaptics.Config.SampleSizeInBytes == 1)
						{
							byte sample = 0; // TODO support multi-byte samples
							if ((aReadCount < a.Count) && (bReadCount < b.Count))
							{
								sample = (byte)(Mathf.Clamp(a.Samples[aReadCount] + b.Samples[bReadCount], 0, System.Byte.MaxValue)); // TODO support multi-byte samples
								aReadCount++;
							}
							else if (bReadCount < b.Count)
							{
								sample = b.Samples[bReadCount]; // TODO support multi-byte samples
							}

							mixClip.WriteSample(sample); // TODO support multi-byte samples
						}
					}
				}

				while (aReadCount < a.Count)
				{
					if (OVRHaptics.Config.SampleSizeInBytes == 1)
					{
						mixClip.WriteSample(a.Samples[aReadCount]); // TODO support multi-byte samples
					}
					aReadCount++;
				}

				m_pendingClips[0] = new ClipPlaybackTracker(mixClip);
				for (int i = 1; i < numClipsToMix; i++)
				{
					m_pendingClips.RemoveAt(1);
				}
			}
			else
			{
				m_pendingClips.Add(new ClipPlaybackTracker(clip));
			}
		}

		public void Clear()
		{
			m_pendingClips.Clear();
		}
	}

	/// <summary>
	/// The system calls this each frame to update haptics playback.
	/// </summary>
	public static void Process()
	{
		Config.Load();

		for (int i = 0; i < m_outputs.Length; i++)
		{
			m_outputs[i].Process();
		}
	}
}
