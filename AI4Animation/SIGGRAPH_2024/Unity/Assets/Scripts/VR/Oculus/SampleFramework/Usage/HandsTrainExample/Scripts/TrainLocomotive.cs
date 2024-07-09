/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using System.Collections;
using UnityEngine;
using UnityEngine.Assertions;

namespace OculusSampleFramework
{
	public class TrainLocomotive : TrainCarBase
	{
		private const float MIN_SPEED = 0.2f;
		private const float MAX_SPEED = 2.7f;
		private const float SMOKE_SPEED_MULTIPLIER = 8f;
		private const int MAX_PARTICLES_MULTIPLIER = 3;

		private enum EngineSoundState
		{
			Start = 0,
			AccelerateOrSetProperSpeed,
			Stop
		};

		[SerializeField]
		[Range(MIN_SPEED, MAX_SPEED)]
		protected float _initialSpeed = 0f;

		[SerializeField] private GameObject _startStopButton = null;
		[SerializeField] private GameObject _decreaseSpeedButton = null;
		[SerializeField] private GameObject _increaseSpeedButton = null;
		[SerializeField] private GameObject _smokeButton = null;
		[SerializeField] private GameObject _whistleButton = null;
		[SerializeField] private GameObject _reverseButton = null;
		[SerializeField] private AudioSource _whistleAudioSource = null;
		[SerializeField] private AudioClip _whistleSound = null;
		[SerializeField] private AudioSource _engineAudioSource = null;
		[SerializeField] private AudioClip[] _accelerationSounds = null;
		[SerializeField] private AudioClip[] _decelerationSounds = null;
		[SerializeField] private AudioClip _startUpSound = null;

		[SerializeField] private AudioSource _smokeStackAudioSource = null;
		[SerializeField] private AudioClip _smokeSound = null;

		[SerializeField] private ParticleSystem _smoke1 = null;
		[SerializeField] private ParticleSystem _smoke2 = null;
		[SerializeField] private TrainCarBase[] _childCars = null;

		private bool _isMoving = true;
		private bool _reverse = false;
		private float _currentSpeed, _speedDiv;

		private float _standardRateOverTimeMultiplier;
		private int _standardMaxParticles;

		private Coroutine _startStopTrainCr;

		private void Start()
		{
			Assert.IsNotNull(_startStopButton);
			Assert.IsNotNull(_decreaseSpeedButton);
			Assert.IsNotNull(_increaseSpeedButton);
			Assert.IsNotNull(_smokeButton);
			Assert.IsNotNull(_whistleButton);
			Assert.IsNotNull(_reverseButton);
			Assert.IsNotNull(_whistleAudioSource);
			Assert.IsNotNull(_whistleSound);
			Assert.IsNotNull(_smoke1);

			Assert.IsNotNull(_engineAudioSource);
			Assert.IsNotNull(_accelerationSounds);
			Assert.IsNotNull(_decelerationSounds);
			Assert.IsNotNull(_startUpSound);

			Assert.IsNotNull(_smokeStackAudioSource);
			Assert.IsNotNull(_smokeSound);


			_standardRateOverTimeMultiplier = _smoke1.emission.rateOverTimeMultiplier;
			_standardMaxParticles = _smoke1.main.maxParticles;

			Distance = 0.0f;
			_speedDiv = (MAX_SPEED - MIN_SPEED) / _accelerationSounds.Length;
			_currentSpeed = _initialSpeed;
			UpdateCarPosition();

			_smoke1.Stop();
			_startStopTrainCr = StartCoroutine(StartStopTrain(true));
		}

		private void Update()
		{
			UpdatePosition();
		}

		public override void UpdatePosition()
		{
			if (!_isMoving)
			{
				return;
			}

			if (_trainTrack != null)
			{
				UpdateDistance();
				UpdateCarPosition();
				RotateCarWheels();
			}

			foreach (var trainCarBase in _childCars)
			{
				trainCarBase.UpdatePosition();
			}
		}

		public void StartStopStateChanged()
		{
			if (_startStopTrainCr == null)
			{
				_startStopTrainCr = StartCoroutine(StartStopTrain(!_isMoving));
			}
		}

		private IEnumerator StartStopTrain(bool startTrain)
		{
			float endSpeed = startTrain ? _initialSpeed : 0.0f;

			var timePeriodForSpeedChange = 3.0f;
			if (startTrain)
			{
				_smoke1.Play();
				_isMoving = true;
				var emissionModule1 = _smoke1.emission;
				var mainModule = _smoke1.main;
				emissionModule1.rateOverTimeMultiplier = _standardRateOverTimeMultiplier;
				mainModule.maxParticles = _standardMaxParticles;
				timePeriodForSpeedChange = PlayEngineSound(EngineSoundState.Start);
			}
			else
			{
				timePeriodForSpeedChange = PlayEngineSound(EngineSoundState.Stop);
			}

			// don't loop audio at first; only do when train continues movement below
			_engineAudioSource.loop = false;

			// make time period a tad shorter so that if it's not looping, we don't
			// catch the beginning of the sound
			timePeriodForSpeedChange = timePeriodForSpeedChange * 0.9f;
			float startTime = Time.time;
			float endTime = Time.time + timePeriodForSpeedChange;
			float startSpeed = _currentSpeed;
			while (Time.time < endTime)
			{
				float t = (Time.time - startTime) / timePeriodForSpeedChange;
				_currentSpeed = startSpeed * (1.0f - t) + endSpeed * t;
				UpdateSmokeEmissionBasedOnSpeed();
				yield return null;
			}

			_currentSpeed = endSpeed;
			_startStopTrainCr = null;
			_isMoving = startTrain;
			if (!_isMoving)
			{
				_smoke1.Stop();
			}
			else
			{
				_engineAudioSource.loop = true;
				PlayEngineSound(EngineSoundState.AccelerateOrSetProperSpeed);
			}
		}

		private float PlayEngineSound(EngineSoundState engineSoundState)
		{
			AudioClip audioClip = null;

			if (engineSoundState == EngineSoundState.Start)
			{
				audioClip = _startUpSound;
			}
			else
			{
				AudioClip[] audioClips = engineSoundState == EngineSoundState.AccelerateOrSetProperSpeed
					? _accelerationSounds
					: _decelerationSounds;
				int numSounds = audioClips.Length;
				int speedIndex = (int)Mathf.Round((_currentSpeed - MIN_SPEED) / _speedDiv);
				audioClip = audioClips[Mathf.Clamp(speedIndex, 0, numSounds - 1)];
			}

			// if audio is already playing and we are playing the same track, don't interrupt it
			if (_engineAudioSource.clip == audioClip && _engineAudioSource.isPlaying &&
				engineSoundState == EngineSoundState.AccelerateOrSetProperSpeed)
			{
				return 0.0f;
			}

			_engineAudioSource.clip = audioClip;
			_engineAudioSource.timeSamples = 0;
			_engineAudioSource.Play();

			return audioClip.length;
		}

		private void UpdateDistance()
		{
			var signedSpeed = _reverse ? -_currentSpeed : _currentSpeed;
			Distance = (Distance + signedSpeed * Time.deltaTime) % _trainTrack.TrackLength;
		}

		public void DecreaseSpeedStateChanged()
		{
			if (_startStopTrainCr == null && _isMoving)
			{
				_currentSpeed = Mathf.Clamp(_currentSpeed - _speedDiv, MIN_SPEED, MAX_SPEED);
				UpdateSmokeEmissionBasedOnSpeed();
				PlayEngineSound(EngineSoundState.AccelerateOrSetProperSpeed);
			}
		}

		public void IncreaseSpeedStateChanged()
		{
			if (_startStopTrainCr == null && _isMoving)
			{
				_currentSpeed = Mathf.Clamp(_currentSpeed + _speedDiv, MIN_SPEED, MAX_SPEED);
				UpdateSmokeEmissionBasedOnSpeed();
				PlayEngineSound(EngineSoundState.AccelerateOrSetProperSpeed);
			}
		}

		private void UpdateSmokeEmissionBasedOnSpeed()
		{
			var emissionModule = _smoke1.emission;
			emissionModule.rateOverTimeMultiplier = GetCurrentSmokeEmission();
			var mainModule = _smoke1.main;
			mainModule.maxParticles = (int)Mathf.Lerp(_standardMaxParticles, _standardMaxParticles * MAX_PARTICLES_MULTIPLIER,
				_currentSpeed / (MAX_SPEED - MIN_SPEED));
		}

		private float GetCurrentSmokeEmission()
		{
			return Mathf.Lerp(_standardRateOverTimeMultiplier, _standardRateOverTimeMultiplier * SMOKE_SPEED_MULTIPLIER,
				_currentSpeed / (MAX_SPEED - MIN_SPEED));
		}

		public void SmokeButtonStateChanged()
		{
			if (_isMoving)
			{
				_smokeStackAudioSource.clip = _smokeSound;
				_smokeStackAudioSource.timeSamples = 0;
				_smokeStackAudioSource.Play();

				_smoke2.time = 0.0f;
				_smoke2.Play();
			}
		}

		public void WhistleButtonStateChanged()
		{
			if (_whistleSound != null)
			{
				_whistleAudioSource.clip = _whistleSound;
				_whistleAudioSource.timeSamples = 0;
				_whistleAudioSource.Play();
			}
		}

		public void ReverseButtonStateChanged()
		{
			_reverse = !_reverse;
		}
	}
}
