// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class MoviePlayerSampleControls : MonoBehaviour
{
    public MoviePlayerSample Player;

    public UnityEngine.EventSystems.OVRInputModule InputModule;
    public OVRGazePointer GazePointer;

    public GameObject LeftHand;
    public GameObject RightHand;

    public Canvas Canvas;
    public ButtonDownListener PlayPause;
    public MediaPlayerImage PlayPauseImage;
    public Slider ProgressBar;
    public ButtonDownListener FastForward;
    public MediaPlayerImage FastForwardImage;
    public ButtonDownListener Rewind;
    public MediaPlayerImage RewindImage;

    public float TimeoutTime = 10f;

    private bool _isVisible = false;

    private float _lastButtonTime = 0f;

    private bool _didSeek = false;
    private long _seekPreviousPosition;

    private long _rewindStartPosition;
    private float _rewindStartTime;

    private enum PlaybackState
    {
        Playing,
        Paused,
        Rewinding,
        FastForwarding
    }

    private PlaybackState _state = PlaybackState.Playing;


    void Start()
    {
        PlayPause.onButtonDown += OnPlayPauseClicked;
        FastForward.onButtonDown += OnFastForwardClicked;
        Rewind.onButtonDown += OnRewindClicked;
        ProgressBar.onValueChanged.AddListener(OnSeekBarMoved);

        PlayPauseImage.buttonType = MediaPlayerImage.ButtonType.Pause;
        FastForwardImage.buttonType = MediaPlayerImage.ButtonType.SkipForward;
        RewindImage.buttonType = MediaPlayerImage.ButtonType.SkipBack;
        SetVisible(false);
    }

    void OnPlayPauseClicked()
    {
        switch(_state)
        {
            case PlaybackState.Paused:
                Player.Play();
                PlayPauseImage.buttonType = MediaPlayerImage.ButtonType.Pause;
                FastForwardImage.buttonType = MediaPlayerImage.ButtonType.FastForward;
                RewindImage.buttonType = MediaPlayerImage.ButtonType.Rewind;
                _state = PlaybackState.Playing;
                break;
            case PlaybackState.Playing:
                Player.Pause();
                PlayPauseImage.buttonType = MediaPlayerImage.ButtonType.Play;
                FastForwardImage.buttonType = MediaPlayerImage.ButtonType.SkipForward;
                RewindImage.buttonType = MediaPlayerImage.ButtonType.SkipBack;
                _state = PlaybackState.Paused;
                break;
            case PlaybackState.FastForwarding:
                Player.SetPlaybackSpeed(1);
                PlayPauseImage.buttonType = MediaPlayerImage.ButtonType.Pause;
                _state = PlaybackState.Playing;
                break;
            case PlaybackState.Rewinding:
                Player.Play();
                _state = PlaybackState.Playing;
                PlayPauseImage.buttonType = MediaPlayerImage.ButtonType.Pause;
                break;
        }
    }

    void OnFastForwardClicked()
    {
        switch(_state)
        {
            case PlaybackState.FastForwarding:
                Player.SetPlaybackSpeed(1);
                _state = PlaybackState.Playing;
                PlayPauseImage.buttonType = MediaPlayerImage.ButtonType.Pause;
                break;
            case PlaybackState.Rewinding:
                Player.Play();
                Player.SetPlaybackSpeed(2);
                _state = PlaybackState.FastForwarding;
                break;
            case PlaybackState.Playing:
                Player.SetPlaybackSpeed(2);
                PlayPauseImage.buttonType = MediaPlayerImage.ButtonType.Play;
                _state = PlaybackState.FastForwarding;
                break;
            case PlaybackState.Paused:
                // skip ahead 15 seconds
                Seek(Player.PlaybackPosition + 15000);
                break;
        }
    }

    void OnRewindClicked()
    {
        switch (_state)
        {
            case PlaybackState.FastForwarding:
            case PlaybackState.Playing:
                Player.SetPlaybackSpeed(1);
                Player.Pause();
                // Player's do not support negative speed. Instead, we need to seek step by step
                _rewindStartPosition = Player.PlaybackPosition;
                _rewindStartTime = Time.time;
                PlayPauseImage.buttonType = MediaPlayerImage.ButtonType.Play;
                _state = PlaybackState.Rewinding;
                break;
            case PlaybackState.Rewinding:
                Player.Play();
                PlayPauseImage.buttonType = MediaPlayerImage.ButtonType.Pause;
                _state = PlaybackState.Playing;
                break;
            case PlaybackState.Paused:
                // skip ahead 15 seconds
                Seek(Player.PlaybackPosition - 15000);
                break;
        }
    }

    void OnSeekBarMoved(float value)
    {
        long newPos = (long)(value * Player.Duration);

        // only seek if the position changed more than 200ms
        if (Mathf.Abs(newPos - Player.PlaybackPosition) > 200)
        {            
            Seek(newPos);
        }
    }

    private void Seek(long pos)
    {
        _didSeek = true;
        _seekPreviousPosition = Player.PlaybackPosition;
        Player.SeekTo(pos);
    }

    private void Update()
    {
        if(OVRInput.Get(OVRInput.Button.One) || OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger) || OVRInput.Get(OVRInput.Button.SecondaryIndexTrigger))
        {
            _lastButtonTime = Time.time;
            if (!_isVisible)
            {
                SetVisible(true);
            }
        }

        if (OVRInput.GetActiveController() == OVRInput.Controller.LTouch)
        {
            InputModule.rayTransform = LeftHand.transform;
            GazePointer.rayTransform = LeftHand.transform;
        }
        else
        {
            InputModule.rayTransform = RightHand.transform;
            GazePointer.rayTransform = RightHand.transform;
        }

        // if back is pressed, hide controls immediately
        if (OVRInput.Get(OVRInput.Button.Back))
        {
            if (_isVisible)
            {
                SetVisible(false);
            }
        }

        if (_state == PlaybackState.Rewinding)
        {
            // smoothly update our seekbar
            ProgressBar.value = Mathf.Clamp01((_rewindStartPosition - 1000L * (Time.time - _rewindStartTime)) / Player.Duration);
        }

        // if we are playing, hide the controls after 15 seconds
        if (_isVisible && _state == PlaybackState.Playing && Time.time - _lastButtonTime > TimeoutTime)
        {
            SetVisible(false);
        }

        if (_isVisible)
        {
            if (!_didSeek || Mathf.Abs(_seekPreviousPosition - Player.PlaybackPosition) > 50)
            {
                _didSeek = false;

                if (Player.Duration > 0)
                {
                    // update our progress bar
                    ProgressBar.value = (float)(Player.PlaybackPosition / (double)Player.Duration);
                }
                else
                {
                    ProgressBar.value = 0;
                }
            }
        }
    }

    private void SetVisible(bool visible)
    {
        Canvas.enabled = visible;
        _isVisible = visible;
        Player.DisplayMono = visible;
        LeftHand.SetActive(visible);
        RightHand.SetActive(visible);
        Debug.Log("Controls Visible: " + visible);
    }
}
