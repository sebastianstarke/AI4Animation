// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using UnityEngine;
using System;
using System.IO;

public class MoviePlayerSample : MonoBehaviour
{
    private bool    videoPausedBeforeAppPause = false;

    private UnityEngine.Video.VideoPlayer videoPlayer = null;
    private OVROverlay          overlay = null;
    private Renderer            mediaRenderer = null;

    public bool IsPlaying { get; private set; }
    public long Duration { get; private set; }
    public long PlaybackPosition { get; private set; }

    private RenderTexture copyTexture;
    private Material externalTex2DMaterial;

    public string MovieName;
    public string DrmLicenseUrl;
    public bool LoopVideo;
    public VideoShape Shape;
    public VideoStereo Stereo;
    public bool AutoDetectStereoLayout;
    public bool DisplayMono;

    // keep track of last state so we know when to update our display
    VideoShape _LastShape = (VideoShape)(-1);
    VideoStereo _LastStereo = (VideoStereo)(-1);
    bool _LastDisplayMono = false;

    public enum VideoShape
    {
        _360,
        _180,
        Quad
    }

    public enum VideoStereo
    {
        Mono,
        TopBottom,
        LeftRight,
        BottomTop
    }

    /// <summary>
    /// Initialization of the movie surface
    /// </summary>
    void Awake()
    {
        Debug.Log("MovieSample Awake");

        mediaRenderer = GetComponent<Renderer>();

        videoPlayer = GetComponent<UnityEngine.Video.VideoPlayer>();
        if (videoPlayer == null)
            videoPlayer = gameObject.AddComponent<UnityEngine.Video.VideoPlayer>();
        videoPlayer.isLooping = LoopVideo;

        overlay = GetComponent<OVROverlay>();
        if (overlay == null)
            overlay = gameObject.AddComponent<OVROverlay>();

        // disable it to reset it.
        overlay.enabled = false;
        // only can use external surface with native plugin
        overlay.isExternalSurface = NativeVideoPlayer.IsAvailable;
        // only mobile has Equirect shape
        overlay.enabled = (overlay.currentOverlayShape != OVROverlay.OverlayShape.Equirect || Application.platform == RuntimePlatform.Android);

#if UNITY_EDITOR
        overlay.currentOverlayShape = OVROverlay.OverlayShape.Quad;
        overlay.enabled = true;
#endif
    }

    private bool IsLocalVideo(string movieName)
    {
        // if the path contains any url scheme, it is not local
        return !movieName.Contains("://");
    }

    private void UpdateShapeAndStereo()
    {
        if (AutoDetectStereoLayout)
        {
            if (overlay.isExternalSurface)
            {
                int w = NativeVideoPlayer.VideoWidth;
                int h = NativeVideoPlayer.VideoHeight;
                switch(NativeVideoPlayer.VideoStereoMode)
                {
                    case NativeVideoPlayer.StereoMode.Mono:
                        Stereo = VideoStereo.Mono;
                        break;
                    case NativeVideoPlayer.StereoMode.LeftRight:
                        Stereo = VideoStereo.LeftRight;
                        break;
                    case NativeVideoPlayer.StereoMode.TopBottom:
                        Stereo = VideoStereo.TopBottom;
                        break;
                    case NativeVideoPlayer.StereoMode.Unknown:
                        if (w > h)
                        {
                            Stereo = VideoStereo.LeftRight;
                        }
                        else
                        {
                            Stereo = VideoStereo.TopBottom;
                        }
                        break;
                }
            }
        }

        if (Shape != _LastShape || Stereo != _LastStereo || DisplayMono != _LastDisplayMono)
        {
            Rect destRect = new Rect(0, 0, 1, 1);
            switch (Shape)
            {
                case VideoShape._360:
                    // set shape to Equirect
                    overlay.currentOverlayShape = OVROverlay.OverlayShape.Equirect;
                    break;
                case VideoShape._180:
                    overlay.currentOverlayShape = OVROverlay.OverlayShape.Equirect;
                    destRect = new Rect(0.25f, 0, 0.5f, 1.0f);
                    break;
                case VideoShape.Quad:
                default:
                    overlay.currentOverlayShape = OVROverlay.OverlayShape.Quad;
                    break;
            }

            overlay.overrideTextureRectMatrix = true;
            overlay.invertTextureRects = false;

            Rect sourceLeft = new Rect(0, 0, 1, 1);
            Rect sourceRight = new Rect(0, 0, 1, 1);
            switch (Stereo)
            {
                case VideoStereo.LeftRight:
                    // set source matrices for left/right
                    sourceLeft  = new Rect(0.0f, 0.0f, 0.5f, 1.0f);
                    sourceRight = new Rect(0.5f, 0.0f, 0.5f, 1.0f);
                    break;
                case VideoStereo.TopBottom:
                    // set source matrices for top/bottom
                    sourceLeft  = new Rect(0.0f, 0.5f, 1.0f, 0.5f);
                    sourceRight = new Rect(0.0f, 0.0f, 1.0f, 0.5f);
                    break;
                case VideoStereo.BottomTop:
                    // set source matrices for top/bottom
                    sourceLeft  = new Rect(0.0f, 0.0f, 1.0f, 0.5f);
                    sourceRight = new Rect(0.0f, 0.5f, 1.0f, 0.5f);
                    break;
            }

            overlay.SetSrcDestRects(sourceLeft, DisplayMono ? sourceLeft : sourceRight, destRect, destRect);

            _LastDisplayMono = DisplayMono;
            _LastStereo = Stereo;
            _LastShape = Shape;
        }
    }

    private System.Collections.IEnumerator Start()
    {
        if (mediaRenderer.material == null)
        {
            Debug.LogError("No material for movie surface");
            yield break;
        }

        // wait 1 second to start (there is a bug in Unity where starting
        // the video too soon will cause it to fail to load)
        yield return new WaitForSeconds(1.0f);

        if (!string.IsNullOrEmpty(MovieName))
        {
            if (IsLocalVideo(MovieName))
            {
#if UNITY_EDITOR
                // in editor, just pull in the movie file from wherever it lives (to test without putting in streaming assets)
                var guids = UnityEditor.AssetDatabase.FindAssets(Path.GetFileNameWithoutExtension(MovieName));

                if (guids.Length > 0)
                {
                    string video = UnityEditor.AssetDatabase.GUIDToAssetPath(guids[0]);
                    Play(video, null);
                }
#else
                Play(Application.streamingAssetsPath +"/" + MovieName, null);
#endif
            }
            else
            {
                Play(MovieName, DrmLicenseUrl);
            }
        }
    }

    public void Play(string moviePath, string drmLicencesUrl)
    {
        if (moviePath != string.Empty)
        {
            Debug.Log("Playing Video: " + moviePath);
            if (overlay.isExternalSurface)
            {
                OVROverlay.ExternalSurfaceObjectCreated surfaceCreatedCallback = () =>
                {
                    Debug.Log("Playing ExoPlayer with SurfaceObject");
                    NativeVideoPlayer.PlayVideo(moviePath, drmLicencesUrl, overlay.externalSurfaceObject);
                    NativeVideoPlayer.SetLooping(LoopVideo);
                };

                if (overlay.externalSurfaceObject == IntPtr.Zero)
                {
                    overlay.externalSurfaceObjectCreated = surfaceCreatedCallback;
                }
                else
                {
                    surfaceCreatedCallback.Invoke();
                }
            }
            else
            {
                Debug.Log("Playing Unity VideoPlayer");
                videoPlayer.url = moviePath;
                videoPlayer.Prepare();
                videoPlayer.Play();
            }

            Debug.Log("MovieSample Start");
            IsPlaying = true;
        }
        else
        {
            Debug.LogError("No media file name provided");
        }
    }

    public void Play()
    {
        if (overlay.isExternalSurface)
        {
            NativeVideoPlayer.Play();
        }
        else
        {
            videoPlayer.Play();
        }
        IsPlaying = true;
    }

    public void Pause()
    {
        if (overlay.isExternalSurface)
        {
            NativeVideoPlayer.Pause();
        }
        else
        {
            videoPlayer.Pause();
        }
        IsPlaying = false;
    }

    public void SeekTo(long position)
    {
        long seekPos = Math.Max(0, Math.Min(Duration, position));
        if (overlay.isExternalSurface)
        {
            NativeVideoPlayer.PlaybackPosition = seekPos;
        }
        else
        {
            videoPlayer.time = seekPos / 1000.0;
        }
    }

    void Update()
    {
        UpdateShapeAndStereo();
        if (!overlay.isExternalSurface)
        {
            var displayTexture = videoPlayer.texture != null ? videoPlayer.texture : Texture2D.blackTexture;
            if (overlay.enabled)
            {
                if (overlay.textures[0] != displayTexture)
                {
                    // OVROverlay won't check if the texture changed, so disable to clear old texture
                    overlay.enabled = false;
                    overlay.textures[0] = displayTexture;
                    overlay.enabled = true;
                }
            }
            else
            {
                mediaRenderer.material.mainTexture = displayTexture;
                mediaRenderer.material.SetVector("_SrcRectLeft", overlay.srcRectLeft.ToVector());
                mediaRenderer.material.SetVector("_SrcRectRight", overlay.srcRectRight.ToVector());
            }
            IsPlaying = videoPlayer.isPlaying;
            PlaybackPosition = (long)(videoPlayer.time * 1000L);

#if UNITY_2019_1_OR_NEWER
            Duration = (long)(videoPlayer.length * 1000L); 
#else
            Duration = videoPlayer.frameRate > 0 ? (long)(videoPlayer.frameCount / videoPlayer.frameRate * 1000L) : 0L;
#endif
        }
        else
        {
            NativeVideoPlayer.SetListenerRotation(Camera.main.transform.rotation);
            IsPlaying = NativeVideoPlayer.IsPlaying;
            PlaybackPosition = NativeVideoPlayer.PlaybackPosition;
            Duration = NativeVideoPlayer.Duration;
            if (IsPlaying && (int)OVRManager.display.displayFrequency != 60)
            {
                OVRManager.display.displayFrequency = 60.0f;
            }
            else if (!IsPlaying && (int)OVRManager.display.displayFrequency != 72)
            {
                OVRManager.display.displayFrequency = 72.0f;
            }
        }
  }

    public void SetPlaybackSpeed(float speed)
    {
        // clamp at 0
        speed = Mathf.Max(0, speed);
        if (overlay.isExternalSurface)
        {
            NativeVideoPlayer.SetPlaybackSpeed(speed);
        }
        else
        {
            videoPlayer.playbackSpeed = speed;
        }
    }    

    public void Stop()
    {
        if (overlay.isExternalSurface)
        {
            NativeVideoPlayer.Stop();
        }
        else
        {
            videoPlayer.Stop();
        }

        IsPlaying = false;
    }

  /// <summary>
  /// Pauses video playback when the app loses or gains focus
  /// </summary>
  void OnApplicationPause(bool appWasPaused)
    {
        Debug.Log("OnApplicationPause: " + appWasPaused);
        if (appWasPaused)
        {
            videoPausedBeforeAppPause = !IsPlaying;
        }

        // Pause/unpause the video only if it had been playing prior to app pause
        if (!videoPausedBeforeAppPause)
        {
            if (appWasPaused)
            {
                Pause();
            }
            else
            {
                Play();
            }
        }
    }
}
