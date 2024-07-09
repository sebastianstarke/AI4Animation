// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class NativeVideoPlayer
{

    public enum PlabackState
    {
        Idle = 1,
        Preparing = 2,
        Buffering = 3,
        Ready = 4,
        Ended = 5
    }

    public enum StereoMode
    {
        Unknown = -1,
        Mono = 0,
        TopBottom = 1,
        LeftRight = 2,
        Mesh = 3
    }

    private static System.IntPtr? _Activity;
    private static System.IntPtr? _VideoPlayerClass;

    private static readonly jvalue[] EmptyParams = new jvalue[0];

    private static System.IntPtr getIsPlayingMethodId;
    private static System.IntPtr getCurrentPlaybackStateMethodId;
    private static System.IntPtr getDurationMethodId;
    private static System.IntPtr getStereoModeMethodId;
    private static System.IntPtr getWidthMethodId;
    private static System.IntPtr getHeightMethodId;
    private static System.IntPtr getPlaybackPositionMethodId;
    private static System.IntPtr setPlaybackPositionMethodId;
    private static jvalue[] setPlaybackPositionParams;
    private static System.IntPtr playVideoMethodId;
    private static jvalue[] playVideoParams;
    private static System.IntPtr stopMethodId;
    private static System.IntPtr resumeMethodId;
    private static System.IntPtr pauseMethodId;
    private static System.IntPtr setPlaybackSpeedMethodId;
    private static jvalue[] setPlaybackSpeedParams;
    private static System.IntPtr setLoopingMethodId;
    private static jvalue[] setLoopingParams;
    private static System.IntPtr setListenerRotationQuaternionMethodId;
    private static jvalue[] setListenerRotationQuaternionParams;

    private static System.IntPtr VideoPlayerClass
    {
        get
        {
            if (!_VideoPlayerClass.HasValue)
            {
                try
                {
                    System.IntPtr myVideoPlayerClass = AndroidJNI.FindClass("com/oculus/videoplayer/NativeVideoPlayer");

                    if (myVideoPlayerClass != System.IntPtr.Zero)
                    {
                        _VideoPlayerClass = AndroidJNI.NewGlobalRef(myVideoPlayerClass);

                        AndroidJNI.DeleteLocalRef(myVideoPlayerClass);
                    }
                    else
                    {
                        Debug.LogError("Failed to find NativeVideoPlayer class");
                        _VideoPlayerClass = System.IntPtr.Zero;
                    }
                }
                catch (System.Exception ex)
                {
                    Debug.LogError("Failed to find NativeVideoPlayer class");
                    Debug.LogException(ex);
                    _VideoPlayerClass = System.IntPtr.Zero;
                }
            }
            return _VideoPlayerClass.GetValueOrDefault();
        }
    }

    private static System.IntPtr Activity
    {
        get
        {
            if (!_Activity.HasValue)
            {
                try
                {
                    System.IntPtr unityPlayerClass = AndroidJNI.FindClass("com/unity3d/player/UnityPlayer");
                    System.IntPtr currentActivityField = AndroidJNI.GetStaticFieldID(unityPlayerClass, "currentActivity", "Landroid/app/Activity;");
                    System.IntPtr activity = AndroidJNI.GetStaticObjectField(unityPlayerClass, currentActivityField);

                    _Activity = AndroidJNI.NewGlobalRef(activity);

                    AndroidJNI.DeleteLocalRef(activity);
                    AndroidJNI.DeleteLocalRef(unityPlayerClass);
                }
                catch (System.Exception ex)
                {
                    Debug.LogException(ex);
                    _Activity = System.IntPtr.Zero;
                }
            }
            return _Activity.GetValueOrDefault();
        }
    }

    public static bool IsAvailable
    {
        get
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            return VideoPlayerClass != System.IntPtr.Zero;
#else
            return false;
#endif
        }
    }

    public static bool IsPlaying
    {
        get
        {
            if (getIsPlayingMethodId == System.IntPtr.Zero)
            {
                getIsPlayingMethodId = AndroidJNI.GetStaticMethodID(VideoPlayerClass, "getIsPlaying", "()Z");
            }

            return AndroidJNI.CallStaticBooleanMethod(VideoPlayerClass, getIsPlayingMethodId, EmptyParams);
        }
    }

    public static PlabackState CurrentPlaybackState {
        get {
            if (getCurrentPlaybackStateMethodId == System.IntPtr.Zero)
            {
                getCurrentPlaybackStateMethodId = AndroidJNI.GetStaticMethodID(VideoPlayerClass, "getCurrentPlaybackState", "()I");
            }

            return (PlabackState)AndroidJNI.CallStaticIntMethod(VideoPlayerClass, getCurrentPlaybackStateMethodId, EmptyParams);
        }
    }

    public static long Duration
    {
        get
        {
            if (getDurationMethodId == System.IntPtr.Zero)
            {
                getDurationMethodId = AndroidJNI.GetStaticMethodID(VideoPlayerClass, "getDuration", "()J");
            }

            return AndroidJNI.CallStaticLongMethod(VideoPlayerClass, getDurationMethodId, EmptyParams);
        }
    }

    public static StereoMode VideoStereoMode
    {
        get
        {
            if (getStereoModeMethodId == System.IntPtr.Zero)
            {
                getStereoModeMethodId = AndroidJNI.GetStaticMethodID(VideoPlayerClass, "getStereoMode", "()I");
            }

            return (StereoMode)AndroidJNI.CallStaticIntMethod(VideoPlayerClass, getStereoModeMethodId, EmptyParams);
        }
    }

    public static int VideoWidth
    {
        get
        {
            if (getWidthMethodId == System.IntPtr.Zero)
            {
                getWidthMethodId = AndroidJNI.GetStaticMethodID(VideoPlayerClass, "getWidth", "()I");
            }

            return AndroidJNI.CallStaticIntMethod(VideoPlayerClass, getWidthMethodId, EmptyParams);
        }
    }

    public static int VideoHeight
    {
        get
        {
            if (getHeightMethodId == System.IntPtr.Zero)
            {
                getHeightMethodId = AndroidJNI.GetStaticMethodID(VideoPlayerClass, "getHeight", "()I");
            }

            return AndroidJNI.CallStaticIntMethod(VideoPlayerClass, getHeightMethodId, EmptyParams);
        }
    }

    public static long PlaybackPosition
    {
        get
        {
            if (getPlaybackPositionMethodId == System.IntPtr.Zero)
            {
                getPlaybackPositionMethodId = AndroidJNI.GetStaticMethodID(VideoPlayerClass, "getPlaybackPosition", "()J");
            }

            return AndroidJNI.CallStaticLongMethod(VideoPlayerClass, getPlaybackPositionMethodId, EmptyParams);
        }
        set
        {
            if (setPlaybackPositionMethodId == System.IntPtr.Zero)
            {
                setPlaybackPositionMethodId = AndroidJNI.GetStaticMethodID(VideoPlayerClass, "setPlaybackPosition", "(J)V");
                setPlaybackPositionParams = new jvalue[1];
            }

            setPlaybackPositionParams[0].j = value;

            AndroidJNI.CallStaticVoidMethod(VideoPlayerClass, setPlaybackPositionMethodId, setPlaybackPositionParams);
        }
    }

    public static void PlayVideo(string path, string drmLicenseUrl, System.IntPtr surfaceObj)
    {
        if (playVideoMethodId == System.IntPtr.Zero)
        {
            playVideoMethodId = AndroidJNI.GetStaticMethodID(VideoPlayerClass, "playVideo", "(Landroid/content/Context;Ljava/lang/String;Ljava/lang/String;Landroid/view/Surface;)V");
            playVideoParams = new jvalue[4];
        }

        System.IntPtr filePathJString = AndroidJNI.NewStringUTF(path);
        System.IntPtr drmLicenseUrlJString = AndroidJNI.NewStringUTF(drmLicenseUrl);

        playVideoParams[0].l = Activity;
        playVideoParams[1].l = filePathJString;
        playVideoParams[2].l = drmLicenseUrlJString;
        playVideoParams[3].l = surfaceObj;
        AndroidJNI.CallStaticVoidMethod(VideoPlayerClass, playVideoMethodId, playVideoParams);

        AndroidJNI.DeleteLocalRef(filePathJString);
        AndroidJNI.DeleteLocalRef(drmLicenseUrlJString);
    }

    public static void Stop()
    {
        if (stopMethodId == System.IntPtr.Zero)
        {
            stopMethodId = AndroidJNI.GetStaticMethodID(VideoPlayerClass, "stop", "()V");
        }

        AndroidJNI.CallStaticVoidMethod(VideoPlayerClass, stopMethodId, EmptyParams);
    }

    public static void Play()
    {
        if (resumeMethodId == System.IntPtr.Zero)
        {
            resumeMethodId = AndroidJNI.GetStaticMethodID(VideoPlayerClass, "resume", "()V");
        }

        AndroidJNI.CallStaticVoidMethod(VideoPlayerClass, resumeMethodId, EmptyParams);
    }

    public static void Pause()
    {
        if (pauseMethodId == System.IntPtr.Zero)
        {
            pauseMethodId = AndroidJNI.GetStaticMethodID(VideoPlayerClass, "pause", "()V");
        }

        AndroidJNI.CallStaticVoidMethod(VideoPlayerClass, pauseMethodId, EmptyParams);
    }

    public static void SetPlaybackSpeed(float speed)
    {
        if (setPlaybackSpeedMethodId == System.IntPtr.Zero)
        {
            setPlaybackSpeedMethodId = AndroidJNI.GetStaticMethodID(VideoPlayerClass, "setPlaybackSpeed", "(F)V");
            setPlaybackSpeedParams = new jvalue[1];
        }

        setPlaybackSpeedParams[0].f = speed;
        AndroidJNI.CallStaticVoidMethod(VideoPlayerClass, setPlaybackSpeedMethodId, setPlaybackSpeedParams);
    }
    public static void SetLooping(bool looping)
    {
        if (setLoopingMethodId == System.IntPtr.Zero)
        {
            setLoopingMethodId = AndroidJNI.GetStaticMethodID(VideoPlayerClass, "setLooping", "(Z)V");
            setLoopingParams = new jvalue[1];
        }

        setLoopingParams[0].z = looping;
        AndroidJNI.CallStaticVoidMethod(VideoPlayerClass, setLoopingMethodId, setLoopingParams);
    }

  public static void SetListenerRotation(Quaternion rotation)
    {
        if (setListenerRotationQuaternionMethodId == System.IntPtr.Zero)
        {
            setListenerRotationQuaternionMethodId = AndroidJNI.GetStaticMethodID(VideoPlayerClass, "setListenerRotationQuaternion", "(FFFF)V");
            setListenerRotationQuaternionParams = new jvalue[4];
        }

        setListenerRotationQuaternionParams[0].f = rotation.x;
        setListenerRotationQuaternionParams[1].f = rotation.y;
        setListenerRotationQuaternionParams[2].f = rotation.z;
        setListenerRotationQuaternionParams[3].f = rotation.w;
        AndroidJNI.CallStaticVoidMethod(VideoPlayerClass, setListenerRotationQuaternionMethodId, setListenerRotationQuaternionParams);
    }

}
