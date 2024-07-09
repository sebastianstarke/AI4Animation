/**
 * **********************************************************************************
 *
 * <p>Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.
 *
 * <p>**********************************************************************************
 */
package com.oculus.videoplayer;

import android.content.Context;
import android.net.Uri;
import android.os.Handler;
import android.os.Looper;
import android.os.storage.OnObbStateChangeListener;
import android.os.storage.StorageManager;
import android.util.Log;
import android.view.Surface;
import com.google.android.exoplayer2.DefaultRenderersFactory;
import com.google.android.exoplayer2.ExoPlayer;
import com.google.android.exoplayer2.Format;
import com.google.android.exoplayer2.MediaItem;
import com.google.android.exoplayer2.PlaybackParameters;
import com.google.android.exoplayer2.Player;
import com.google.android.exoplayer2.Renderer;
import com.google.android.exoplayer2.Timeline;
import com.google.android.exoplayer2.audio.AudioRendererEventListener;
import com.google.android.exoplayer2.audio.AudioSink;
import com.google.android.exoplayer2.database.DatabaseProvider;
import com.google.android.exoplayer2.database.StandaloneDatabaseProvider;
import com.google.android.exoplayer2.drm.DefaultDrmSessionManagerProvider;
import com.google.android.exoplayer2.metadata.MetadataOutput;
import com.google.android.exoplayer2.source.DefaultMediaSourceFactory;
import com.google.android.exoplayer2.source.MediaSource;
import com.google.android.exoplayer2.text.TextOutput;
import com.google.android.exoplayer2.trackselection.TrackSelectionParameters;
import com.google.android.exoplayer2.upstream.DataSource;
import com.google.android.exoplayer2.upstream.DefaultDataSource;
import com.google.android.exoplayer2.upstream.DefaultHttpDataSource;
import com.google.android.exoplayer2.upstream.HttpDataSource;
import com.google.android.exoplayer2.upstream.cache.Cache;
import com.google.android.exoplayer2.upstream.cache.CacheDataSource;
import com.google.android.exoplayer2.upstream.cache.NoOpCacheEvictor;
import com.google.android.exoplayer2.upstream.cache.SimpleCache;
import com.google.android.exoplayer2.util.Util;
import com.google.android.exoplayer2.video.VideoRendererEventListener;
import com.twobigears.audio360.AudioEngine;
import com.twobigears.audio360.ChannelMap;
import com.twobigears.audio360.SpatDecoderQueue;
import com.twobigears.audio360.TBQuat;
import com.twobigears.audio360exo2.Audio360Sink;
import com.twobigears.audio360exo2.OpusRenderer;
import java.io.File;
import java.net.CookieHandler;
import java.net.CookieManager;
import java.net.CookiePolicy;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

/** Created by trevordasch on 9/19/2018. */
public class NativeVideoPlayer {
  private static final String TAG = "NativeVideoPlayer";

  static AudioEngine engine;
  static SpatDecoderQueue spat;
  static final float SAMPLE_RATE = 48000.f;

  static ExoPlayer exoPlayer;
  static AudioSink audio360Sink;
  static File downloadDirectory;
  static Cache downloadCache;
  static DatabaseProvider databaseProvider;
  static HttpDataSource.Factory httpDataSourceFactory;

  static Handler handler;

  static Object stateMutex = new Object();

  static boolean isPlaying;
  static int currentPlaybackState;
  static int stereoMode = -1;
  static int width;
  static int height;

  static long duration;

  static long lastPlaybackPosition;
  static long lastPlaybackUpdateTime;
  static float lastPlaybackSpeed;

  private static void updatePlaybackState(boolean playbackStateChanged, int playbackState) {
    synchronized (stateMutex) {
      if (playbackStateChanged) {
        isPlaying =
            (playbackState == Player.STATE_READY || playbackState == Player.STATE_BUFFERING);
        currentPlaybackState = playbackState;
      }
      duration = exoPlayer.getDuration();
      lastPlaybackPosition = exoPlayer.getCurrentPosition();
      lastPlaybackSpeed = isPlaying ? exoPlayer.getPlaybackParameters().speed : 0;
      lastPlaybackUpdateTime = System.currentTimeMillis();
      Format format = exoPlayer.getVideoFormat();
      if (format != null) {
        stereoMode = format.stereoMode;
        width = format.width;
        height = format.height;
      } else {
        stereoMode = -1;
        width = 0;
        height = 0;
      }
    }
  }

  private static Handler getHandler() {
    if (handler == null) {
      handler = new Handler(Looper.getMainLooper());
    }

    return handler;
  }

  private static class CustomRenderersFactory extends DefaultRenderersFactory {

    public CustomRenderersFactory(Context context) {
      super(context);
    }

    @Override
    public DefaultRenderersFactory setAllowedVideoJoiningTimeMs(long allowedVideoJoiningTimeMs) {
      super.setAllowedVideoJoiningTimeMs(allowedVideoJoiningTimeMs);
      return this;
    }

    @Override
    public Renderer[] createRenderers(
        Handler eventHandler,
        VideoRendererEventListener videoRendererEventListener,
        AudioRendererEventListener audioRendererEventListener,
        TextOutput textRendererOutput,
        MetadataOutput metadataRendererOutput) {

      Renderer[] renderers =
          super.createRenderers(
              eventHandler,
              videoRendererEventListener,
              audioRendererEventListener,
              textRendererOutput,
              metadataRendererOutput);

      ArrayList<Renderer> rendererList = new ArrayList<>(Arrays.asList(renderers));

      // The output latency of the engine can be used to compensate for sync
      double latency = engine.getOutputLatencyMs();

      // Audio: opus codec with the spatial audio engine
      // TBE_8_2 implies 10 channels of audio (8 channels of spatial audio, 2 channels of
      // head-locked)
      if (audio360Sink == null) {
        audio360Sink = new Audio360Sink(spat, ChannelMap.TBE_8_2, latency);
      }
      final OpusRenderer audioRenderer = new OpusRenderer(audio360Sink);

      // place our audio renderer first in the list to prioritize it
      rendererList.add(0, audioRenderer);
      Log.e(TAG, "TEMP Added OpusRenderer to renderers list!!");

      renderers = rendererList.toArray(renderers);
      return renderers;
    }
  }

  private static File getDownloadDirectory(Context context) {
    if (downloadDirectory == null) {
      downloadDirectory = context.getExternalFilesDir(null);
      if (downloadDirectory == null) {
        downloadDirectory = context.getFilesDir();
      }
    }
    return downloadDirectory;
  }

  private static synchronized Cache getDownloadCache(Context context) {
    if (downloadCache == null) {
      File downloadContentDirectory = new File(getDownloadDirectory(context), "downloads");
      downloadCache =
          new SimpleCache(
              downloadContentDirectory, new NoOpCacheEvictor(), getDatabaseProvider(context));
    }
    return downloadCache;
  }

  private static synchronized DatabaseProvider getDatabaseProvider(Context context) {
    if (databaseProvider == null) {
      databaseProvider = new StandaloneDatabaseProvider(context);
    }
    return databaseProvider;
  }

  private static CacheDataSource.Factory buildReadOnlyCacheDataSource(
      DefaultDataSource.Factory upstreamFactory, Cache cache) {
    return new CacheDataSource.Factory()
        .setCache(cache)
        .setUpstreamDataSourceFactory(upstreamFactory)
        .setCacheWriteDataSinkFactory(null)
        .setFlags(CacheDataSource.FLAG_IGNORE_CACHE_ON_ERROR);
  }

  /** Returns a {@link DataSource.Factory}. */
  public static DataSource.Factory buildDataSourceFactory(Context context) {
    DefaultDataSource.Factory upstreamFactory =
        new DefaultDataSource.Factory(context, getHttpDataSourceFactory(context));
    return buildReadOnlyCacheDataSource(upstreamFactory, getDownloadCache(context));
  }

  /** Returns a {@link HttpDataSource.Factory}. */
  public static HttpDataSource.Factory getHttpDataSourceFactory(Context context) {

    if (httpDataSourceFactory == null) {
      CookieManager cookieManager = new CookieManager();
      cookieManager.setCookiePolicy(CookiePolicy.ACCEPT_ORIGINAL_SERVER);
      CookieHandler.setDefault(cookieManager);
      httpDataSourceFactory = new DefaultHttpDataSource.Factory();
    }
    return httpDataSourceFactory;
  }

  private static List<MediaItem> buildMediaItems(Uri uri, String drmLicenseUrl) {
    List<MediaItem> mediaItems = new ArrayList<>();
    mediaItems.add(buildMediaItem(uri, drmLicenseUrl));
    return mediaItems;
  }

  private static MediaItem buildMediaItem(Uri uri, String drmLicenseUrl) {
    MediaItem.Builder builder = new MediaItem.Builder().setUri(uri);

    // Set the drm configuration if drmLicenseUrl is set
    if (drmLicenseUrl != null && drmLicenseUrl.length() > 0) {
      builder.setDrmConfiguration(
          buildDrmConfiguration(Util.getDrmUuid("widevine"), drmLicenseUrl));
    }
    return builder.build();
  }

  private static MediaItem.DrmConfiguration buildDrmConfiguration(UUID uuid, String drmLicenseUrl) {
    return new MediaItem.DrmConfiguration.Builder(uuid).setLicenseUri(drmLicenseUrl).build();
  }

  private static MediaSource.Factory buildMediaSourceFactory(
      Context context, DataSource.Factory dataSourceFactory) {
    DefaultDrmSessionManagerProvider drmSessionManagerProvider =
        new DefaultDrmSessionManagerProvider();
    drmSessionManagerProvider.setDrmHttpDataSourceFactory(getHttpDataSourceFactory(context));
    return new DefaultMediaSourceFactory(context)
        .setDataSourceFactory(dataSourceFactory)
        .setDrmSessionManagerProvider(drmSessionManagerProvider);
  }

  public static void playVideo(
      final Context context,
      final String filePath,
      final String drmLicenseUrl,
      final Surface surface) {
    // set up exoplayer on main thread
    getHandler()
        .post(
            new Runnable() {
              @Override
              public void run() {
                // Produces DataSource instances through which media data is loaded.
                DataSource.Factory dataSourceFactory = buildDataSourceFactory(context);

                Uri uri = Uri.parse(filePath);

                if (filePath.startsWith("jar:file:")) {
                  if (filePath.contains(".apk")) { // APK
                    uri =
                        new Uri.Builder()
                            .scheme("asset")
                            .path(
                                filePath.substring(
                                    filePath.indexOf("/assets/") + "/assets/".length()))
                            .build();
                  } else if (filePath.contains(".obb")) { // OBB
                    String obbPath = filePath.substring(11, filePath.indexOf(".obb") + 4);

                    StorageManager sm =
                        (StorageManager) context.getSystemService(Context.STORAGE_SERVICE);
                    if (!sm.isObbMounted(obbPath)) {
                      sm.mountObb(
                          obbPath,
                          null,
                          new OnObbStateChangeListener() {
                            @Override
                            public void onObbStateChange(String path, int state) {
                              super.onObbStateChange(path, state);
                            }
                          });
                    }

                    uri =
                        new Uri.Builder()
                            .scheme("file")
                            .path(
                                sm.getMountedObbPath(obbPath)
                                    + filePath.substring(filePath.indexOf(".obb") + 5))
                            .build();
                  }
                }

                Log.d(TAG, "Requested play of " + filePath + " uri: " + uri.toString());

                List<MediaItem> mediaItems = buildMediaItems(uri, drmLicenseUrl);

                // Create the player
                // --------------------------------------
                // - Audio Engine
                if (engine == null) {
                  engine = AudioEngine.create(SAMPLE_RATE, context);
                  spat = engine.createSpatDecoderQueue();
                  engine.start();
                }

                // --------------------------------------
                // - ExoPlayer

                // Create our modified ExoPlayer instance
                if (exoPlayer != null) {
                  exoPlayer.release();
                }

                ExoPlayer.Builder playerBuilder =
                    new ExoPlayer.Builder(context)
                        .setMediaSourceFactory(buildMediaSourceFactory(context, dataSourceFactory))
                        .setRenderersFactory(new CustomRenderersFactory(context));

                exoPlayer = playerBuilder.build();

                exoPlayer.setTrackSelectionParameters(
                    new TrackSelectionParameters.Builder(context).build());
                exoPlayer.setMediaItems(mediaItems);

                exoPlayer.addListener(
                    new Player.Listener() {

                      @Override
                      public void onPlaybackStateChanged(int playbackState) {
                        updatePlaybackState(true, playbackState);
                      }

                      @Override
                      public void onPlaybackParametersChanged(PlaybackParameters params) {
                        updatePlaybackState(false, 0);
                      }

                      @Override
                      public void onPositionDiscontinuity(int reason) {
                        updatePlaybackState(false, 0);
                      }
                    });

                exoPlayer.setVideoSurface(surface);
                exoPlayer.prepare();

                exoPlayer.setPlayWhenReady(true);
              }
            });
  }

  public static void setLooping(final boolean looping) {
    getHandler()
        .post(
            new Runnable() {
              @Override
              public void run() {
                if (exoPlayer != null) {
                  if (looping) {
                    exoPlayer.setRepeatMode(Player.REPEAT_MODE_ONE);
                  } else {
                    exoPlayer.setRepeatMode(Player.REPEAT_MODE_OFF);
                  }
                }
              }
            });
  }

  public static void stop() {
    getHandler()
        .post(
            new Runnable() {
              @Override
              public void run() {
                if (exoPlayer != null) {
                  exoPlayer.stop();
                  exoPlayer.release();
                  exoPlayer = null;
                }
                if (engine != null) {
                  engine.destroySpatDecoderQueue(spat);
                  engine.delete();
                  spat = null;
                  engine = null;
                  audio360Sink = null;
                }
              }
            });
  }

  public static void pause() {
    getHandler()
        .post(
            new Runnable() {
              @Override
              public void run() {
                if (exoPlayer != null) {
                  exoPlayer.setPlayWhenReady(false);
                }
              }
            });
  }

  public static void resume() {
    getHandler()
        .post(
            new Runnable() {
              @Override
              public void run() {
                if (exoPlayer != null) {
                  exoPlayer.setPlayWhenReady(true);
                }
              }
            });
  }

  public static void setPlaybackSpeed(final float speed) {
    getHandler()
        .post(
            new Runnable() {
              @Override
              public void run() {
                if (exoPlayer != null) {
                  PlaybackParameters param = new PlaybackParameters(speed);
                  exoPlayer.setPlaybackParameters(param);
                }
              }
            });
  }

  public static void setListenerRotationQuaternion(float x, float y, float z, float w) {
    if (engine != null) {
      engine.setListenerRotation(new TBQuat(x, y, z, w));
    }
  }

  public static boolean getIsPlaying() {
    synchronized (stateMutex) {
      return isPlaying;
    }
  }

  public static int getCurrentPlaybackState() {
    synchronized (stateMutex) {
      return currentPlaybackState;
    }
  }

  public static long getDuration() {
    synchronized (stateMutex) {
      return duration;
    }
  }

  public static int getStereoMode() {
    synchronized (stateMutex) {
      return stereoMode;
    }
  }

  public static int getWidth() {
    synchronized (stateMutex) {
      return width;
    }
  }

  public static int getHeight() {
    synchronized (stateMutex) {
      return height;
    }
  }

  public static long getPlaybackPosition() {
    synchronized (stateMutex) {
      return Math.max(
          0,
          Math.min(
              duration,
              lastPlaybackPosition
                  + (long)
                      ((System.currentTimeMillis() - lastPlaybackUpdateTime) * lastPlaybackSpeed)));
    }
  }

  public static void setPlaybackPosition(final long position) {
    getHandler()
        .post(
            new Runnable() {
              @Override
              public void run() {
                if (exoPlayer != null) {
                  Timeline timeline = exoPlayer.getCurrentTimeline();
                  if (timeline != null) {
                    int windowIndex = timeline.getFirstWindowIndex(false);
                    long windowPositionUs = position * 1000L;
                    Timeline.Window tmpWindow = new Timeline.Window();
                    for (int i = timeline.getFirstWindowIndex(false);
                        i < timeline.getLastWindowIndex(false);
                        i++) {
                      timeline.getWindow(i, tmpWindow);

                      if (tmpWindow.durationUs > windowPositionUs) {
                        break;
                      }

                      windowIndex++;
                      windowPositionUs -= tmpWindow.durationUs;
                    }

                    exoPlayer.seekTo(windowIndex, windowPositionUs / 1000L);
                  }
                }
              }
            });
  }
}
