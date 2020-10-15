# About the Analytics Package

This Analytics package supports the following Unity Analytics features:

* [Standard Events](https://docs.unity3d.com/Manual/UnityAnalyticsStandardEvents.html)
* [Analytics Event Tracker](https://docs.unity3d.com/Manual/class-AnalyticsEventTracker.html)
* [Unity Analytics Data Privacy Plug-in](https://docs.unity3d.com/Manual/UnityAnalyticsDataPrivacy.html)

For instructions on using the features in the Analytics package, refer to the [Analytics section of
the Unity Manual](https://docs.unity3d.com/Manual/UnityAnalytics.html).

The package is supported by Unity 2018.3+ and includes functionality previously included in
earlier Unity Asset Store and Package Manager packages. When upgrading existing projects to
2018.3 or later, older, redundant packages should be removed from the project.


## Installing the Analytics Package

The Analytics package is built into the Unity Editor and enabled automatically. Use the Unity
Package Manager (menu: **Window** > **Package Manager**) to disable or enable the package.
The Analytics package is listed under the built-in packages.


<a name="UsingAnalytics"></a>
## Using the Analytics Package

For instructions on using the features in the Analytics package, refer to the Unity Manual:

* [Standard Events](https://docs.unity3d.com/Manual/UnityAnalyticsStandardEvents.html)
* [Analytics Event Tracker](https://docs.unity3d.com/Manual/class-AnalyticsEventTracker.html)
* [Unity Analytics Data Privacy Plug-in](https://docs.unity3d.com/Manual/UnityAnalyticsDataPrivacy.html)


## Package contents

The following table indicates the major classes, components, and files included in the Analytics package:

|Item|Description|
|---|---|
|[`AnalyticsEvent` class](https://docs.unity3d.com/2018.3/Documentation/ScriptReference/Analytics.AnalyticsEvent.html) | The primary class for sending Standard and Custom analytics events to the Unity Analytics service.|
|[Analytics Event Tracker component](https://docs.unity3d.com/Manual/class-AnalyticsEventTracker.html) | A Unity component that you can use to send Standard and Custom analytics events (without writing code).|
|[DataPrivacy class](https://docs.unity3d.com/Manual/UnityAnalyticsDataPrivacyAPI.html)| A utility class that helps applications using Unity Analytics comply with the EU General Data Protection Regulation (GDPR).|
|`Packages/Analytics Library/DataPrivacy/DataPrivacyButton`| A Prefab GameObject you can use when building a user interface to allow players to opt out of Analytics data collection.|
|`Packages/Analytics Library/DataPrivacy/DataPrivacyIcon`| An icon graphic you can use when creating your own opt-out button or control.|


## Document revision history

|Date|Reason|
|---|---|
|October 5, 2018|Document created. Matches package version 3.2.0.|
