# Description
This sample shows how to capture Firebase Analytics and Crashlytics metrics in your Oculus app

# Prerequisites
## Install Firebase
1. Before enabling this sample, please follow steps 1 through 4 oulined in the article ["Add Firebase to your Unity project"](https://firebase.google.com/docs/unity/setup). Step 5 is not required as it is included in the sample code.

2. From the `firebase_unity_sdk` that you downloaded, import `dotnet4/FirebaseAnalytics.unitypackage` and `dotnet4/FirebaseCrashlytics.unitypackage`

3. Make sure to enable the Android Auto-Resolver if prompted

4. Replace the template `google-services.json` with your own

## Enable Project Code
Once Firebase Analytics and Crashlytics are added to the project, enable the sample code through the Oculus menu: `Oculus > Samples > Firebase > Enable Firebase Sample`

## Allow 'unsafe'
In order to force a crash, the Crashlytics sample makes use of `C#`'s `unsafe` keyword. This is prohibited by default, you'll have to enable it in the player settings: `Edit > Project Settings... > Player > Android settings > Allow 'unsafe' code`

## Build and Run
At this point you should be able to open the sample scene and trigger some events and crashes, which will show up in your [Firebase console](https://console.firebase.google.com/).
