# VSCode
> Seamless Visual Studio Code Integration in Unity - As seen in the [Visual Studio Code documentation!](https://code.visualstudio.com/Docs/runtimes/unity)

### Requirements
##### Unity > 4.5 && Unity < 5.5
I am not sure exactly where in the 4.x cycle some of the features I'm using were introduced, but I'm guessing its around the 4.5+ mark. I've checked with the latest 4.5.0f6 release. Unity has also committed to having full support for Code when Unity 5.5 drops. So at that point all bets are off for the usefulness of this plugin. Like our MonoDevelop.Unity plugin of past, Unity catches up eventually.

#####Visual Studio Code 0.10.1+
Get the [latest version](https://code.visualstudio.com), or have a look at the past releases of the VSCode [plugin](https://github.com/dotBunny/VSCode/releases/tag/1.6.5) for support for your version.

######Mono
A good number of people have needed to install Mono in order for many of the issues with OmniSharp to be resolved.
I would suggest installing the latest package available at the [Mono Project](http://www.mono-project.com/download/). Don't worry it will not mess with Unity.

### Installation
It is important to make sure that the `VSCode.cs` file is placed under an `Editor` folder in the project. An example of this arrangement would be placing the file in `/Assets/Plugins/Editor/VSCode.cs`. By default it has its own folder structure which does this for you.

### Unity Asset Store Package
A UAS packaged version of the plugin is [available](http://u3d.as/jmM) for your consumption.

### Usage
Once the VSCode file is in place, simply navigate your way to the `Unity Preferences` and select `VSCode` and check the `Enable Integration` option.

That's it! Your ready to go!

OK, so maybe some people need a little video explaining some of the finer details of how to use the plugin. So I shot a [quick video](https://vimeo.com/dotbunny/vscode) that highlights the ups and the downs (like Unity hanging after debugging sometimes) for people to watch. Please note this video is from a previous version of the plugin where things are in the menu, this has changed since then.

### Platform Support
I use the plugin every day on a Mac (so it's battle tested there), and occasionally test it on a Windows VM. As for the recently announced Linux support, it should work just like the Mac version. I'll get around to installing the Linux editor sometime in the near future.

The Windows version of Visual Studio Code currently does not support debugging Mono, and will just throw a warning if you try to do it. The "Code" team is aware of this limitation, and we'll leave it at that.

### Automatic Update
With version 2.0 of the plugin, I've introduced a feature where it will auto-update itself if allowed. This should make things a lot easier in the future.

### Support
Please provide feedback through the GitHub [Issue](https://github.com/dotBunny/VSCode/issues) system.
