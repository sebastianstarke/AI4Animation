using System.Collections.Generic;
using UnityEditor;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace Gradle
{
    public static class Configuration
    {
        private static readonly string androidPluginsFolder = "Assets/Plugins/Android/";
        private static readonly string gradleTemplatePath = androidPluginsFolder + "mainTemplate.gradle";
        private static readonly string disabledGradleTemplatePath = gradleTemplatePath + ".DISABLED";
        private static readonly string internalGradleTemplatePath = Path.Combine(Path.Combine(GetBuildToolsDirectory(BuildTarget.Android), "GradleTemplates"), "mainTemplate.gradle");

        private static readonly string gradlePropertiesPath = androidPluginsFolder + "gradleTemplate.properties";
        private static readonly string disabledGradlePropertiesPath = gradleTemplatePath + ".DISABLED";
        private static readonly string internalGradlePropertiesPath = Path.Combine(Path.Combine(GetBuildToolsDirectory(BuildTarget.Android), "GradleTemplates"), "gradleTemplate.properties");

        private static string GetBuildToolsDirectory(UnityEditor.BuildTarget bt)
        {
            return (string)(typeof(BuildPipeline).GetMethod("GetBuildToolsDirectory", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic).Invoke(null, new object[] { bt }));
        }

        public static void UseGradle()
        {
            EditorUserBuildSettings.androidBuildSystem = AndroidBuildSystem.Gradle;

            // create android plugins directory if it doesn't exist
            if (!Directory.Exists(androidPluginsFolder))
            {
                Directory.CreateDirectory(androidPluginsFolder);
            }

            if (!File.Exists(gradleTemplatePath))
            {
                if (File.Exists(gradleTemplatePath + ".DISABLED"))
                {
                    File.Move(disabledGradleTemplatePath, gradleTemplatePath);
                    File.Move(disabledGradleTemplatePath + ".meta", gradleTemplatePath + ".meta");
                }
                else
                {
                    File.Copy(internalGradleTemplatePath, gradleTemplatePath);
                }
                AssetDatabase.ImportAsset(gradleTemplatePath);
            }

            if (!File.Exists(gradlePropertiesPath))
            {
                if (File.Exists(gradlePropertiesPath + ".DISABLED"))
                {
                    File.Move(disabledGradlePropertiesPath, gradlePropertiesPath);
                    File.Move(disabledGradlePropertiesPath + ".meta", gradlePropertiesPath + ".meta");
                }
                else
                {
                    File.Copy(internalGradlePropertiesPath, gradlePropertiesPath);
                }
                AssetDatabase.ImportAsset(gradlePropertiesPath);
            }
        }

        public static bool IsUsingGradle()
        {
            return EditorUserBuildSettings.androidBuildSystem == AndroidBuildSystem.Gradle
                && Directory.Exists(androidPluginsFolder)
                && File.Exists(gradleTemplatePath)
                && File.Exists(gradlePropertiesPath);
        }

        public static Template OpenTemplate()
        {
            return IsUsingGradle() ? new Template(gradleTemplatePath) : null;
        }

        public static Properties OpenProperties()
        {
            return IsUsingGradle() ? new Properties(gradlePropertiesPath) : null;
        }
    }

    public class Template
    {
        private static class Parsing
        {
            public static string GetVersion(string text)
            {
                return new System.Text.RegularExpressions.Regex("com.android.tools.build:gradle:([0-9]+\\.[0-9]+\\.[0-9]+)").Match(text).Groups[1].Value;
            }
            public static int GoToSection(string section, List<string> lines)
            {
                return GoToSection(section, 0, lines);
            }

            public static int GoToSection(string section, int start, List<string> lines)
            {
                var sections = section.Split('.');

                int p = start - 1;
                for (int i = 0; i < sections.Length; i++)
                {
                    p = FindInScope("\\s*" + sections[i] + "\\s*\\{\\s*", p + 1, lines);
                }

                return p;
            }

            public static int FindInScope(string search, int start, List<string> lines)
            {
                var regex = new System.Text.RegularExpressions.Regex(search);

                int depth = 0;

                for (int i = start; i < lines.Count; i++)
                {
                    if (depth == 0 && regex.IsMatch(lines[i]))
                    {
                        return i;
                    }

                    // count the number of open and close braces. If we leave the current scope, break
                    if (lines[i].Contains("{"))
                    {
                        depth++;
                    }
                    if (lines[i].Contains("}"))
                    {
                        depth--;
                    }
                    if (depth < 0)
                    {
                        break;
                    }
                }
                return -1;
            }

            public static int GetScopeEnd(int start, List<string> lines)
            {
                int depth = 0;
                for (int i = start; i < lines.Count; i++)
                {
                    // count the number of open and close braces. If we leave the current scope, break
                    if (lines[i].Contains("{"))
                    {
                        depth++;
                    }
                    if (lines[i].Contains("}"))
                    {
                        depth--;
                    }
                    if (depth < 0)
                    {
                        return i;
                    }
                }

                return -1;
            }
        }

        private readonly string _templatePath;
        private readonly List<string> _lines;

        internal Template(string templatePath)
        {
            _templatePath = templatePath;
            _lines = File.ReadAllLines(_templatePath).ToList();
        }

        public void Save()
        {
            File.WriteAllLines(_templatePath, _lines);
        }

        public void AddRepository(string section, string name)
        {
            int sectionIndex = Parsing.GoToSection($"{section}.repositories", _lines);
            if (Parsing.FindInScope($"{name}\\(\\)", sectionIndex + 1, _lines) == -1)
            {
                _lines.Insert(Parsing.GetScopeEnd(sectionIndex + 1, _lines), $"\t\t{name}()");
            }
        }

        public void AddDependency(string name, string version)
        {
            int dependencies = Parsing.GoToSection("dependencies", _lines);
            int target = Parsing.FindInScope(Regex.Escape(name), dependencies + 1, _lines);
            if (target != -1)
            {
                _lines[target] = $"\timplementation '{name}:{version}'";
            }
            else
            {
                _lines.Insert(Parsing.GetScopeEnd(dependencies + 1, _lines), $"\timplementation '{name}:{version}'");
            }
        }

        public void RemoveDependency(string name)
        {
            int dependencies = Parsing.GoToSection("dependencies", _lines);
            int target = Parsing.FindInScope(Regex.Escape(name), dependencies + 1, _lines);
            if (target != -1)
            {
                _lines.RemoveAt(target);
            }
        }

        public void RemoveAndroidSetting(string name)
        {
            int android = Parsing.GoToSection("android", _lines);
            int target = Parsing.FindInScope(Regex.Escape(name), android + 1, _lines);
            if (target != -1)
            {
                _lines.RemoveAt(target);
            }
        }
    }

    public class Properties
    {
        private readonly string _propertiesPath;
        private readonly List<string> _lines;

        internal Properties(string propertiesPath)
        {
            _propertiesPath = propertiesPath;
            _lines = File.ReadAllLines(_propertiesPath).ToList();
        }

        public void Save()
        {
            File.WriteAllLines(_propertiesPath, _lines);
        }

        private int FindProperty(string name)
        {
            int p = -1;
            string propStr = name + "=";
            for (int i = 0; i < _lines.Count; i++)
            {
                if (_lines[i].StartsWith(propStr))
                {
                    p = i;
                    break;
                }
            }

            return p;
        }

        public void SetProperty(string name, string value)
        {
            int line = FindProperty(name);
            if (line == -1)
            {
                _lines.Add($"{name}={value}");
            }
            else
            {
                _lines[line] = $"{name}={value}";
            }
        }

        public void RemoveProperty(string name)
        {
            int line = FindProperty(name);
            if (line != -1)
            {
                _lines.RemoveAt(line);
            }
        }

        public bool TryGetProperty(string name, out string value)
        {
            int line = FindProperty(name);
            if (line != -1)
            {
                value = _lines[line].Split('=')[1];
                return true;
            }
            else
            {
                value = null;
                return false;
            }
        }
    }
}
