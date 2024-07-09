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

using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System;

public class DirectorySyncer
{
	public delegate void SyncResultDelegate(SyncResult syncResult);

	public readonly string Source;
	public readonly string Target;
	public SyncResultDelegate WillPerformOperations;
	private readonly Regex _ignoreExpression;

	// helper classes to simplify transition beyond .NET runtime 3.5
	public abstract class CancellationToken
	{
		protected abstract bool _IsCancellationRequested();

		public virtual bool IsCancellationRequested
		{
			get { return _IsCancellationRequested(); }
		}

		public void ThrowIfCancellationRequested()
		{
			if (IsCancellationRequested)
			{
				throw new Exception("Operation Cancelled");
			}
		}

		public static readonly CancellationToken None = new CancellationTokenNone();

		private class CancellationTokenNone : CancellationToken
		{
			protected override bool _IsCancellationRequested()
			{
				return false;
			}
		}
	}

	public class CancellationTokenSource : CancellationToken
	{
		private bool _isCancelled;

		protected override bool _IsCancellationRequested()
		{
			return _isCancelled;
		}

		public void Cancel()
		{
			_isCancelled = true;
		}

		public CancellationToken Token
		{
			get { return this; }
		}
	}

	private static string EnsureTrailingDirectorySeparator(string path)
	{
		return path.EndsWith("" + Path.DirectorySeparatorChar)
			? path
			: path + Path.DirectorySeparatorChar;
	}

	private static string CheckedDirectory(string nameInExceptionText, string directory)
	{
		directory = Path.GetFullPath(directory);
		if (!Directory.Exists(directory))
		{
			throw new ArgumentException(string.Format("{0} is not a valid directory for argument ${1}", directory,
				nameInExceptionText));
		}

		return EnsureTrailingDirectorySeparator(directory);
	}

	public DirectorySyncer(string source, string target, string ignoreRegExPattern = null)
	{
		Source = CheckedDirectory("source", source);
		Target = CheckedDirectory("target", target);
		if (Source.StartsWith(Target, StringComparison.OrdinalIgnoreCase) ||
			Target.StartsWith(Source, StringComparison.OrdinalIgnoreCase))
		{
			throw new ArgumentException(string.Format("Paths must not contain each other (source: {0}, target: {1}",
				Source, Target));
		}

		ignoreRegExPattern = ignoreRegExPattern ?? "^$";
		_ignoreExpression = new Regex(ignoreRegExPattern, RegexOptions.IgnoreCase);
	}

	public class SyncResult
	{
		public readonly IEnumerable<string> Created;
		public readonly IEnumerable<string> Updated;
		public readonly IEnumerable<string> Deleted;

		public SyncResult(IEnumerable<string> created, IEnumerable<string> updated, IEnumerable<string> deleted)
		{
			Created = created;
			Updated = updated;
			Deleted = deleted;
		}
	}

	public bool RelativeFilePathIsRelevant(string relativeFilename)
	{
		return !_ignoreExpression.IsMatch(relativeFilename);
	}

	public bool RelativeDirectoryPathIsRelevant(string relativeDirName)
	{
		// Since our ignore patterns look at file names, they may contain trailing path separators
		// In order for paths to match those rules, we add a path separator here
		return !_ignoreExpression.IsMatch(EnsureTrailingDirectorySeparator(relativeDirName));
	}

	private HashSet<string> RelevantRelativeFilesBeneathDirectory(string path, CancellationToken cancellationToken)
	{
		return new HashSet<string>(Directory.GetFiles(path, "*", SearchOption.AllDirectories)
			.TakeWhile((s) => !cancellationToken.IsCancellationRequested)
			.Select(p => PathHelper.MakeRelativePath(path, p)).Where(RelativeFilePathIsRelevant));
	}

	private HashSet<string> RelevantRelativeDirectoriesBeneathDirectory(string path,
		CancellationToken cancellationToken)
	{
		return new HashSet<string>(Directory.GetDirectories(path, "*", SearchOption.AllDirectories)
			.TakeWhile((s) => !cancellationToken.IsCancellationRequested)
			.Select(p => PathHelper.MakeRelativePath(path, p)).Where(RelativeDirectoryPathIsRelevant));
	}

	public SyncResult Synchronize()
	{
		return Synchronize(CancellationToken.None);
	}

	private void DeleteOutdatedFilesFromTarget(SyncResult syncResult, CancellationToken cancellationToken)
	{
		var outdatedFiles = syncResult.Updated.Union(syncResult.Deleted);
		foreach (var fileName in outdatedFiles)
		{
			File.Delete(Path.Combine(Target, fileName));
			cancellationToken.ThrowIfCancellationRequested();
		}
	}

	[SuppressMessage("ReSharper", "ParameterTypeCanBeEnumerable.Local")]
	private void DeleteOutdatedEmptyDirectoriesFromTarget(HashSet<string> sourceDirs, HashSet<string> targetDirs,
		CancellationToken cancellationToken)
	{
		var deleted = targetDirs.Except(sourceDirs).OrderByDescending(s => s);

		// By sorting in descending order above, we delete leaf-first,
		// this is simpler than collapsing the list above (which would also allow us to run these ops in parallel).
		// Assumption is that there are few empty folders to delete
		foreach (var dir in deleted)
		{
			Directory.Delete(Path.Combine(Target, dir));
			cancellationToken.ThrowIfCancellationRequested();
		}
	}

	[SuppressMessage("ReSharper", "ParameterTypeCanBeEnumerable.Local")]
	private void CreateRelevantDirectoriesAtTarget(HashSet<string> sourceDirs, HashSet<string> targetDirs,
		CancellationToken cancellationToken)
	{
		var created = sourceDirs.Except(targetDirs);
		foreach (var dir in created)
		{
			Directory.CreateDirectory(Path.Combine(Target, dir));
			cancellationToken.ThrowIfCancellationRequested();
		}
	}

	private void MoveRelevantFilesToTarget(SyncResult syncResult, CancellationToken cancellationToken)
	{
		// step 3: we move all new files to target
		var newFiles = syncResult.Created.Union(syncResult.Updated);
		foreach (var fileName in newFiles)
		{
			var sourceFileName = Path.Combine(Source, fileName);
			var destFileName = Path.Combine(Target, fileName);
			// target directory exists due to step CreateRelevantDirectoriesAtTarget()
			File.Move(sourceFileName, destFileName);
			cancellationToken.ThrowIfCancellationRequested();
		}
	}

	public SyncResult Synchronize(CancellationToken cancellationToken)
	{
		var sourceDirs = RelevantRelativeDirectoriesBeneathDirectory(Source, cancellationToken);
		var targetDirs = RelevantRelativeDirectoriesBeneathDirectory(Target, cancellationToken);
		var sourceFiles = RelevantRelativeFilesBeneathDirectory(Source, cancellationToken);
		var targetFiles = RelevantRelativeFilesBeneathDirectory(Target, cancellationToken);

		var created = sourceFiles.Except(targetFiles).OrderBy(s => s).ToList();
		var updated = sourceFiles.Intersect(targetFiles).OrderBy(s => s).ToList();
		var deleted = targetFiles.Except(sourceFiles).OrderBy(s => s).ToList();
		var syncResult = new SyncResult(created, updated, deleted);

		if (WillPerformOperations != null)
		{
			WillPerformOperations.Invoke(syncResult);
		}

		DeleteOutdatedFilesFromTarget(syncResult, cancellationToken);
		DeleteOutdatedEmptyDirectoriesFromTarget(sourceDirs, targetDirs, cancellationToken);
		CreateRelevantDirectoriesAtTarget(sourceDirs, targetDirs, cancellationToken);
		MoveRelevantFilesToTarget(syncResult, cancellationToken);

		return syncResult;
	}
}
