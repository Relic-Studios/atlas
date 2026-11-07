; ATLAS's Windows setup program. Built by tools/build_installer.py from a clean public export plus staging
; (runtime\python: a standalone CPython 3.12; desktop\node_modules: the prebuilt Electron shell), so users need neither
; Python nor Node.js. It installs per user into %LOCALAPPDATA%\Programs\ATLAS (no admin rights) and stays one window
; from start to finish: ATLAS's dark style, a GPU check after Welcome, and the long part (install.ps1 -Progress: the
; Python environment, PyTorch, the packages, the models) run hidden, its steps, bytes and time left shown on the
; Installing page. Uninstall: Windows Settings > Apps > ATLAS, or "Uninstall ATLAS" in the Start menu.
#if Ver < EncodeVer(6,7,3)
  #error Inno Setup 6.7.3 or later is required (ExecAndLogOutput, PNG images, the dark style, background images)
#endif
#ifndef SrcDir
  #error SrcDir must be defined (path to the staged public export)
#endif
#ifndef AppVersion
  #define AppVersion "0.7.0"
#endif
#define Art SrcDir + "\installer\art"
#ifdef TEST
  #define OutName "ATLAS-Setup-" + AppVersion + "-test"
#else
  #define OutName "ATLAS-Setup-" + AppVersion
#endif

[Setup]
AppId={{6E4C2B7A-1F3D-4C8E-9A51-ATLAS0000001}
AppName=ATLAS
AppVersion={#AppVersion}
AppVerName=ATLAS {#AppVersion}
AppPublisher=Relic Studios
AppPublisherURL=https://github.com/Relic-Studios/atlas
DefaultDirName={userpf}\ATLAS
DisableWelcomePage=no
DisableDirPage=no
DisableProgramGroupPage=yes
DisableReadyPage=no
PrivilegesRequired=lowest
OutputBaseFilename={#OutName}
SetupIconFile={#Art}\setup.ico
LicenseFile={#SrcDir}\LICENSE
UninstallDisplayIcon={app}\static\favicon.ico
UninstallDisplayName=ATLAS
#ifdef QUICK
; local iteration only (build_installer.py --quick): fast to build, large
Compression=zip/1
SolidCompression=no
#else
Compression=lzma2/max
SolidCompression=yes
#endif
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupMutex=ATLASSetup
; RedirectionGuard blocks following links made by unelevated users. Setup runs unelevated and writes only where the
; user can already write, so it protects nothing here, and Windows applies it to the processes Setup starts too: pip
; failed on junctions in PATH and the model download on a Hugging Face cache with symbolic links (WinError 448), and
; ATLAS started by "Launch ATLAS" would inherit it.
RedirectionGuard=no
SetupLogging=yes
CloseApplications=no
WizardStyle=modern dark includetitlebar hidebevels
WizardBackColor=#0B0F17
WizardBackImageFile={#Art}\back_100.png,{#Art}\back_150.png,{#Art}\back_200.png
WizardImageFile=
WizardSmallImageFile={#Art}\mark_100.png,{#Art}\mark_150.png,{#Art}\mark_200.png
WizardSizePercent=130

[Messages]
SetupAppTitle=ATLAS Setup
SetupWindowTitle=%1 Setup
UninstallAppTitle=Uninstall ATLAS
UninstallAppFullTitle=Uninstall %1
WizardLicense=License
LicenseLabel=ATLAS is free and open source.
LicenseLabel3=ATLAS is free and open source under the AGPL-3.0. If you distribute a modified version, or let others use one over a network, you must offer its source code.
LicenseAccepted=I &accept the license
LicenseNotAccepted=I &do not accept the license
WizardSelectDir=Install location
SelectDirDesc=Where ATLAS and its own copy of Python go.
SelectDirLabel3=ATLAS will be installed in this folder. Your settings, voices and agents are kept here too.
SelectDirBrowseLabel=Click Next to continue, or Browse to choose another folder.
WizardSelectTasks=Shortcuts
SelectTasksDesc=How you'd like to open ATLAS.
SelectTasksLabel2=The Start menu always gets an ATLAS shortcut.
WizardReady=Ready to install
ReadyLabel1=Setup is ready to install ATLAS.
ReadyLabel2a=Click Install to start. You can keep using your PC while it runs.
ReadyLabel2b=Click Install to start. You can keep using your PC while it runs.
WizardInstalling=Installing ATLAS
InstallingLabel=This takes 20 to 30 minutes on a 100 Mbit/s connection. You can keep using your PC.
FinishedHeadingLabel=ATLAS is ready.
FinishedLabelNoIcons=ATLAS is installed.
FinishedLabel=ATLAS is installed.
ClickFinish=
ExitSetupTitle=Exit Setup
ExitSetupMessage=ATLAS isn't installed yet. You can run Setup again at any time.%n%nExit Setup?
ConfirmUninstall=Remove ATLAS from this PC? Next you choose whether to keep your data.
WizardUninstalling=Removing ATLAS
UninstallStatusLabel=This takes a few seconds.
StatusUninstalling=Removing %1...
UninstalledAll=ATLAS was removed from this PC.

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"

[Files]
; Setup-only files first: with solid compression a file Setup extracts early must come before the large ones.
Source: "{#Art}\*.png"; Flags: dontcopy noencryption
Source: "{#SrcDir}\installer\gpu_check.ps1"; Flags: dontcopy noencryption
Source: "{#SrcDir}\installer\atlas_processes.ps1"; Flags: dontcopy noencryption
; The uninstaller's background (the uninstaller can't extract temporary files).
Source: "{#Art}\back_*.png"; DestDir: "{app}\installer\art"; Flags: ignoreversion
; The app, the bundled runtime and the prebuilt desktop shell. A leading "\" anchors a pattern at the top folder, so
; a folder of the same name deep inside the runtime or node_modules is never dropped. Data folders are never written.
; Folders are excluded by name (not "name\*"), so no empty folder is left behind for them.
Source: "{#SrcDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion; Excludes: "\.venv,*.log,*.out,\boot.pid,\logs,\user,\private,\voices,\personas,\agent_state,\memory_db,\recordings,\site,\docs\media,\marketing,\brand,\tools\demo,\.github,\installer\art,\installer\*.iss,\installer\gpu_check.ps1"

[Icons]
Name: "{userprograms}\Uninstall ATLAS"; Filename: "{uninstallexe}"; IconFilename: "{app}\static\favicon.ico"; Comment: "Uninstall ATLAS"

[Run]
Filename: "{app}\desktop\node_modules\electron\dist\electron.exe"; Parameters: "."; WorkingDir: "{app}\desktop"; Description: "Launch ATLAS"; Flags: postinstall nowait skipifsilent; Check: PartTwoDone

[UninstallDelete]
; Made by install.ps1 or by running ATLAS, not by this installer (uninstall.ps1 removes them first; these are the fallback).
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\desktop\node_modules"
Type: filesandordirs; Name: "{app}\runtime"
Type: files; Name: "{app}\install*.log"

[Code]
const
  ROWS = 6;                 { row 0: part 1 (copying files); rows 1-5: install.ps1's steps }
  VK_ESCAPE = $1B;
  MB = 1048576;

var
  Styled, Upgrade, GateClosing: Boolean;
  Scale: String;
  CText, CSub, CCyan, CDim, CCard, CBorder, CErr: TColor;
  PngDone, PngTodo: TPngImage;
  PngNow: array[0..3] of TPngImage;
  { Welcome and Finished }
  WelcomeLockup, FinishedLockup: TBitmapImage;
  WelcomeTitle, GateLabel, VersionLabel: TNewStaticText;
  FinishedTitle, FinishedText: TNewStaticText;
  { inner pages }
  PageBar: array[0..4] of TPanel;
  ReadyCard: TPanel;
  ReadyKey, ReadyValue: array[0..3] of TNewStaticText;
  NeedDrives: TArrayOfString;
  NeedBytes: array of Int64;
  ShortDrives: String;
  { Installing page }
  RowIcon: array[0..5] of TBitmapImage;
  RowLabel, RowState: array[0..5] of TNewStaticText;
  DetailLabel, RightLabel: TNewStaticText;
  BarTrack, BarFill: TPanel;
  SysBar: TNewProgressBar;
  StopButton, KeepButton, StopNowButton: TNewButton;
  StopCard: TPanel;
  CurRow, Frame, Shown: Integer;
  { part 2 }
  Pid, LogFile, FailCode, FailMsg, Minutes, Detail: String;
  InPart2, Stopped, Done, WarnModels: Boolean;
  LastP: DWORD;
  TimerId: LongWord;
  FinishPaintTimerId: LongWord;
  OrigCloseQuery: TCloseQueryEvent;
  { uninstall }
  RemoveUserData: Boolean;

function SetTimer(hWnd: HWND; nIDEvent, uElapse: LongWord; lpTimerFunc: LongWord): LongWord;
  external 'SetTimer@user32.dll stdcall';
function KillTimer(hWnd: HWND; uIDEvent: LongWord): BOOL;
  external 'KillTimer@user32.dll stdcall';
function EnableWindow(hWnd: HWND; bEnable: BOOL): BOOL;
  external 'EnableWindow@user32.dll stdcall';
function GetTickCount: DWORD;
  external 'GetTickCount@kernel32.dll stdcall';
function RedrawWindow(hWnd: HWND; UpdateRect, UpdateRegion, Flags: LongWord): BOOL;
  external 'RedrawWindow@user32.dll stdcall';

{ ---- helpers ---------------------------------------------------------------------------------------------------- }

function HasParam(const Name: String): Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
    if CompareText(ParamStr(I), Name) = 0 then Result := True;
end;

{ The value of a /NAME=value parameter, or '' }
function ParamValue(const Name: String): String;
var
  I: Integer;
begin
  Result := '';
  for I := 1 to ParamCount do
    if CompareText(Copy(ParamStr(I), 1, Length(Name) + 1), Name + '=') = 0 then
      Result := Copy(ParamStr(I), Length(Name) + 2, MaxInt);
end;

function PowerShell: String;
begin
  Result := ExpandConstant('{sysnative}\WindowsPowerShell\v1.0\powershell.exe');
end;

function TitleFont: String;
begin
  if FontExists('Segoe UI Variable Display') then Result := 'Segoe UI Variable Display' else Result := 'Segoe UI';
end;

function TextFont: String;
begin
  if FontExists('Segoe UI Variable Text') then Result := 'Segoe UI Variable Text' else Result := 'Segoe UI';
end;

function ArtPath(const Name: String): String;
begin
  Result := ExpandConstant('{tmp}\') + Name + '_' + Scale + '.png';
end;

{ One image at the chosen scale. Extracted by its exact name: a wildcard would also match every .png the app itself
  installs, and with solid compression extracting those means unpacking most of the installer before the first page. }
function LoadPng(const Name: String): TPngImage;
begin
  ExtractTemporaryFile(Name + '_' + Scale + '.png');
  Result := TPngImage.Create;
  Result.LoadFromFile(ArtPath(Name));
end;

function GB(Bytes: Int64): String;
begin
  Result := Format('%d GB', [(Bytes + 512 * MB) div (1024 * MB)]);
end;

{ A label whose own color wins over the style's (StyleElements without seFont). }
function NewLabel(Parent: TWinControl; X, Y, W: Integer; const Caption: String; Size: Integer; Color: TColor): TNewStaticText;
begin
  Result := TNewStaticText.Create(WizardForm);
  Result.Parent := Parent;
  Result.Left := X;
  Result.Top := Y;
  Result.AutoSize := False;
  Result.Width := W;
  Result.WordWrap := True;
  Result.Font.Name := TextFont;
  Result.Font.Size := Size;
  if Styled then begin
    Result.Font.Color := Color;
    Result.StyleElements := Result.StyleElements - [seFont];
  end;
  Result.Caption := Caption;
  Result.AdjustHeight;
end;

{ A flat colored panel; under the style its own color wins (StyleElements empty). }
function NewPanel(Parent: TWinControl; X, Y, W, H: Integer; Color: TColor): TPanel;
begin
  Result := TPanel.Create(WizardForm);
  Result.Parent := Parent;
  Result.SetBounds(X, Y, W, H);
  Result.BevelOuter := bvNone;
  Result.Caption := '';
  if Styled then begin
    Result.ParentBackground := False;
    Result.Color := Color;
    Result.StyleElements := [];
  end;
end;

{ A card: a 1-pixel border panel around a card-colored panel; returns the inner one. }
function NewCard(Parent: TWinControl; X, Y, W, H: Integer): TPanel;
var
  Outer: TPanel;
begin
  Outer := NewPanel(Parent, X, Y, W, H, CBorder);
  Result := NewPanel(Outer, 1, 1, W - 2, H - 2, CCard);
end;

function NewImage(Parent: TWinControl; X, Y, W, H: Integer): TBitmapImage;
begin
  Result := TBitmapImage.Create(WizardForm);
  Result.Parent := Parent;
  Result.SetBounds(X, Y, W, H);
  Result.Stretch := True;
  Result.BackColor := clNone;
end;

{ ---- the GPU gate (section 2 of the design) ------------------------------------------------------------------- }

function RunGpuCheck: String;
var
  Output: TExecOutput;
  RC, I: Integer;
  Params, W: String;
begin
  Result := 'UNKNOWN no verdict';
  try
    ExtractTemporaryFile('gpu_check.ps1');
    Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{tmp}\gpu_check.ps1') + '"';
#ifdef TEST
    if ParamValue('/NVIDIASMI') <> '' then Params := Params + ' -NvidiaSmi "' + ParamValue('/NVIDIASMI') + '"';
#endif
    if ExecAndCaptureOutput(PowerShell, Params, '', SW_SHOWNORMAL, ewWaitUntilTerminated, RC, Output) then
      for I := 0 to GetArrayLength(Output.StdOut) - 1 do begin
        W := Trim(Output.StdOut[I]);
        if (Pos('OK', W) = 1) or (Pos('NOGPU', W) = 1) or (Pos('UNKNOWN', W) = 1) or (Pos('OLDDRIVER', W) = 1) or
           (Pos('OLDGPU', W) = 1) or (Pos('LOWVRAM', W) = 1) then begin
          Result := W;
          Break;
        end;
      end;
  except
    Result := 'UNKNOWN ' + GetExceptionMessage;
  end;
end;

{ Shows the verdict's message; True when Setup may go on. }
function GateAllows(const Verdict: String): Boolean;
var
  Rest, Name, Mem: String;
  K: Integer;
begin
  Result := True;
  if Pos('NOGPU', Verdict) = 1 then begin
    TaskDialogMsgBox('No NVIDIA graphics card was found',
      'ATLAS needs an NVIDIA GeForce GTX 16-series, RTX 20-series or newer with 8 GB of graphics memory. ' +
      'If you have one, install its driver from nvidia.com/drivers, restart your PC and run Setup again.',
      mbError, MB_OK, ['Close Setup'], 0);
    Result := False;
  end else if Pos('OLDDRIVER', Verdict) = 1 then begin
    TaskDialogMsgBox('Your NVIDIA driver is too old for ATLAS',
      'Update it from nvidia.com/drivers, restart your PC and run Setup again.', mbError, MB_OK, ['Close Setup'], 0);
    Result := False;
  end else if Pos('OLDGPU', Verdict) = 1 then begin
    Name := Trim(Copy(Verdict, 7, MaxInt));
    { a task dialog's heading doesn't wrap, so the card's name goes in the text }
    TaskDialogMsgBox('Your graphics card is too old for ATLAS',
      'Graphics card: ' + Name + #13#10#13#10 + 'ATLAS needs an NVIDIA GeForce GTX 16-series, RTX 20-series or newer.',
      mbError, MB_OK, ['Close Setup'], 0);
    Result := False;
  end else if Pos('LOWVRAM', Verdict) = 1 then begin
    Rest := Trim(Copy(Verdict, 8, MaxInt));
    K := Length(Rest);
    while (K > 0) and (Rest[K] <> ' ') do K := K - 1;
    Name := Trim(Copy(Rest, 1, K));
    Mem := Trim(Copy(Rest, K + 1, MaxInt));
    Result := TaskDialogMsgBox('Your graphics card has ' + Mem + ' GB of memory',
      'Graphics card: ' + Name + #13#10#13#10 + 'ATLAS needs about 8 GB of graphics memory when it uses a cloud ' +
      'language model, and 12 GB or more for a local one. Install anyway?', mbConfirmation, MB_YESNO, [
      'Don''t install', 'Install anyway'], 0) = IDNO;
    { the task dialog's default is its first button and it takes no MB_DEFBUTTON flag: the safe answer goes first }
  end;
end;

{ ---- disk space (section 3) ------------------------------------------------------------------------------------- }

procedure AddNeed(const Path: String; Bytes: Int64);
var
  D: String;
  I, N: Integer;
begin
  D := Uppercase(ExtractFileDrive(Path));
  if D = '' then Exit;
  N := GetArrayLength(NeedDrives);
  for I := 0 to N - 1 do
    if NeedDrives[I] = D then begin
      NeedBytes[I] := NeedBytes[I] + Bytes;
      Exit;
    end;
  SetArrayLength(NeedDrives, N + 1);
  SetArrayLength(NeedBytes, N + 1);
  NeedDrives[N] := D;
  NeedBytes[N] := Bytes;
end;

function HfCache: String;
begin
  Result := GetEnv('HF_HUB_CACHE');
  if Result = '' then
    if GetEnv('HF_HOME') <> '' then Result := AddBackslash(GetEnv('HF_HOME')) + 'hub'
    else Result := ExpandConstant('{%USERPROFILE}\.cache\huggingface\hub');
end;

function PipCache: String;
begin
  Result := GetEnv('PIP_CACHE_DIR');
  if Result = '' then Result := ExpandConstant('{localappdata}\pip\cache');
end;

{ Fills NeedDrives/NeedBytes and returns the per-drive lines; ShortDrives lists the drives short of space. }
function ComputeSpace: String;
var
  I: Integer;
  Extra, Free, Total: Int64;
  Line: String;
begin
  SetArrayLength(NeedDrives, 0);
  SetArrayLength(NeedBytes, 0);
  if Upgrade then begin
    { the environment, PyTorch and most models are already there }
    AddNeed(WizardDirValue, Int64(1024) * MB);
    AddNeed(HfCache, Int64(1300) * MB);
    AddNeed(PipCache, Int64(500) * MB);
    AddNeed(GetEnv('TEMP'), Int64(500) * MB);
  end else begin
    AddNeed(WizardDirValue, Int64(7168) * MB);     { measured 2026-11-06: .venv 5.6 GB + app 0.45 GB, plus room }
    AddNeed(HfCache, Int64(4000) * MB);            { the five models, 3.7 GiB, plus the Silero VAD }
    AddNeed(PipCache, Int64(3400) * MB);           { PyTorch 2.9 GB + packages 0.33 GB }
    AddNeed(GetEnv('TEMP'), Int64(3000) * MB);     { PyTorch's wheel while it downloads }
  end;
  Extra := 0;
#ifdef TEST
  Extra := Int64(StrToIntDef(ParamValue('/EXTRANEED'), 0)) * 1024 * MB;
#endif
  Result := '';
  ShortDrives := '';
  for I := 0 to GetArrayLength(NeedDrives) - 1 do begin
    NeedBytes[I] := NeedBytes[I] + Extra;
    Line := NeedDrives[I] + '  ' + GB(NeedBytes[I]) + ' needed';
    if GetSpaceOnDisk64(NeedDrives[I] + '\', Free, Total) then begin
      Line := Line + ', ' + GB(Free) + ' free';
      if Free < NeedBytes[I] then begin
        Line := Line + '  (not enough)';
        if ShortDrives <> '' then ShortDrives := ShortDrives + ' and ';
        ShortDrives := ShortDrives + NeedDrives[I];
      end;
    end;
    if Result <> '' then Result := Result + #13#10;
    Result := Result + Line;
  end;
end;

{ ---- ATLAS running (section 8) ---------------------------------------------------------------------------------- }

{ The lines atlas_processes.ps1 prints for the install folder (with Stop: the ones still running after the wait). }
function AtlasProcesses(Stop: Boolean): TArrayOfString;
var
  Output: TExecOutput;
  RC, I, N: Integer;
  Params: String;
begin
  SetArrayLength(Result, 0);
  try
    ExtractTemporaryFile('atlas_processes.ps1');
    Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{tmp}\atlas_processes.ps1') +
      '" -Folder "' + WizardDirValue + '"';
    if Stop then Params := Params + ' -Stop -WaitSec 10';
    if ExecAndCaptureOutput(PowerShell, Params, '', SW_SHOWNORMAL, ewWaitUntilTerminated, RC, Output) then
      for I := 0 to GetArrayLength(Output.StdOut) - 1 do
        if Trim(Output.StdOut[I]) <> '' then begin
          N := GetArrayLength(Result);
          SetArrayLength(Result, N + 1);
          Result[N] := Trim(Output.StdOut[I]);
        end;
  except
    Log('ATLAS process check failed: ' + GetExceptionMessage);
  end;
end;

{ ---- page bar, Welcome, Ready, Finished -------------------------------------------------------------------------- }

procedure PaintPageBar(Current: Integer);
var
  I: Integer;
begin
  for I := 0 to 4 do begin
    PageBar[I].Visible := Styled and (Current >= 0);
    if I <= Current then PageBar[I].Color := CCyan else PageBar[I].Color := CDim;
  end;
end;

procedure BuildWelcome;
var
  P: TNewNotebookPage;
  X, W, Y: Integer;
  Info: TNewStaticText;
  Mark: TPanel;
  I: Integer;
  Facts: array[0..2] of String;
begin
  P := WizardForm.WelcomePage;
  WizardForm.WizardBitmapImage.Visible := False;
  WizardForm.WelcomeLabel1.Visible := False;
  WizardForm.WelcomeLabel2.Visible := False;
  X := ScaleX(44);
  W := P.Width - 2 * X;
  Y := ScaleY(44);
  if Styled then begin
    WelcomeLockup := NewImage(P, X - ScaleX(4), Y, ScaleX(260), ScaleY(72));
    WelcomeLockup.PngImage := LoadPng('lockup');
    Y := Y + ScaleY(92);
  end else begin
    Info := NewLabel(P, X, Y, W, 'ATLAS', 24, CText);
    Info.Font.Name := TitleFont;
    Info.AdjustHeight;
    Y := Y + Info.Height + ScaleY(16);
  end;
  WelcomeTitle := NewLabel(P, X, Y, W, 'Voice agents for your calls, running on your PC.', 15, CText);
  WelcomeTitle.Font.Name := TitleFont;
  WelcomeTitle.AdjustHeight;
  Y := Y + WelcomeTitle.Height + ScaleY(22);
  Facts[0] := 'Setup installs ATLAS with its own copy of Python, PyTorch and the speech and voice models.';
  Facts[1] := 'About 7 GB to download: 20 to 30 minutes on a 100 Mbit/s connection.';
  Facts[2] := 'Needs an NVIDIA GeForce GTX 16-series, RTX 20-series or newer with 8 GB of graphics memory.';
  for I := 0 to 2 do begin
    Mark := NewPanel(P, X, Y + ScaleY(3), ScaleX(3), ScaleY(14), CCyan);
    Mark.Visible := Styled;
    Info := NewLabel(P, X + ScaleX(14), Y, W - ScaleX(14), Facts[I], 10, CSub);
    Y := Y + Info.Height + ScaleY(12);
  end;
  GateLabel := NewLabel(P, X, P.Height - ScaleY(40), W, 'Checking your graphics card' + #$2026, 10, CCyan);
  GateLabel.Visible := False;
  VersionLabel := NewLabel(P, X, P.Height - ScaleY(20), W, 'Version {#AppVersion}', 8, CDim);
end;

procedure BuildFinished;
var
  P: TNewNotebookPage;
  X, W, Y: Integer;
begin
  P := WizardForm.FinishedPage;
  WizardForm.WizardBitmapImage2.Visible := False;
  WizardForm.FinishedHeadingLabel.Visible := False;
  WizardForm.FinishedLabel.Visible := False;
  X := ScaleX(44);
  W := P.Width - 2 * X;
  Y := ScaleY(44);
  if Styled then begin
    FinishedLockup := NewImage(P, X - ScaleX(4), Y, ScaleX(260), ScaleY(72));
    FinishedLockup.PngImage := LoadPng('lockup');
    Y := Y + ScaleY(92);
  end;
  FinishedTitle := NewLabel(P, X, Y, W, 'ATLAS is ready.', 15, CText);
  FinishedTitle.Font.Name := TitleFont;
  FinishedTitle.AdjustHeight;
  FinishedText := NewLabel(P, X, Y + FinishedTitle.Height + ScaleY(14), W, '', 10, CSub);
  WizardForm.RunList.Left := X - ScaleX(2);
  WizardForm.RunList.Width := W;
end;

procedure FinishPaintProc(Wnd: HWND; Msg: UINT; IdEvent: UINT_PTR; Time: DWORD);
begin
  KillTimer(0, FinishPaintTimerId);
  FinishPaintTimerId := 0;
  if WizardForm.CurPageID <> wpFinished then Exit;
  { The styled list retains copied parent pixels when moved/resized. Avoid its
    unused blank rows and repaint after Inno has finished laying out this page. }
  if WizardForm.RunList.Items.Count > 0 then
    WizardForm.RunList.Height := ScaleY(26) * WizardForm.RunList.Items.Count;
  RedrawWindow(WizardForm.Handle, 0, 0, $0185);
end;

procedure ShowFinished;
var
  Y: Integer;
begin
  if Done then begin
    FinishedTitle.Caption := 'ATLAS is ready.';
    FinishedText.Caption := 'Choose a language model, your microphone and speakers on first launch. ' +
      'Juniper is ready with a voice; you can customize or create agents whenever you like.';
    if WarnModels then
      FinishedText.Caption := FinishedText.Caption + #13#10#13#10 +
        'Some voice files will finish downloading the first time ATLAS starts.';
  end else if Stopped then begin
    FinishedTitle.Caption := 'ATLAS isn''t finished.';
    FinishedText.Caption := 'You stopped the install. Run Setup again to finish; finished steps are skipped. ' +
      'To remove ATLAS instead, use Settings > Apps.';
  end else begin
    FinishedTitle.Caption := 'ATLAS isn''t finished.';
    FinishedText.Caption := FailMsg + ' Run Setup again to finish; finished steps are skipped.';
  end;
  FinishedText.AdjustHeight;
  Y := FinishedText.Top + FinishedText.Height + ScaleY(22);
  WizardForm.RunList.Top := Y;
  WizardForm.RunList.Visible := Done;
  if FinishPaintTimerId <> 0 then KillTimer(0, FinishPaintTimerId);
  FinishPaintTimerId := SetTimer(0, 0, 80, CreateCallback(@FinishPaintProc));
end;

procedure BuildReady;
var
  P: TNewNotebookPage;
  I, KeyW: Integer;
begin
  P := WizardForm.ReadyPage;
  WizardForm.ReadyMemo.Visible := False;
  ReadyCard := NewCard(P, 0, WizardForm.ReadyMemo.Top, P.Width, WizardForm.ReadyMemo.Height);
  KeyW := ScaleX(104);
  for I := 0 to 3 do begin
    ReadyKey[I] := NewLabel(ReadyCard, ScaleX(20), 0, KeyW, '', 9, CSub);
    ReadyValue[I] := NewLabel(ReadyCard, ScaleX(20) + KeyW, 0, ReadyCard.Width - KeyW - ScaleX(40), '', 10, CText);
  end;
end;

procedure FillReady;
var
  I, Y: Integer;
  Keys, Values: array[0..3] of String;
begin
  Keys[0] := 'Folder';
  Values[0] := WizardDirValue;
  if Upgrade then begin
    Keys[1] := 'Update';
    Values[1] := 'Keeps your Python environment, PyTorch and models, and installs what changed.';
  end else begin
    Keys[1] := 'Download';
    Values[1] := 'About 7 GB: 20 to 30 minutes on a 100 Mbit/s connection';
  end;
  Keys[2] := 'Shortcuts';
  if WizardIsTaskSelected('desktopicon') then Values[2] := 'Desktop and Start menu' else Values[2] := 'Start menu';
  Keys[3] := 'Space';
  Values[3] := ComputeSpace;
  Y := ScaleY(16);
  for I := 0 to 3 do begin
    ReadyKey[I].Caption := Keys[I];
    ReadyKey[I].Top := Y + ScaleY(2);
    ReadyKey[I].AdjustHeight;
    ReadyValue[I].Caption := Values[I];
    ReadyValue[I].Top := Y;
    ReadyValue[I].AdjustHeight;
    Y := Y + ReadyValue[I].Height + ScaleY(14);
  end;
  if Styled then
    if ShortDrives <> '' then ReadyValue[3].Font.Color := CErr else ReadyValue[3].Font.Color := CText;
  ReadyCard.Height := Y + ScaleY(2);
  ReadyCard.Parent.Height := ReadyCard.Height + 2;
end;

{ ---- the Installing page (section 6) ------------------------------------------------------------------------------ }

function RowName(I: Integer): String;
begin
  case I of
    0: Result := 'Copying ATLAS''s files';
    1: Result := 'Preparing ATLAS''s Python';
    2: Result := 'Downloading PyTorch';
    3: Result := 'Installing ATLAS''s packages';
    4: Result := 'Downloading speech and voice models';
  else
    Result := 'Finishing';
  end;
end;

procedure SetRow(I, State: Integer);   { State: 0 later, 1 in progress, 2 done }
begin
  if State = 2 then begin
    if Styled then RowIcon[I].PngImage := PngDone;
    RowState[I].Caption := 'Done';
    RowLabel[I].Font.Style := [];
    if Styled then begin RowLabel[I].Font.Color := CText; RowState[I].Font.Color := CSub; end;
  end else if State = 1 then begin
    if Styled then RowIcon[I].PngImage := PngNow[0];
    RowState[I].Caption := 'In progress';
    RowLabel[I].Font.Style := [fsBold];
    if Styled then begin RowLabel[I].Font.Color := CText; RowState[I].Font.Color := CCyan; end;
  end else begin
    if Styled then RowIcon[I].PngImage := PngTodo;
    RowState[I].Caption := '';
    RowLabel[I].Font.Style := [];
    if Styled then RowLabel[I].Font.Color := CSub;
  end;
end;

procedure GoRow(N: Integer);
var
  I: Integer;
begin
  CurRow := N;
  Frame := 0;
  for I := 0 to ROWS - 1 do
    if I < N then SetRow(I, 2) else if I = N then SetRow(I, 1) else SetRow(I, 0);
end;

procedure SetBar(Value: Integer);    { 0..1000 }
begin
  if Value < Shown then Value := Shown;   { never backwards }
  if Value > 1000 then Value := 1000;
  Shown := Value;
  if Styled then BarFill.Width := (BarTrack.Width - 2) * Value div 1000
  else SysBar.Position := Value;
end;

procedure ShowRight;
begin
  if Minutes = '' then RightLabel.Caption := IntToStr(Shown div 10) + '%'
  else if Minutes = '1' then RightLabel.Caption := IntToStr(Shown div 10) + '%   ' + #$00B7 + '   About 1 minute left'
  else RightLabel.Caption := IntToStr(Shown div 10) + '%   ' + #$00B7 + '   About ' + Minutes + ' minutes left';
end;

procedure ShowStopCard(Show: Boolean);
begin
  StopCard.Parent.Visible := Show;
  if Show then begin
    StopCard.Parent.BringToFront;
    WizardForm.ActiveControl := KeepButton;
  end;
end;

procedure StopButtonClick(Sender: TObject);
begin
  if InPart2 then ShowStopCard(True);
end;

procedure KeepButtonClick(Sender: TObject);
begin
  ShowStopCard(False);
end;

procedure StopNowButtonClick(Sender: TObject);
var
  RC: Integer;
begin
  ShowStopCard(False);
  if (Pid = '') or Stopped then Exit;
  Stopped := True;
  DetailLabel.Caption := 'Stopping' + #$2026;
  Log('Stop: ending process tree ' + Pid);
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/T /F /PID ' + Pid, '', SW_HIDE, ewWaitUntilTerminated, RC);
end;

procedure FormKeyDown(Sender: TObject; var Key: Word; Shift: TShiftState);
begin
  if (Key = VK_ESCAPE) and InPart2 then begin
    Key := 0;
    ShowStopCard(not StopCard.Parent.Visible);
  end;
end;

procedure FormCloseQuery(Sender: TObject; var CanClose: Boolean);
begin
  CanClose := False;           { the close button during part 2: ask on the page instead }
  if InPart2 then ShowStopCard(True);
end;

procedure BuildInstalling;
var
  P: TNewNotebookPage;
  I, Y, RowH, W: Integer;
  Outer: TPanel;
  Title, Text: TNewStaticText;
begin
  P := WizardForm.InstallingPage;
  W := P.Width;
  RowH := ScaleY(30);
  Y := ScaleY(6);
  for I := 0 to ROWS - 1 do begin
    RowIcon[I] := NewImage(P, ScaleX(2), Y + I * RowH, ScaleX(20), ScaleY(20));
    RowIcon[I].Visible := Styled;
    RowLabel[I] := NewLabel(P, ScaleX(34), Y + I * RowH + ScaleY(1), W - ScaleX(34) - ScaleX(110), RowName(I), 10, CSub);
    RowState[I] := NewLabel(P, W - ScaleX(104), Y + I * RowH + ScaleY(2), ScaleX(102), '', 9, CSub);
    RowState[I].Alignment := taRightJustify;
  end;
  Y := Y + ROWS * RowH + ScaleY(10);
  DetailLabel := NewLabel(P, ScaleX(2), Y, W div 2, '', 9, CSub);
  RightLabel := NewLabel(P, W div 2, Y, W div 2 - ScaleX(2), '', 9, CSub);
  RightLabel.Alignment := taRightJustify;
  Y := Y + ScaleY(24);
  if Styled then begin
    BarTrack := NewPanel(P, ScaleX(2), Y, W - ScaleX(4), ScaleY(6), CBorder);
    BarFill := NewPanel(BarTrack, 1, 1, 0, BarTrack.Height - 2, CCyan);
  end else begin
    SysBar := TNewProgressBar.Create(WizardForm);
    SysBar.Parent := P;
    SysBar.SetBounds(ScaleX(2), Y, W - ScaleX(4), ScaleY(16));
    SysBar.Min := 0;
    SysBar.Max := 1000;
  end;

  { the Stop card, over the step list }
  StopCard := NewCard(P, 0, ScaleY(2) + RowH, W, ScaleY(150));
  Outer := TPanel(StopCard.Parent);
  Title := NewLabel(StopCard, ScaleX(22), ScaleY(18), StopCard.Width - ScaleX(44), 'Stop installing?', 12, CText);
  Title.Font.Name := TitleFont;
  Title.Font.Style := [fsBold];
  Title.AdjustHeight;
  Text := NewLabel(StopCard, ScaleX(22), ScaleY(50), StopCard.Width - ScaleX(44),
    'ATLAS isn''t finished yet. You can run Setup again later; finished steps are skipped.', 10, CSub);
  StopNowButton := TNewButton.Create(WizardForm);
  StopNowButton.Parent := StopCard;
  StopNowButton.Caption := 'Stop installing';
  StopNowButton.SetBounds(StopCard.Width - ScaleX(22) - ScaleX(130), StopCard.Height - ScaleY(46), ScaleX(130), WizardForm.NextButton.Height);
  StopNowButton.OnClick := @StopNowButtonClick;
  KeepButton := TNewButton.Create(WizardForm);
  KeepButton.Parent := StopCard;
  KeepButton.Caption := 'Keep installing';
  KeepButton.SetBounds(StopNowButton.Left - ScaleX(10) - ScaleX(130), StopNowButton.Top, ScaleX(130), WizardForm.NextButton.Height);
  KeepButton.OnClick := @KeepButtonClick;
  Outer.Visible := False;

  { Screen readers list a page's labels in stacking order, newest first; restack them so they read top to bottom
    (each BringToFront puts a label first, so go from the bottom of the page up). The Stop card comes to the front
    again whenever it is shown. }
  RightLabel.BringToFront;
  DetailLabel.BringToFront;
  for I := ROWS - 1 downto 0 do begin
    RowState[I].BringToFront;
    RowLabel[I].BringToFront;
  end;

  StopButton := TNewButton.Create(WizardForm);
  StopButton.Parent := WizardForm;
  StopButton.SetBounds(WizardForm.CancelButton.Left, WizardForm.CancelButton.Top, WizardForm.CancelButton.Width, WizardForm.CancelButton.Height);
  StopButton.Caption := 'Stop';
  StopButton.OnClick := @StopButtonClick;
  StopButton.Visible := False;
end;

{ Every 500 ms while the Installing page shows: the dot pulses; a quiet part 2 hides the time left after a minute and
  says so after 10 minutes. }
procedure TimerProc(Wnd: HWND; Msg: UINT; IdEvent: UINT_PTR; Time: DWORD);
var
  Quiet: DWORD;
begin
  if Styled and (CurRow >= 0) and (CurRow < ROWS) then begin
    Frame := (Frame + 1) mod 4;
    RowIcon[CurRow].PngImage := PngNow[Frame];
  end;
  if InPart2 and not Stopped then begin
    Quiet := GetTickCount - LastP;
    if (Quiet > 60000) and (Minutes <> '') then begin
      Minutes := '';
      ShowRight;
    end;
    if Quiet > 600000 then DetailLabel.Caption := 'Still working: no progress for 10 minutes.';
  end;
end;

procedure StartTimer;
begin
  if TimerId = 0 then TimerId := SetTimer(0, 0, 500, CreateCallback(@TimerProc));
end;

procedure StopTimer;
begin
  if TimerId <> 0 then KillTimer(0, TimerId);
  TimerId := 0;
end;

{ ---- part 2 ----------------------------------------------------------------------------------------------------- }

procedure OnLine(const S: String; const Error, FirstLine: Boolean);
var
  Rest: String;
  K, O: Integer;
begin
  if Error then begin
    Log('part 2 output error: ' + S);
    Exit;
  end;
  Log(S);
  if Pos('@@ ', S) <> 1 then Exit;
  if Pos('@@ PID ', S) = 1 then begin
    Pid := Trim(Copy(S, 8, 20));
    { Inno disables the window while it waits; Stop, the close button and Esc need it (design section 6) }
    EnableWindow(WizardForm.Handle, True);
    WizardForm.CancelButton.Visible := False;
    StopButton.Visible := True;
    OrigCloseQuery := WizardForm.OnCloseQuery;
    WizardForm.OnCloseQuery := @FormCloseQuery;
    WizardForm.KeyPreview := True;
    WizardForm.OnKeyDown := @FormKeyDown;
    InPart2 := True;
    LastP := GetTickCount;
  end else if Pos('@@ LOG ', S) = 1 then
    LogFile := ExpandConstant('{app}\') + Trim(Copy(S, 8, MaxInt))
  else if Pos('@@ STEP ', S) = 1 then
    GoRow(StrToIntDef(Trim(Copy(S, 9, 3)), CurRow))
  else if Pos('@@ P ', S) = 1 then begin
    LastP := GetTickCount;
    Rest := Copy(S, 6, MaxInt);
    K := Pos('|', Rest);
    if K = 0 then Exit;
    O := StrToIntDef(Copy(Rest, 1, K - 1), 0);
    Rest := Copy(Rest, K + 1, MaxInt);
    K := Pos('|', Rest);
    if K > 0 then begin
      Detail := Copy(Rest, 1, K - 1);
      Minutes := Trim(Copy(Rest, K + 1, MaxInt));
    end else begin
      Detail := Rest;
      Minutes := '';
    end;
    if not Stopped then DetailLabel.Caption := Detail;
    SetBar(30 + O * 970 div 1000);
    ShowRight;
  end else if Pos('@@ WARN MODELS', S) = 1 then
    WarnModels := True
  else if Pos('@@ FAIL ', S) = 1 then begin
    Rest := Copy(S, 9, MaxInt);
    K := Pos('|', Rest);
    if K > 0 then begin
      FailCode := Copy(Rest, 1, K - 1);
      FailMsg := Copy(Rest, K + 1, MaxInt);
    end else
      FailCode := Rest;
  end else if Pos('@@ DONE', S) = 1 then begin
    Done := True;
    GoRow(ROWS);
    SetBar(1000);
    Minutes := '';
    ShowRight;
    DetailLabel.Caption := '';
  end;
end;

function DriveList: String;
var
  I: Integer;
begin
  Result := '';
  for I := 0 to GetArrayLength(NeedDrives) - 1 do begin
    if Result <> '' then Result := Result + ' and ';
    Result := Result + NeedDrives[I];
  end;
  if Result = '' then Result := 'your drive';
end;

{ The wording for a failure code (design section 6), from the label of the step that failed. }
function FailText: String;
var
  Lbl: String;
begin
  if (CurRow >= 1) and (CurRow < ROWS) then Lbl := RowName(CurRow) else Lbl := '';
  if FailCode = 'NODISK' then
    Result := 'There isn''t enough disk space on ' + DriveList + '. Free some space and try again.'
  else if FailCode = 'LOCKED' then
    Result := 'A file ATLAS needs is in use or blocked. Close ATLAS and any antivirus scan of the ATLAS folder, then try again.'
  else if FailCode = 'TLS' then
    Result := 'A secure connection was blocked (often a proxy or antivirus web filter). Allow downloads from ' +
      'pypi.org, download.pytorch.org and huggingface.co, then try again.'
  else if FailCode = 'SERVER' then
    Result := 'The download server isn''t responding. Try again in a few minutes.'
  else if (FailCode = 'NET') and (Lbl <> '') then
    Result := Lbl + ' failed: the connection dropped. Check your internet connection and try again.'
  else if Lbl <> '' then
    Result := Lbl + ' failed. Details are in the log.'
  else
    Result := 'Installing ATLAS''s components stopped unexpectedly. Details are in the log.';
end;

{ Runs install.ps1 once; True when it finished (@@ DONE and exit code 0). }
function RunPartTwo: Boolean;
var
  RC: Integer;
  Params: String;
begin
  Pid := '';
  Done := False;
  FailCode := '';
  FailMsg := '';
  LogFile := ExpandConstant('{app}\install.log');
#ifdef TEST
  if ParamValue('/FAKEFAIL') <> '' then begin
    GoRow(3);
    FailCode := Uppercase(ParamValue('/FAKEFAIL'));
    Result := False;
    Exit;
  end;
#endif
  Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\install.ps1') + '" -Progress -Runtime "' +
    ExpandConstant('{app}\runtime\python\python.exe') + '" -GpuChecked';
  if not WizardIsTaskSelected('desktopicon') then Params := Params + ' -NoDesktopShortcut';
#ifdef TEST
  if ParamValue('/FAKERUN') <> '' then
    Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ParamValue('/FAKERUN') + '"';
#endif
  Log('Part 2: ' + Params);
  RC := -1;
  try
    if not ExecAndLogOutput(PowerShell, Params, ExpandConstant('{app}'), SW_SHOWNORMAL, ewWaitUntilTerminated, RC, @OnLine) then
      Log('Part 2 could not start: ' + SysErrorMessage(RC));
  except
    Log('Part 2 raised: ' + GetExceptionMessage);
    RC := -1;
  end;
  InPart2 := False;
  if (Pid <> '') then begin
    WizardForm.OnCloseQuery := OrigCloseQuery;
    WizardForm.KeyPreview := False;
    WizardForm.OnKeyDown := nil;
  end;
  ShowStopCard(False);
  Log(Format('Part 2 ended: exit code %d, done=%d, stopped=%d, fail=%s', [RC, Ord(Done), Ord(Stopped), FailCode]));
  Result := Done and (RC = 0);
end;

{ The failure dialog: True for Try again. Open log keeps it open. }
function AskTryAgain: Boolean;
var
  Answer, Code: Integer;
begin
  Result := False;
  repeat
    Answer := TaskDialogMsgBox('ATLAS couldn''t finish installing', FailText, mbError, MB_YESNOCANCEL, [
      'Try again', 'Open log', 'Close'], 0);
    if Answer = IDNO then
      ShellExec('', ExpandConstant('{win}\notepad.exe'), '"' + LogFile + '"', '', SW_SHOWNORMAL, ewNoWait, Code);
  until Answer <> IDNO;
  Result := Answer = IDYES;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then begin
    GoRow(1);
    SetBar(30);
    repeat
      if RunPartTwo then Break;
      if Stopped then Break;
      FailMsg := FailText;
      Log('Part 2 failed: ' + FailMsg);
      if WizardSilent then Break;
      if not AskTryAgain then Break;
      DetailLabel.Caption := '';
    until False;
    StopTimer;
    StopButton.Visible := False;
    WizardForm.CancelButton.Visible := True;
  end;
end;

function PartTwoDone: Boolean;
begin
  Result := Done;
end;

function GetCustomSetupExitCode: Integer;
begin
  if Stopped then Result := 5
  else if not Done then Result := 1
  else Result := 0;
end;

#ifdef TEST
{ ---- test build only: /AUTOPILOT walks the pages by itself (for screenshots); see the design's "Test build" ------ }
const
  BM_CLICK = $00F5;
var
  AutoTimer: LongWord;

function PostMessage(hWnd: HWND; Msg: UINT; wParam, lParam: Longint): BOOL;
  external 'PostMessageW@user32.dll stdcall';

procedure AutoProc(Wnd: HWND; Msg: UINT; IdEvent: UINT_PTR; Time: DWORD);
begin
  if WizardForm.CurPageID = wpLicense then WizardForm.LicenseAcceptedRadio.Checked := True;
  if WizardForm.CurPageID = wpFinished then
    if WizardForm.RunList.Items.Count > 0 then WizardForm.RunList.Checked[0] := False;
  if (WizardForm.CurPageID <> wpInstalling) and WizardForm.NextButton.Enabled and WizardForm.NextButton.Visible then
    PostMessage(WizardForm.NextButton.Handle, BM_CLICK, 0, 0);
end;
#endif

{ ---- wizard events ------------------------------------------------------------------------------------------------ }

procedure InitializeWizard;
var
  I, W, X, Gap, SegW: Integer;
begin
  Styled := IsDarkInstallMode;
  if Styled then begin
    CText := StrToColor('#F3F5F7');
    CSub := StrToColor('#9AA4B2');
    CCyan := StrToColor('#22D3EE');
    CDim := StrToColor('#3A4554');
    CCard := StrToColor('#11161E');
    CBorder := StrToColor('#1E2632');
    CErr := StrToColor('#F87171');
  end else begin
    CText := clWindowText; CSub := clGrayText; CCyan := clHighlight; CDim := clGrayText;
    CCard := clWindow; CBorder := clWindowFrame; CErr := clWindowText;
  end;
  if ScaleX(100) >= 175 then Scale := '200' else if ScaleX(100) >= 125 then Scale := '150' else Scale := '100';
  PngDone := LoadPng('step_done');
  PngTodo := LoadPng('step_todo');
  for I := 0 to 3 do PngNow[I] := LoadPng('step_now' + IntToStr(I + 1));
  Upgrade := (WizardForm.PrevAppDir <> '') and FileExists(AddBackslash(WizardForm.PrevAppDir) + '.venv\Scripts\python.exe');
  CurRow := -1;

  { page titles: the display face one size up, tall enough for descenders, the description aligned under it }
  WizardForm.PageNameLabel.Font.Name := TitleFont;
  WizardForm.PageNameLabel.Font.Size := 11;
  I := ScaleY(21) - WizardForm.PageNameLabel.Height;
  if I > 0 then begin
    WizardForm.PageNameLabel.Height := WizardForm.PageNameLabel.Height + I;
    WizardForm.PageDescriptionLabel.Top := WizardForm.PageDescriptionLabel.Top + I - ScaleY(2);
  end;
  WizardForm.PageDescriptionLabel.Left := WizardForm.PageNameLabel.Left;
  WizardForm.PageDescriptionLabel.Width := WizardForm.PageNameLabel.Width;
  WizardForm.PageDescriptionLabel.Font.Name := TextFont;
  if Styled then begin
    WizardForm.PageDescriptionLabel.Font.Color := CSub;
    WizardForm.PageDescriptionLabel.StyleElements := WizardForm.PageDescriptionLabel.StyleElements - [seFont];
  end;
  { the page bar runs from the title's left edge to just before the mark in the header's corner (never under it) }
  X := WizardForm.PageNameLabel.Left;
  W := WizardForm.WizardSmallBitmapImage.Left - X - ScaleX(18);
  Gap := ScaleX(6);
  SegW := (W - 4 * Gap) div 5;
  for I := 0 to 4 do
    PageBar[I] := NewPanel(WizardForm.MainPanel, X + I * (SegW + Gap), WizardForm.MainPanel.Height - ScaleY(5), SegW, ScaleY(3), CDim);
  PaintPageBar(-1);

  BuildWelcome;
  BuildReady;
  BuildInstalling;
  BuildFinished;
#ifdef TEST
  if HasParam('/AUTOPILOT') then AutoTimer := SetTimer(0, 0, 2500, CreateCallback(@AutoProc));
#endif
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpLicense then PaintPageBar(0)
  else if CurPageID = wpSelectDir then begin
    PaintPageBar(1);
    WizardForm.DiskSpaceLabel.Caption := 'ATLAS needs about 7 GB here, plus room for its downloads and models.';
  end else if CurPageID = wpSelectTasks then PaintPageBar(2)
  else if CurPageID = wpReady then begin
    PaintPageBar(3);
    WizardForm.ReadyMemo.Visible := False;      { Inno shows its own summary memo again on each visit }
    FillReady;
  end else if CurPageID = wpInstalling then begin
    PaintPageBar(4);
    WizardForm.StatusLabel.Visible := False;
    WizardForm.FilenameLabel.Visible := False;
    WizardForm.ProgressGauge.Visible := False;
    GoRow(0);
    SetBar(0);
    ShowRight;
    StartTimer;
  end else if CurPageID = wpFinished then begin
    PaintPageBar(-1);
    ShowFinished;
  end else
    PaintPageBar(-1);
end;

procedure DeinitializeSetup;
begin
  StopTimer;
  if FinishPaintTimerId <> 0 then KillTimer(0, FinishPaintTimerId);
  FinishPaintTimerId := 0;
end;

procedure CurInstallProgressChanged(CurProgress, MaxProgress: Integer);
begin
  if MaxProgress > 0 then begin
    SetBar(30 * CurProgress div MaxProgress);
    ShowRight;
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Verdict: String;
  Running, Left: TArrayOfString;
begin
  Result := True;
  if (CurPageID = wpSelectDir) and
      (FileExists(AddBackslash(WizardDirValue) + 'AGENTS.md') or DirExists(AddBackslash(WizardDirValue) + 'dev_pack')) then begin
    SuppressibleMsgBox('Choose a separate install folder. This is the ATLAS development folder.', mbError, MB_OK, IDOK);
    Result := False;
    Exit;
  end;
  if (CurPageID = wpWelcome) and not WizardSilent and not Upgrade and not HasParam('/SKIPGPUCHECK') then begin
    GateLabel.Visible := True;
    WizardForm.NextButton.Enabled := False;
    WizardForm.Update;
    Verdict := RunGpuCheck;
    Log('GPU check: ' + Verdict);
    GateLabel.Visible := False;
    WizardForm.NextButton.Enabled := True;
    if not GateAllows(Verdict) then begin
      GateClosing := True;
      WizardForm.Close;
      Result := False;
    end;
  end else if CurPageID = wpReady then begin
    ComputeSpace;
    if ShortDrives <> '' then begin
      Log('Not enough space on ' + ShortDrives);
      if not WizardSilent then
        Result := TaskDialogMsgBox('There may not be enough space on ' + ShortDrives,
          'The Ready page lists what each drive needs. The caches may already hold part of it. Install anyway?',
          mbConfirmation, MB_YESNO, ['Go back', 'Install anyway'], 0) = IDNO;   { default: the first, safe one }
      if not Result then Exit;
    end;
    Running := AtlasProcesses(False);
    if GetArrayLength(Running) > 0 then begin
      Log('ATLAS is running: ' + Running[0]);
      if not WizardSilent then
        Result := TaskDialogMsgBox('ATLAS is running', 'Close it and continue?', mbConfirmation, MB_YESNO, [
          'Close ATLAS and continue', 'Go back'], 0) = IDYES;
      if not Result then Exit;
      Left := AtlasProcesses(True);
      if GetArrayLength(Left) > 0 then begin
        Log('ATLAS did not close: ' + Left[0]);
        if not WizardSilent then
          TaskDialogMsgBox('ATLAS didn''t close', 'Close it yourself, then click Install again.', mbError, MB_OK, [], 0);
        Result := False;
      end;
    end;
  end;
end;

procedure CancelButtonClick(CurPageID: Integer; var Cancel, Confirm: Boolean);
begin
  if GateClosing then Confirm := False;
end;

{ ---- uninstall (section 9) ---------------------------------------------------------------------------------------- }

function InitializeUninstall(): Boolean;
begin
  Result := not (FileExists(ExpandConstant('{app}\AGENTS.md')) or DirExists(ExpandConstant('{app}\dev_pack')));
  if not Result then
    SuppressibleMsgBox('Refusing to uninstall from the ATLAS development folder.', mbError, MB_OK, IDOK);
end;

procedure InitializeUninstallProgressForm;
var
  Images: TArrayOfGraphic;
  I: Integer;
  Names: array[0..2] of String;
begin
  Names[0] := '100'; Names[1] := '150'; Names[2] := '200';
  UninstallProgressForm.PageNameLabel.Font.Name := TitleFont;
  UninstallProgressForm.PageNameLabel.Font.Size := 11;
  UninstallProgressForm.PageNameLabel.Height := ScaleY(21);
  UninstallProgressForm.PageDescriptionLabel.Top := UninstallProgressForm.PageNameLabel.Top + ScaleY(21);
  UninstallProgressForm.PageDescriptionLabel.Left := UninstallProgressForm.PageNameLabel.Left;
  UninstallProgressForm.PageDescriptionLabel.Font.Name := TextFont;
  try
    SetArrayLength(Images, 3);
    for I := 0 to 2 do begin
      Images[I] := TPngImage.Create;
      Images[I].LoadFromFile(ExpandConstant('{app}\installer\art\back_') + Names[I] + '.png');
    end;
    WizardSetBackImage(Images, True, True, 255);
  except
    Log('Uninstall background image: ' + GetExceptionMessage);
  end;
end;

{ Before the app files go: ask about the data (after Windows' own confirmation), then stop ATLAS and remove the
  shortcuts that point here, the Python environment and the desktop packages (and, if asked, the data), all through
  uninstall.ps1. A silent uninstall keeps the data. }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Rc: Integer;
  Args: String;
begin
  if CurUninstallStep = usUninstall then begin
    RemoveUserData := SuppressibleTaskDialogMsgBox('Keep your ATLAS data?',
      'Your settings, voices, agents, memories and call recordings are in the ATLAS folder.', mbConfirmation, MB_YESNO, [
      'Keep my data' + #13#10 + 'They stay in the ATLAS folder for a later reinstall.',
      'Delete my data too' + #13#10 + 'Settings, voices, agents, memories and recordings are deleted.'], 0, IDYES) = IDNO;
    Args := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\uninstall.ps1') + '" -FromInstaller -Yes';
    if RemoveUserData then Args := Args + ' -RemoveData' else Args := Args + ' -KeepData';
    if FileExists(ExpandConstant('{app}\uninstall.ps1')) then begin
      if not Exec(PowerShell, Args, ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, Rc) or (Rc <> 0) then begin
        RemoveUserData := False;
        Log('ATLAS cleanup did not complete; preserving remaining data.');
      end;
    end else
      RemoveUserData := False;
  end;
  if (CurUninstallStep = usPostUninstall) and RemoveUserData then
    { Explicit Atlas data paths were handled above. Never recursively delete a user-selected folder. }
    RemoveDir(ExpandConstant('{app}'));
end;
