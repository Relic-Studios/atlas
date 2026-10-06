; ATLAS one-file Windows installer. Built by tools/build_installer.py from a clean
; public export (no dev agents, voices, keys). Copies the app to a per-user folder,
; then runs install.ps1 (venv, torch, deps, models, shortcut) in a visible console.
#ifndef SrcDir
  #error SrcDir must be defined (path to the public export)
#endif
#ifndef AppVersion
  #define AppVersion "0.4.0"
#endif

[Setup]
AppId={{6E4C2B7A-1F3D-4C8E-9A51-ATLAS0000001}
AppName=ATLAS
AppVersion={#AppVersion}
AppPublisher=Relic Studios
DefaultDirName={localappdata}\ATLAS
DisableDirPage=no
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputBaseFilename=ATLAS-Setup-{#AppVersion}
SetupIconFile={#SrcDir}\static\favicon.ico
LicenseFile={#SrcDir}\LICENSE
UninstallDisplayIcon={app}\static\favicon.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "{#SrcDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion; Excludes: ".venv\*,__pycache__\*,logs\*,user\*,*.log,*.out,boot.pid,desktop\node_modules\*,site\*,docs\media\*,marketing\*,brand\*,tools\demo\*,.github\*"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\desktop\node_modules"

[Code]
var
  ComponentsFailed: Boolean;

{ Run install.ps1 ourselves so a failure is surfaced: the old [Run] entry ignored
  its exit code and Setup reported success on a broken install. }
procedure CurStepChanged(CurStep: TSetupStep);
var
  Rc: Integer;
  Ps: String;
begin
  if CurStep = ssPostInstall then
  begin
    WizardForm.StatusLabel.Caption := 'Installing ATLAS components (Python env, speech models)... this can take 10-20 minutes.';
    Ps := ExpandConstant('{sysnative}') + '\WindowsPowerShell\v1.0\powershell.exe';
    if not Exec(Ps, '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\install.ps1') + '"',
                ExpandConstant('{app}'), SW_SHOW, ewWaitUntilTerminated, Rc) or (Rc <> 0) then
    begin
      ComponentsFailed := True;
      SuppressibleMsgBox('ATLAS was copied, but installing its components failed.' + #13#10 +
        'Details are in ' + ExpandConstant('{app}\install.log') + '.' + #13#10#13#10 +
        'Fix the problem shown there, then double-click "Install ATLAS.cmd" in that folder to finish (finished steps are skipped).',
        mbError, MB_OK, IDOK);
    end;
  end;
end;

function GetCustomSetupExitCode: Integer;
begin
  if ComponentsFailed then Result := 1 else Result := 0;
end;

function InitializeUninstall(): Boolean;
begin
  Result := True;
  if MsgBox('Also delete your ATLAS settings, voices and personas (the "user" folder)?', mbConfirmation, MB_YESNO) = IDYES then
    DelTree(ExpandConstant('{app}\user'), True, True, True);
end;
