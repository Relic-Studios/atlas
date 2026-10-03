; ATLAS one-file installer (public build). Built by tools/build_installer.py:
;   ISCC /DSrcDir=<fresh public export> /DOutDir=<dist> /DAppVersion=<x.y.z> atlas.iss
; Per-user install, no admin. Copies the app, then runs install.ps1 in a visible
; console (Python env, PyTorch, packages, Electron, speech models). No keys bundled.
#ifndef SrcDir
  #error SrcDir not defined
#endif
#ifndef OutDir
  #define OutDir "."
#endif
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{6E7A1F3C-0B4D-4C2A-9A51-ATLASVOICE01}
AppName=ATLAS
AppVersion={#AppVersion}
AppPublisher=ATLAS
DefaultDirName={localappdata}\Programs\ATLAS
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir={#OutDir}
OutputBaseFilename=ATLAS-Setup-{#AppVersion}
SetupIconFile={#SrcDir}\static\favicon.ico
UninstallDisplayIcon={app}\static\favicon.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableDirPage=no

[Messages]
FinishedLabel=ATLAS is installed. Open it from the ATLAS shortcut; the first launch walks you through your language model (local or cloud), your microphone and speakers, a voice, and your first persona.

[Files]
Source: "{#SrcDir}\*"; DestDir: "{app}"; Excludes: ".venv\*,desktop\node_modules\*,__pycache__\*,*.pyc,logs\*,*.log,*.out,boot.pid"; Flags: recursesubdirs createallsubdirs ignoreversion

[Run]
; The heavy setup (several GB of downloads) runs in a console so progress is visible.
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install.ps1"""; WorkingDir: "{app}"; StatusMsg: "Setting up ATLAS (Python, PyTorch, speech models). This can take 10-30 minutes; keep the console open..."; Flags: waituntilterminated
Filename: "{app}\desktop\node_modules\electron\dist\electron.exe"; Parameters: "."; WorkingDir: "{app}\desktop"; Description: "Launch ATLAS"; Flags: postinstall nowait skipifsilent skipifdoesntexist

[UninstallDelete]
; Created by install.ps1, not by this installer. User data (user\, voices\user\,
; personas created in the app) is kept on purpose so a reinstall keeps your agents.
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\desktop\node_modules"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\logs"
Type: files; Name: "{app}\install*.log"
Type: files; Name: "{userdesktop}\ATLAS.lnk"
Type: files; Name: "{userprograms}\ATLAS.lnk"
