# Gozetimsiz kosuyu AYRIK bir surecte baslatir.
#
# Fark: bu pencereyi/terminali kapatsaniz da, oturumu kapatsaniz da surec
# devam eder. Ekran kilidi zaten sorun degil.
#
#   .\scripts\run_unattended.ps1                    # tam kosu (~9-10 saat)
#   .\scripts\run_unattended.ps1 -Flags "--list"
#   .\scripts\run_unattended.ps1 -Flags "--only speed"
#
# Ilerlemeyi izlemek icin:
#   Get-Content outputs\logs\driver.log -Wait -Tail 20
#   python scripts\run_unattended.py --list
#
# NOT: parametre adi -Flags; -Args KULLANILAMAZ, PowerShell'in yerlesik
# $Args otomatik degiskeniyle cakisiyor ve sessizce bos kaliyor.

param([string]$Flags = "")

$ErrorActionPreference = "Stop"
$root   = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\PC1\AppData\Local\Programs\Python\Python312\python.exe"
$logDir = Join-Path $root "outputs\logs"

if (-not (Test-Path $python)) { throw "Python bulunamadi: $python" }
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$out = Join-Path $logDir "driver.log"
$err = Join-Path $logDir "driver.err"

# -u : tamponsuz cikti. Olmadan log dosyasi saatlerce bos gorunur.
$argList = @("-u", "scripts\run_unattended.py")
if ($Flags) { $argList += $Flags.Split(" ") }

$p = Start-Process -FilePath $python `
                   -ArgumentList $argList `
                   -WorkingDirectory $root `
                   -RedirectStandardOutput $out `
                   -RedirectStandardError $err `
                   -WindowStyle Hidden `
                   -PassThru

"Baslatildi. PID = $($p.Id)"
"  surucu logu  : $out"
"  hata logu    : $err"
"  asama loglari: $logDir\<asama>.log"
""
"Izlemek icin  : Get-Content `"$out`" -Wait -Tail 20"
"Durum icin    : python scripts\run_unattended.py --list"
"Durdurmak icin: Stop-Process -Id $($p.Id)"
