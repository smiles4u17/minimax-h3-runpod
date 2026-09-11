param(
  [Parameter(Mandatory=$true)][string]$InputTextPath,
  [Parameter(Mandatory=$true)][string]$OutputPath,
  [string]$Voice = "",
  [int]$Rate = 0,
  [int]$Volume = 100
)

Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($Voice -ne "") {
  $speaker.SelectVoice($Voice)
}
$speaker.Rate = [Math]::Max(-10, [Math]::Min(10, $Rate))
$speaker.Volume = [Math]::Max(0, [Math]::Min(100, $Volume))
$speaker.SetOutputToWaveFile($OutputPath)
$text = [System.IO.File]::ReadAllText($InputTextPath, [System.Text.Encoding]::UTF8)
$speaker.Speak($text)
$speaker.Dispose()
