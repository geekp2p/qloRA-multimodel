\
Param(
  [string]$ConfigPath = "configs\models.yaml",
  [string]$ArtifactsRoot = "artifacts",
  [string]$OllamaModelsRoot = "G:\ollama\models",  # แก้ได้
  [switch]$Create = $true
)

function Read-Yaml {
  param([string]$Path)
  $yaml = Get-Content -Raw -Path $Path
  $tmp = [System.IO.Path]::GetTempFileName()
  Set-Content -Path $tmp -Value $yaml
  try {
    $py = @"
import yaml, sys, json
cfg = yaml.safe_load(open(sys.argv[1],'r',encoding='utf-8'))
print(json.dumps(cfg))
"@
    $json = python - <<#PY# "$py" "$tmp" #PY#
    return $json | ConvertFrom-Json
  } finally { Remove-Item $tmp -Force }
}

try { python --version | Out-Null } catch { Write-Error "Python not found in PATH"; exit 1 }

$config = Read-Yaml -Path $ConfigPath
$models = $config.models

foreach ($alias in $models.PSObject.Properties.Name) {
  $info = $models.$alias
  $baseOllama = $info.base_ollama
  $srcAdapter = Join-Path $ArtifactsRoot (Join-Path $alias "adapter.gguf")
  if (-not (Test-Path $srcAdapter)) {
    Write-Warning "Skip $alias (adapter not found at $srcAdapter)"
    continue
  }

  $dstDir = Join-Path $OllamaModelsRoot (Join-Path "finetunes" $alias)
  New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
  Copy-Item $srcAdapter (Join-Path $dstDir "adapter.gguf") -Force

  $modelfile = @"
FROM $baseOllama
ADAPTER ./adapter.gguf

SYSTEM \"\"\"
คุณเป็นผู้ช่วยที่ตอบเป็นภาษาไทยอย่างกระชับ ชัดเจน เหมาะกับโดเมนของชุดข้อมูลที่ฝึก
\"\"\"
PARAMETER num_ctx 4096
PARAMETER temperature 0.3
"@
  $mfPath = Join-Path $dstDir "Modelfile"
  Set-Content -Path $mfPath -Value $modelfile -Encoding UTF8

  $newName = $baseOllama + "-ft"
  Write-Host "Creating Ollama model: $newName from $mfPath"
  if ($Create) {
    Push-Location $dstDir
    ollama create $newName -f .\Modelfile
    Pop-Location
  }
}
