param(
    [Parameter(Mandatory = $true)]
    [string]$Txid
)

# Fetch transaction details
$tx = Invoke-RestMethod "https://blockstream.info/testnet/api/tx/$Txid"

# Fetch live fee estimates
$feeEst = Invoke-RestMethod "https://blockstream.info/testnet/api/fee-estimates"

# Calculate vsize and actual fee rate
$actualFee = $tx.fee
$vsize = [math]::Ceiling($tx.weight / 4)
$actualRate = [math]::Round($actualFee / $vsize, 2)

# Determine confirmation target
$predictedBlocks = $null
foreach ($target in $feeEst.PSObject.Properties.Name) {
    $recommended = $feeEst.$target
    if ($actualRate -ge $recommended) {
        $predictedBlocks = $target
        break
    }
}

if ($predictedBlocks) {
    # Estimate time based on average block interval (10 minutes)
    $estMinutes = [int]$predictedBlocks * 10
    $estTime = "$predictedBlocks block(s) (~$estMinutes minutes)"
} else {
    $estTime = "⚠️ Could be delayed significantly (fee rate is very low)"
}

Write-Host "🧾 Transaction ID: $Txid"
Write-Host "💰 Fee paid: $actualFee sats"
Write-Host "📏 Virtual size: $vsize vbytes"
Write-Host "⚡ Actual fee rate: $actualRate sat/vB"
Write-Host "📡 Current fastest fee rate: $($feeEst.'1') sat/vB"
Write-Host "⏳ Estimated confirmation time: $estTime"
