<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Barzin Financial | V4 Intelligence</title>
    <style>
        body { background: #0d1117; color: #c9d1d9; font-family: -apple-system, system-ui, sans-serif; padding: 40px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 30px; max-width: 500px; margin: auto; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        h1 { color: #58a6ff; text-align: center; margin-bottom: 5px; }
        .subtitle { text-align: center; color: #8b949e; font-size: 0.85em; margin-bottom: 25px; border-bottom: 1px solid #30363d; padding-bottom: 15px; }
        .v4-score { font-size: 4em; text-align: center; color: #3fb950; margin: 20px 0; font-weight: 800; }
        input { width: 100%; padding: 14px; background: #0d1117; border: 1px solid #30363d; color: white; border-radius: 6px; box-sizing: border-box; font-size: 1.1em; text-align: center; }
        button { width: 100%; padding: 15px; margin-top: 15px; background: #238636; border: none; color: white; cursor: pointer; border-radius: 6px; font-weight: bold; font-size: 1.1em; }
        button:hover { background: #2ea043; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; font-size: 0.95em; border-bottom: 1px ridge #30363d; padding-bottom: 5px; }
        .verdict { background: #21262d; padding: 12px; border-radius: 6px; text-align: center; margin-bottom: 15px; font-weight: bold; color: #58a6ff; border: 1px solid #30363d; }
    </style>
</head>
<body>
    <div class="card">
        <h1>V4 Intelligence Engine</h1>
        <div class="subtitle">Building the Bridge Between Truth and Data</div>
        <input type="text" id="ticker" value="AMD">
        <button onclick="runV4()">Analyze Asymmetry</button>
        <div id="results">
            <div style="text-align:center; margin-top:20px; color: #8b949e; font-size: 0.7em; letter-spacing: 2px;">V4 ASYMMETRY SCORE</div>
            <div class="v4-score" id="scoreValue">--</div>
            <div id="details"></div>
        </div>
    </div>

    <script>
        async function runV4() {
            const ticker = document.getElementById('ticker').value.toUpperCase();
            const apiKey = 'zKqpakTDzCz3u53CFuoCUq3nHIKpazOR';
            const scoreValue = document.getElementById('scoreValue');
            const details = document.getElementById('details');
            
            scoreValue.innerText = "...";
            details.innerText = "Connecting to Financial Core...";

            try {
                // Using a more reliable proxy for the live demo
                const url = `https://financialmodelingprep.com/api/v3/key-metrics-ttm/${ticker}?limit=1&apikey=${apiKey}`;
                const proxyUrl = `https://api.allorigins.win/get?url=${encodeURIComponent(url)}`;
                
                const response = await fetch(proxyUrl);
                const wrapper = await response.json();
                const data = JSON.parse(wrapper.contents);

                if (data && data.length > 0) {
                    const roe = data[0].returnOnEquityTTM || 0;
                    const netMargin = data[0].netProfitMarginTTM || 0;
                    
                    // Simple, clean V4 Logic
                    let score = (netMargin * 20) + (roe * 10);
                    score = Math.min(Math.max(score, 4.2), 9.8).toFixed(1);

                    scoreValue.innerText = score + "/10";
                    scoreValue.style.color = score > 7.5 ? "#3fb950" : "#d29922";
                    
                    details.innerHTML = `
                        <div class="verdict">${score > 7.5 ? 'STRATEGIC ARCHITECT' : 'SYSTEMIC COG'}</div>
                        <div class="metric"><span>Net Profit Margin:</span> <strong>${(netMargin * 100).toFixed(2)}%</strong></div>
                        <div class="metric"><span>Return on Equity:</span> <strong>${(roe * 100).toFixed(2)}%</strong></div>
                        <p style="font-size: 0.75em; color: #8b949e; text-align: center; margin-top: 15px;">Verified Financial Essence for ${ticker}</p>
                    `;
                } else {
                    throw new Error();
                }
            } catch (e) {
                scoreValue.innerText = "!";
                details.innerHTML = "<span style='color:#f85149'>Connection Lag. Please refresh and try again.</span>";
            }
        }
    </script>
</body>
</html>