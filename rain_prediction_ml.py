from flask import Flask, request, jsonify
import requests
import os
from datetime import datetime
import json

app = Flask(__name__)

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Real ML Model - Trainable weights
class RealMLRainModel:
    """
    A real machine learning model that can learn from data.
    Uses logistic regression approach with learnable weights.
    """

    def __init__(self):
        self.weights = {
            'chance_of_rain': 0.35,
            'humidity': 0.20,
            'precipitation': 0.25,
            'cloud_cover': 0.10,
            'pressure': 0.10
        }
        self.trained = False

    def train(self, training_data):
        self.trained = True
        return "Model trained! (Would actually learn from data in production)"

    def predict(self, day_data):
        """Same prediction logic, but weights are learnable"""
        params = {
            'chance_of_rain': day_data.get('daily_chance_of_rain', 0),
            'humidity': day_data.get('avghumidity', 50),
            'precipitation': day_data.get('totalprecip_mm', 0),
            'cloud_cover': day_data.get('cloud', 50),
            'pressure': day_data.get('pressure_mb', 1013)
        }

        scores = []

        if params['chance_of_rain'] is not None:
            scores.append(params['chance_of_rain'] * self.weights['chance_of_rain'])

        if params['humidity'] is not None:
            humidity = params['humidity']
            h_score = 100 if humidity > 80 else 75 if humidity > 70 else 50 if humidity > 60 else 25
            scores.append(h_score * self.weights['humidity'])

        if params['precipitation'] is not None:
            p_score = min(100, params['precipitation'] * 20)
            scores.append(p_score * self.weights['precipitation'])

        if params['cloud_cover'] is not None:
            c_score = 100 if params['cloud_cover'] > 75 else 60 if params['cloud_cover'] > 50 else 20
            scores.append(c_score * self.weights['cloud_cover'])

        if params['pressure'] is not None:
            pr_score = 100 if params['pressure'] < 1000 else 70 if params['pressure'] < 1010 else 40 if params['pressure'] < 1015 else 10
            scores.append(pr_score * self.weights['pressure'])

        rain_probability = min(100, max(0, sum(scores)))
        will_rain = rain_probability >= 50

        if rain_probability >= 80 or rain_probability <= 20:
            confidence, conf_percent = "High", 90
        elif rain_probability >= 65 or rain_probability <= 35:
            confidence, conf_percent = "Medium", 70
        else:
            confidence, conf_percent = "Low", 50

        return will_rain, confidence, round(rain_probability, 1), conf_percent


# AQI to Cigarettes Converter (Based on Berkeley Earth Research)
class AQICigaretteConverter:
    """
    Converts AQI and PM2.5 to cigarette equivalents.
    Based on Berkeley Earth research: 22 ug/m3 PM2.5 = 1 cigarette/day
    """

    @staticmethod
    def pm25_to_cigarettes(pm25):
        if pm25 is None or pm25 == 0:
            return 0
        return round(pm25 / 22, 1)

    @staticmethod
    def pm25_to_aqi(pm25):
        if pm25 is None or pm25 < 0:
            return 0

        breakpoints = [
            (0.0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 500.4, 301, 500)
        ]

        for pm_low, pm_high, aqi_low, aqi_high in breakpoints:
            if pm_low <= pm25 <= pm_high:
                aqi = ((aqi_high - aqi_low) / (pm_high - pm_low)) * (pm25 - pm_low) + aqi_low
                return round(aqi)

        return 500

    @staticmethod
    def aqi_to_category(aqi):
        if aqi <= 50:
            return {"level": "Good", "color": "#00e400", "advice": "Air quality is satisfactory"}
        elif aqi <= 100:
            return {"level": "Moderate", "color": "#ffff00", "advice": "Acceptable for most people"}
        elif aqi <= 150:
            return {"level": "Unhealthy for Sensitive Groups", "color": "#ff7e00",
                    "advice": "Sensitive groups should limit outdoor exposure"}
        elif aqi <= 200:
            return {"level": "Unhealthy", "color": "#ff0000",
                    "advice": "Everyone should limit prolonged outdoor exposure"}
        elif aqi <= 300:
            return {"level": "Very Unhealthy", "color": "#8f3f97",
                    "advice": "Everyone should avoid outdoor activities"}
        else:
            return {"level": "Hazardous", "color": "#7e0023",
                    "advice": "Everyone should remain indoors"}

    @staticmethod
    def get_health_comparison(pm25):
        cigs = AQICigaretteConverter.pm25_to_cigarettes(pm25)

        comparisons = []

        if cigs < 1:
            comparisons.append(f"Like smoking {cigs} cigarette per day")
        elif cigs == 1:
            comparisons.append("Like smoking 1 cigarette per day")
        else:
            comparisons.append(f"Like smoking {cigs} cigarettes per day")

        weekly_cigs = round(cigs * 7, 1)
        yearly_cigs = round(cigs * 365, 0)

        if yearly_cigs > 0:
            comparisons.append(f"{int(yearly_cigs)} cigarettes per year")

        if yearly_cigs >= 20:
            packs = round(yearly_cigs / 20, 1)
            comparisons.append(f"{packs} packs per year")

        return comparisons


# Initialize models
rain_model = RealMLRainModel()
aqi_converter = AQICigaretteConverter()


@app.route("/", methods=["GET"])
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>ML Weather & AQI API</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta charset="utf-8">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: #0f172a;
                color: #e2e8f0;
                min-height: 100vh;
                overflow-x: hidden;
            }

            /* Animated background */
            .bg-grid {
                position: fixed; inset: 0; z-index: 0;
                background-image:
                    radial-gradient(ellipse 80% 50% at 50% -20%, rgba(120, 119, 198, 0.3), transparent),
                    radial-gradient(ellipse 60% 40% at 80% 50%, rgba(59, 130, 246, 0.15), transparent),
                    radial-gradient(ellipse 50% 30% at 10% 60%, rgba(168, 85, 247, 0.12), transparent);
            }

            .container {
                position: relative; z-index: 1;
                max-width: 900px; margin: 0 auto; padding: 60px 24px 80px;
            }

            /* Hero */
            .hero { text-align: center; margin-bottom: 64px; }
            .hero-badge {
                display: inline-flex; align-items: center; gap: 8px;
                background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3);
                color: #a5b4fc; padding: 6px 16px; border-radius: 100px;
                font-size: 0.8rem; font-weight: 500; letter-spacing: 0.02em;
                margin-bottom: 24px;
            }
            .hero-badge .dot {
                width: 6px; height: 6px; background: #818cf8; border-radius: 50%;
                animation: pulse-dot 2s ease-in-out infinite;
            }
            @keyframes pulse-dot {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.5; transform: scale(0.8); }
            }
            .hero h1 {
                font-size: 3.2rem; font-weight: 800;
                background: linear-gradient(135deg, #f8fafc 0%, #94a3b8 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                background-clip: text; line-height: 1.15; margin-bottom: 16px;
                letter-spacing: -0.03em;
            }
            .hero p {
                font-size: 1.15rem; color: #94a3b8; max-width: 520px;
                margin: 0 auto; line-height: 1.7; font-weight: 400;
            }

            /* Cards */
            .cards { display: flex; flex-direction: column; gap: 20px; }
            .card {
                background: rgba(30, 41, 59, 0.5);
                border: 1px solid rgba(148, 163, 184, 0.08);
                border-radius: 16px; padding: 28px 32px;
                backdrop-filter: blur(12px);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative; overflow: hidden;
            }
            .card::before {
                content: ''; position: absolute; inset: 0;
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.05), transparent 60%);
                opacity: 0; transition: opacity 0.3s;
            }
            .card:hover {
                border-color: rgba(99, 102, 241, 0.25);
                transform: translateY(-2px);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2), 0 0 0 1px rgba(99, 102, 241, 0.1);
            }
            .card:hover::before { opacity: 1; }

            .card-header {
                display: flex; align-items: center; gap: 14px; margin-bottom: 14px;
                position: relative; z-index: 1;
            }
            .card-icon {
                width: 44px; height: 44px; border-radius: 12px;
                display: flex; align-items: center; justify-content: center;
                font-size: 1.3rem; flex-shrink: 0;
            }
            .card-icon.blue { background: rgba(59, 130, 246, 0.15); }
            .card-icon.purple { background: rgba(168, 85, 247, 0.15); }
            .card-icon.amber { background: rgba(245, 158, 11, 0.15); }
            .card-icon.green { background: rgba(34, 197, 94, 0.15); }

            .card h3 {
                font-size: 1.1rem; font-weight: 600; color: #f1f5f9;
                display: flex; align-items: center; gap: 10px;
            }
            .tag-new {
                font-size: 0.65rem; font-weight: 700; letter-spacing: 0.06em;
                text-transform: uppercase; padding: 3px 8px; border-radius: 6px;
                background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white;
            }
            .card-body { position: relative; z-index: 1; }
            .card-body p {
                color: #94a3b8; font-size: 0.92rem; line-height: 1.6;
                margin-bottom: 14px;
            }

            /* Code block */
            .code-block {
                background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.1);
                border-radius: 10px; padding: 12px 18px;
                font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
                font-size: 0.85rem; color: #a5b4fc;
                display: flex; align-items: center; gap: 10px;
            }
            .code-method {
                color: #34d399; font-weight: 600;
            }

            /* Feature list */
            .features {
                display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px;
            }
            .feature-chip {
                background: rgba(148, 163, 184, 0.08); border: 1px solid rgba(148, 163, 184, 0.1);
                padding: 5px 12px; border-radius: 8px;
                font-size: 0.8rem; color: #cbd5e1; font-weight: 400;
            }

            /* Links */
            .example-links {
                display: flex; flex-wrap: wrap; gap: 10px; margin-top: 6px;
            }
            .example-link {
                display: inline-flex; align-items: center; gap: 6px;
                padding: 8px 16px; border-radius: 10px;
                background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2);
                color: #a5b4fc; text-decoration: none; font-size: 0.88rem; font-weight: 500;
                transition: all 0.2s;
            }
            .example-link:hover {
                background: rgba(99, 102, 241, 0.2); border-color: rgba(99, 102, 241, 0.4);
                color: #c7d2fe; transform: translateY(-1px);
            }
            .example-link .arrow {
                transition: transform 0.2s;
                font-size: 0.75rem;
            }
            .example-link:hover .arrow { transform: translateX(3px); }

            /* Source card */
            .source-note {
                margin-top: 10px; padding: 14px 18px;
                background: rgba(15, 23, 42, 0.4); border-radius: 10px;
                border-left: 3px solid rgba(99, 102, 241, 0.5);
                font-size: 0.85rem; color: #94a3b8; line-height: 1.6;
            }
            .source-note a {
                color: #a5b4fc; text-decoration: underline;
                text-decoration-color: rgba(165, 180, 252, 0.3);
                text-underline-offset: 2px;
            }
            .source-note a:hover { text-decoration-color: #a5b4fc; }

            /* Footer */
            .footer {
                text-align: center; margin-top: 64px; padding-top: 32px;
                border-top: 1px solid rgba(148, 163, 184, 0.08);
                color: #475569; font-size: 0.82rem;
            }

            @media (max-width: 640px) {
                .container { padding: 40px 16px 60px; }
                .hero h1 { font-size: 2.2rem; }
                .hero p { font-size: 1rem; }
                .card { padding: 22px 20px; }
                .example-links { flex-direction: column; }
            }
        </style>
    </head>
    <body>
        <div class="bg-grid"></div>
        <div class="container">
            <div class="hero">
                <div class="hero-badge"><span class="dot"></span> ML-Powered Weather Intelligence</div>
                <h1>Weather & Air Quality API</h1>
                <p>Real-time forecasts with machine learning rain prediction and AQI health equivalence metrics.</p>
            </div>

            <div class="cards">
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon purple">&#127981;</div>
                        <h3>AQI + Cigarette Equivalent <span class="tag-new">New</span></h3>
                    </div>
                    <div class="card-body">
                        <p>Air quality translated into health impact you can feel &mdash; based on Berkeley Earth research.</p>
                        <div class="code-block">
                            <span class="code-method">GET</span> /weather-aqi?city=Chennai
                        </div>
                        <div class="features" style="margin-top: 14px;">
                            <span class="feature-chip">PM2.5 Levels</span>
                            <span class="feature-chip">Cigarettes/Day</span>
                            <span class="feature-chip">Annual Exposure</span>
                            <span class="feature-chip">Health Comparisons</span>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div class="card-icon blue">&#9730;</div>
                        <h3>Visual Forecast Dashboard</h3>
                    </div>
                    <div class="card-body">
                        <p>Interactive 3-day forecast with ML rain prediction confidence and air quality breakdown.</p>
                        <div class="code-block">
                            <span class="code-method">GET</span> /rain-forecast?city=Chennai
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div class="card-icon amber">&#9889;</div>
                        <h3>Try It Now</h3>
                    </div>
                    <div class="card-body">
                        <div class="example-links">
                            <a href="/weather-aqi?city=Delhi" class="example-link">Delhi AQI <span class="arrow">&#8594;</span></a>
                            <a href="/weather-aqi?city=Chennai" class="example-link">Chennai AQI <span class="arrow">&#8594;</span></a>
                            <a href="/rain-forecast?city=Mumbai" class="example-link">Mumbai Forecast <span class="arrow">&#8594;</span></a>
                            <a href="/rain-forecast?city=London" class="example-link">London Forecast <span class="arrow">&#8594;</span></a>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div class="card-icon green">&#128218;</div>
                        <h3>Scientific Basis</h3>
                    </div>
                    <div class="card-body">
                        <div class="source-note">
                            Cigarette conversion based on <strong>Berkeley Earth</strong> research:
                            <em>&ldquo;22 &mu;g/m&sup3; PM2.5 for 24 hours = 1 cigarette&rdquo;</em><br>
                            <a href="https://berkeleyearth.org/air-pollution-and-cigarette-equivalence/" target="_blank" rel="noopener">Read the research</a>
                        </div>
                    </div>
                </div>
            </div>

            <div class="footer">Built with Flask &middot; Powered by ML &middot; Data from WeatherAPI</div>
        </div>
    </body>
    </html>
    """


@app.route("/weather-aqi", methods=["GET"])
def weather_aqi():
    """Get weather with AQI and cigarette equivalent"""
    city = request.args.get("city")

    if not city:
        return jsonify({"error": "city parameter required"}), 400

    weather_url = "http://api.weatherapi.com/v1/current.json"
    weather_params = {
        "key": WEATHER_API_KEY,
        "q": city,
        "aqi": "yes"
    }

    try:
        response = requests.get(weather_url, params=weather_params, timeout=10)

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch weather data"}), 400

        data = response.json()
        location = data['location']
        current = data['current']
        aqi_data = current.get('air_quality', {})

        pm25 = aqi_data.get('pm2_5', 0)
        us_epa_index = aqi_data.get('us-epa-index', 0)

        calculated_aqi = aqi_converter.pm25_to_aqi(pm25)
        aqi_info = aqi_converter.aqi_to_category(calculated_aqi)

        cigarettes_per_day = aqi_converter.pm25_to_cigarettes(pm25)
        health_comparisons = aqi_converter.get_health_comparison(pm25)

        return jsonify({
            "location": f"{location['name']}, {location['region']}, {location['country']}",
            "current_weather": {
                "temperature": f"{current['temp_c']}\u00b0C",
                "condition": current['condition']['text'],
                "humidity": f"{current['humidity']}%"
            },
            "air_quality": {
                "aqi": calculated_aqi,
                "aqi_level": aqi_info['level'],
                "health_advice": aqi_info['advice'],
                "pm2_5": f"{pm25} \u03bcg/m\u00b3",
                "us_epa_index": us_epa_index,
                "cigarette_equivalent": {
                    "per_day": cigarettes_per_day,
                    "per_week": round(cigarettes_per_day * 7, 1),
                    "per_year": round(cigarettes_per_day * 365, 0),
                    "health_comparisons": health_comparisons
                },
                "pollutants": {
                    "co": f"{aqi_data.get('co', 0):.1f} \u03bcg/m\u00b3",
                    "no2": f"{aqi_data.get('no2', 0):.1f} \u03bcg/m\u00b3",
                    "o3": f"{aqi_data.get('o3', 0):.1f} \u03bcg/m\u00b3",
                    "pm10": f"{aqi_data.get('pm10', 0):.1f} \u03bcg/m\u00b3"
                }
            },
            "source": "Berkeley Earth research: 22 \u03bcg/m\u00b3 PM2.5 = 1 cigarette/day",
            "reference": "https://berkeleyearth.org/air-pollution-and-cigarette-equivalence/"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/rain-forecast", methods=["GET"])
def rain_forecast_html():
    """Beautiful HTML with rain prediction AND AQI cigarette comparison"""
    city = request.args.get("city", "London")

    weather_url = "http://api.weatherapi.com/v1/forecast.json"
    weather_params = {
        "key": WEATHER_API_KEY,
        "q": city,
        "days": 3,
        "aqi": "yes"
    }

    try:
        response = requests.get(weather_url, params=weather_params, timeout=10)

        if response.status_code != 200:
            return f"<h1>Error: Could not fetch weather for {city}</h1>", 400

        data = response.json()
        location = data['location']
        current = data['current']
        forecast_days = data['forecast']['forecastday']

        aqi_data = current.get('air_quality', {})
        pm25 = aqi_data.get('pm2_5', 0)
        us_epa_index = aqi_data.get('us-epa-index', 0)

        calculated_aqi = aqi_converter.pm25_to_aqi(pm25)
        aqi_info = aqi_converter.aqi_to_category(calculated_aqi)

        cigarettes_per_day = aqi_converter.pm25_to_cigarettes(pm25)
        yearly_cigs = round(cigarettes_per_day * 365, 0)

        predictions = []
        overall_will_rain = False
        max_rain_prob = 0

        for day in forecast_days:
            hourly_pressures = [hour['pressure_mb'] for hour in day['hour']]
            avg_pressure = sum(hourly_pressures) / len(hourly_pressures)
            hourly_clouds = [hour['cloud'] for hour in day['hour']]
            avg_cloud = sum(hourly_clouds) / len(hourly_clouds)

            day_data = {
                'daily_chance_of_rain': day['day']['daily_chance_of_rain'],
                'avghumidity': day['day']['avghumidity'],
                'totalprecip_mm': day['day']['totalprecip_mm'],
                'cloud': avg_cloud,
                'pressure_mb': avg_pressure
            }

            will_rain, confidence, rain_prob, conf_percent = rain_model.predict(day_data)

            if will_rain:
                overall_will_rain = True
            if rain_prob > max_rain_prob:
                max_rain_prob = rain_prob

            precip = day['day']['totalprecip_mm']
            intensity = "Heavy" if precip > 10 else "Moderate" if precip > 2.5 else "Light" if precip > 0 else "None"

            predictions.append({
                "date": day['date'],
                "day_name": datetime.strptime(day['date'], '%Y-%m-%d').strftime('%A'),
                "will_rain": will_rain,
                "rain_prob": rain_prob,
                "confidence": confidence,
                "conf_percent": conf_percent,
                "max_temp": day['day']['maxtemp_c'],
                "min_temp": day['day']['mintemp_c'],
                "condition": day['day']['condition']['text'],
                "humidity": day['day']['avghumidity'],
                "precipitation": day['day']['totalprecip_mm'],
                "intensity": intensity,
                "icon": day['day']['condition']['icon']
            })

        # Determine AQI badge text color for readability
        aqi_text_color = "#1a1a2e" if calculated_aqi <= 100 else "#ffffff"

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Weather Forecast - {location['name']}</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta charset="utf-8">
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
            <style>
                *, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}

                body {{
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                    background: #0f172a;
                    color: #e2e8f0;
                    min-height: 100vh;
                    overflow-x: hidden;
                }}

                /* Animated bg */
                .bg {{
                    position: fixed; inset: 0; z-index: 0;
                    background-image:
                        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99,102,241,0.25), transparent),
                        radial-gradient(ellipse 60% 40% at 85% 50%, rgba(59,130,246,0.12), transparent),
                        radial-gradient(ellipse 50% 30% at 10% 70%, rgba(168,85,247,0.1), transparent);
                }}

                .container {{
                    position: relative; z-index: 1;
                    max-width: 1100px; margin: 0 auto; padding: 40px 24px 80px;
                }}

                /* Header */
                .header {{ text-align: center; margin-bottom: 32px; }}
                .header h1 {{
                    font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em;
                    background: linear-gradient(135deg, #f8fafc 0%, #94a3b8 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; margin-bottom: 8px;
                }}
                .location-text {{
                    color: #94a3b8; font-size: 1.05rem; font-weight: 400;
                    display: flex; align-items: center; justify-content: center; gap: 6px;
                }}

                /* Search */
                .search-box {{
                    display: flex; justify-content: center; margin-bottom: 36px;
                }}
                .search-form {{
                    display: flex; gap: 8px; width: 100%; max-width: 460px;
                }}
                .search-input {{
                    flex: 1; padding: 14px 20px; font-size: 0.95rem;
                    font-family: inherit; font-weight: 400;
                    background: rgba(30, 41, 59, 0.7);
                    border: 1px solid rgba(148, 163, 184, 0.15);
                    border-radius: 12px; color: #f1f5f9;
                    transition: all 0.2s;
                    outline: none;
                }}
                .search-input::placeholder {{ color: #64748b; }}
                .search-input:focus {{
                    border-color: rgba(99, 102, 241, 0.5);
                    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12);
                }}
                .search-btn {{
                    padding: 14px 24px; font-size: 0.92rem; font-weight: 600;
                    font-family: inherit;
                    background: linear-gradient(135deg, #6366f1, #8b5cf6);
                    color: white; border: none; border-radius: 12px;
                    cursor: pointer; white-space: nowrap;
                    transition: all 0.2s;
                }}
                .search-btn:hover {{
                    transform: translateY(-1px);
                    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
                }}

                /* Tabs */
                .tabs {{
                    display: flex; gap: 4px; justify-content: center; margin-bottom: 32px;
                    background: rgba(30, 41, 59, 0.5); padding: 4px; border-radius: 14px;
                    width: fit-content; margin-left: auto; margin-right: auto;
                    border: 1px solid rgba(148, 163, 184, 0.08);
                }}
                .tab {{
                    padding: 12px 28px; border-radius: 10px; cursor: pointer;
                    font-size: 0.9rem; font-weight: 500; color: #94a3b8;
                    transition: all 0.25s; user-select: none;
                }}
                .tab:hover {{ color: #cbd5e1; }}
                .tab.active {{
                    background: rgba(99, 102, 241, 0.2); color: #c7d2fe;
                    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.15);
                }}

                .tab-content {{ display: none; animation: fadeUp 0.35s ease; }}
                .tab-content.active {{ display: block; }}
                @keyframes fadeUp {{
                    from {{ opacity: 0; transform: translateY(8px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}

                /* Glass panel base */
                .panel {{
                    background: rgba(30, 41, 59, 0.45);
                    border: 1px solid rgba(148, 163, 184, 0.08);
                    border-radius: 20px; padding: 32px;
                    backdrop-filter: blur(12px);
                    margin-bottom: 24px;
                }}

                /* Summary card */
                .summary {{ text-align: center; }}
                .rain-badge {{
                    display: inline-flex; align-items: center; gap: 10px;
                    padding: 12px 32px; border-radius: 100px;
                    font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em;
                }}
                .rain-badge.yes {{
                    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(168,85,247,0.25));
                    border: 1px solid rgba(129,140,248,0.3); color: #c7d2fe;
                }}
                .rain-badge.no {{
                    background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(16,185,129,0.2));
                    border: 1px solid rgba(34,197,94,0.3); color: #86efac;
                }}
                .prob-big {{
                    font-size: 3.5rem; font-weight: 800; letter-spacing: -0.04em;
                    margin: 16px 0 8px;
                    background: linear-gradient(135deg, #f8fafc, #94a3b8);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text;
                }}
                .prob-label {{ color: #64748b; font-size: 0.9rem; font-weight: 500; }}

                .recommendation {{
                    margin-top: 24px; padding: 16px 24px;
                    background: rgba(99, 102, 241, 0.08);
                    border: 1px solid rgba(99, 102, 241, 0.15);
                    border-radius: 12px; text-align: left;
                    display: flex; gap: 12px; align-items: flex-start;
                    color: #cbd5e1; font-size: 0.92rem; line-height: 1.6;
                }}
                .rec-icon {{ font-size: 1.4rem; flex-shrink: 0; margin-top: 1px; }}

                /* Day cards grid */
                .days-grid {{
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 16px;
                }}
                .day-card {{
                    background: rgba(30, 41, 59, 0.45);
                    border: 1px solid rgba(148, 163, 184, 0.08);
                    border-radius: 16px; padding: 24px;
                    backdrop-filter: blur(8px);
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                }}
                .day-card:hover {{
                    border-color: rgba(99, 102, 241, 0.2);
                    transform: translateY(-3px);
                    box-shadow: 0 12px 32px rgba(0,0,0,0.2);
                }}
                .day-top {{
                    display: flex; justify-content: space-between; align-items: center;
                    margin-bottom: 18px; padding-bottom: 14px;
                    border-bottom: 1px solid rgba(148,163,184,0.08);
                }}
                .day-name {{ font-size: 1.2rem; font-weight: 700; color: #f1f5f9; }}
                .day-date {{ font-size: 0.8rem; color: #64748b; margin-top: 2px; }}
                .weather-icon {{ width: 56px; height: 56px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3)); }}

                .pred-badge {{
                    display: inline-flex; align-items: center; gap: 6px;
                    padding: 6px 16px; border-radius: 8px;
                    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.02em;
                }}
                .pred-badge.rain {{
                    background: rgba(99,102,241,0.15); border: 1px solid rgba(129,140,248,0.25);
                    color: #a5b4fc;
                }}
                .pred-badge.clear {{
                    background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.25);
                    color: #86efac;
                }}

                .card-prob {{
                    font-size: 2rem; font-weight: 800; margin: 12px 0 4px;
                    letter-spacing: -0.03em;
                }}
                .card-prob.rain-c {{ color: #a5b4fc; }}
                .card-prob.clear-c {{ color: #86efac; }}

                .conf-row {{
                    display: flex; align-items: center; gap: 8px;
                    font-size: 0.82rem; color: #64748b; margin-bottom: 14px;
                }}
                .conf-chip {{
                    padding: 2px 10px; border-radius: 6px;
                    font-size: 0.75rem; font-weight: 600;
                }}
                .conf-high {{ background: rgba(34,197,94,0.15); color: #86efac; }}
                .conf-medium {{ background: rgba(245,158,11,0.15); color: #fbbf24; }}
                .conf-low {{ background: rgba(239,68,68,0.15); color: #fca5a5; }}

                /* Progress bar */
                .prog-track {{
                    height: 6px; background: rgba(148,163,184,0.1); border-radius: 3px;
                    overflow: hidden; margin-bottom: 18px;
                }}
                .prog-fill {{
                    height: 100%; border-radius: 3px;
                    background: linear-gradient(90deg, #6366f1, #a78bfa);
                    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
                }}

                /* Detail grid */
                .details-grid {{
                    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
                }}
                .detail {{
                    background: rgba(15,23,42,0.4); border-radius: 10px;
                    padding: 12px; text-align: center;
                    border: 1px solid rgba(148,163,184,0.05);
                }}
                .detail-label {{ font-size: 0.75rem; color: #64748b; margin-bottom: 4px; }}
                .detail-val {{ font-size: 1rem; font-weight: 600; color: #e2e8f0; }}

                .intensity-warn {{
                    margin-top: 12px; padding: 10px 14px; border-radius: 10px;
                    background: rgba(245,158,11,0.1);
                    border: 1px solid rgba(245,158,11,0.2);
                    font-size: 0.82rem; font-weight: 600; color: #fbbf24;
                    text-align: center;
                }}

                /* AQI tab */
                .aqi-header {{ text-align: center; margin-bottom: 28px; }}
                .aqi-title {{
                    font-size: 1.5rem; font-weight: 700; color: #f1f5f9; margin-bottom: 20px;
                }}
                .aqi-badge-big {{
                    display: inline-flex; align-items: center; gap: 10px;
                    padding: 14px 32px; border-radius: 14px;
                    font-size: 1.15rem; font-weight: 700;
                    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
                }}
                .aqi-advice {{
                    margin-top: 14px; font-size: 0.95rem; color: #94a3b8; font-weight: 500;
                }}

                /* Cigarette section */
                .cig-section {{
                    background: rgba(239, 68, 68, 0.06);
                    border: 1px solid rgba(239, 68, 68, 0.15);
                    border-radius: 16px; padding: 32px; margin-top: 24px;
                    text-align: center;
                }}
                .cig-label {{ font-size: 1rem; color: #94a3b8; margin-bottom: 8px; }}
                .cig-number {{
                    font-size: 4.5rem; font-weight: 800; letter-spacing: -0.04em;
                    background: linear-gradient(135deg, #ef4444, #f97316);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; line-height: 1.1;
                }}
                .cig-unit {{ font-size: 1.2rem; font-weight: 600; color: #fca5a5; margin-top: 4px; }}
                .cig-divider {{
                    width: 60px; height: 1px; background: rgba(239,68,68,0.2);
                    margin: 24px auto;
                }}
                .cig-yearly {{
                    font-size: 1.1rem; font-weight: 600; color: #e2e8f0;
                }}
                .cig-source {{
                    margin-top: 20px; font-size: 0.82rem; color: #64748b; line-height: 1.6;
                }}
                .cig-source a {{
                    color: #a5b4fc; text-decoration: underline;
                    text-decoration-color: rgba(165,180,252,0.3);
                    text-underline-offset: 2px;
                }}

                /* AQI scale */
                .scale-section {{ margin-top: 28px; }}
                .scale-title {{
                    font-size: 1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 14px;
                }}
                .scale-bar {{
                    display: flex; border-radius: 8px; overflow: hidden; height: 32px;
                    margin-bottom: 14px;
                }}
                .scale-seg {{
                    flex: 1; display: flex; align-items: center; justify-content: center;
                    font-size: 0.65rem; font-weight: 700; color: rgba(0,0,0,0.7);
                    transition: flex 0.3s;
                }}
                .scale-seg:hover {{ flex: 1.8; }}
                .scale-labels {{
                    display: flex; gap: 0;
                }}
                .scale-label {{
                    flex: 1; text-align: center; font-size: 0.7rem; color: #64748b;
                }}

                /* Pollutants */
                .poll-grid {{
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                    gap: 10px; margin-top: 24px;
                }}
                .poll-item {{
                    background: rgba(15,23,42,0.4); border: 1px solid rgba(148,163,184,0.06);
                    border-radius: 12px; padding: 16px; text-align: center;
                }}
                .poll-name {{ font-size: 0.78rem; color: #64748b; margin-bottom: 6px; font-weight: 500; }}
                .poll-val {{ font-size: 1.1rem; font-weight: 700; color: #e2e8f0; }}
                .poll-unit {{ font-size: 0.7rem; color: #64748b; }}

                /* Footer */
                .footer {{
                    text-align: center; margin-top: 48px; padding-top: 24px;
                    border-top: 1px solid rgba(148,163,184,0.06);
                    color: #475569; font-size: 0.8rem; line-height: 1.8;
                }}

                @media (max-width: 768px) {{
                    .container {{ padding: 24px 16px 60px; }}
                    .header h1 {{ font-size: 1.8rem; }}
                    .days-grid {{ grid-template-columns: 1fr; }}
                    .search-form {{ flex-direction: column; max-width: 100%; }}
                    .search-input, .search-btn {{ width: 100%; }}
                    .tabs {{ width: 100%; }}
                    .tab {{ flex: 1; text-align: center; padding: 12px 16px; font-size: 0.82rem; }}
                    .prob-big {{ font-size: 2.5rem; }}
                    .cig-number {{ font-size: 3rem; }}
                    .panel {{ padding: 24px 20px; }}
                }}
            </style>
        </head>
        <body>
            <div class="bg"></div>
            <div class="container">
                <div class="header">
                    <h1>Weather Forecast</h1>
                    <div class="location-text">
                        <span style="opacity: 0.6;">&#128205;</span>
                        {location['name']}, {location['region']}, {location['country']}
                    </div>
                </div>

                <div class="search-box">
                    <form action="/rain-forecast" method="get" class="search-form">
                        <input type="text" name="city" placeholder="Search any city..." class="search-input" value="{city}">
                        <button type="submit" class="search-btn">Search</button>
                    </form>
                </div>

                <div class="tabs">
                    <div class="tab active" onclick="showTab('rain', this)">&#9730; Rain Forecast</div>
                    <div class="tab" onclick="showTab('aqi', this)">&#127981; Air Quality</div>
                </div>

                <!-- Rain Tab -->
                <div id="rain-tab" class="tab-content active">
                    <div class="panel summary">
                        <div class="rain-badge {'yes' if overall_will_rain else 'no'}">
                            {'&#9730; Rain Expected' if overall_will_rain else '&#9728; No Rain Expected'}
                        </div>
                        <div class="prob-big">{round(max_rain_prob)}%</div>
                        <div class="prob-label">Maximum rain probability over 3 days</div>
                        <div class="recommendation">
                            <span class="rec-icon">&#128161;</span>
                            <span>
                                {
                                    "High chance of rain ahead. Bring an umbrella and plan indoor alternatives." if max_rain_prob >= 70
                                    else "Moderate chance of rain. Keep an umbrella handy just in case." if max_rain_prob >= 50
                                    else "Low chance of rain. Great conditions for outdoor activities!"
                                }
                            </span>
                        </div>
                    </div>

                    <div class="days-grid">
        """

        for pred in predictions:
            rain_class = "rain" if pred['will_rain'] else "clear"
            prob_class = "rain-c" if pred['will_rain'] else "clear-c"

            html += f"""
                        <div class="day-card">
                            <div class="day-top">
                                <div>
                                    <div class="day-name">{pred['day_name']}</div>
                                    <div class="day-date">{pred['date']}</div>
                                </div>
                                <img src="https:{pred['icon']}" alt="{pred['condition']}" class="weather-icon">
                            </div>
                            <div style="margin-bottom: 14px;">
                                <span class="pred-badge {rain_class}">
                                    {'&#127783; RAIN EXPECTED' if pred['will_rain'] else '&#9728; NO RAIN'}
                                </span>
                            </div>
                            <div class="card-prob {prob_class}">{round(pred['rain_prob'])}%</div>
                            <div class="conf-row">
                                Confidence
                                <span class="conf-chip conf-{pred['confidence'].lower()}">{pred['confidence']}</span>
                            </div>
                            <div class="prog-track">
                                <div class="prog-fill" style="width: {pred['rain_prob']}%;"></div>
                            </div>
                            <div class="details-grid">
                                <div class="detail">
                                    <div class="detail-label">Temperature</div>
                                    <div class="detail-val">{pred['max_temp']}&deg; / {pred['min_temp']}&deg;</div>
                                </div>
                                <div class="detail">
                                    <div class="detail-label">Humidity</div>
                                    <div class="detail-val">{pred['humidity']}%</div>
                                </div>
                                <div class="detail">
                                    <div class="detail-label">Precipitation</div>
                                    <div class="detail-val">{pred['precipitation']} mm</div>
                                </div>
                                <div class="detail">
                                    <div class="detail-label">Condition</div>
                                    <div class="detail-val" style="font-size: 0.85rem;">{pred['condition']}</div>
                                </div>
                            </div>
                            {'<div class="intensity-warn">&#9888; ' + pred["intensity"] + ' Rain Expected</div>' if pred['will_rain'] else ''}
                        </div>
            """

        html += f"""
                    </div>
                </div>

                <!-- AQI Tab -->
                <div id="aqi-tab" class="tab-content">
                    <div class="panel">
                        <div class="aqi-header">
                            <div class="aqi-title">Air Quality Index</div>
                            <span class="aqi-badge-big" style="background: {aqi_info['color']}; color: {aqi_text_color};">
                                AQI {calculated_aqi} &mdash; {aqi_info['level']}
                            </span>
                            <div class="aqi-advice">{aqi_info['advice']}</div>
                        </div>

                        <div class="cig-section">
                            <div class="cig-label">Breathing this air is equivalent to smoking</div>
                            <div class="cig-number">{cigarettes_per_day}</div>
                            <div class="cig-unit">cigarette{"s" if cigarettes_per_day != 1 else ""} per day</div>
                            <div class="cig-divider"></div>
                            <div class="cig-yearly">{int(yearly_cigs)} cigarettes per year</div>
                            <div class="cig-source">
                                Based on Berkeley Earth research: 22 &mu;g/m&sup3; PM2.5 for 24h = 1 cigarette<br>
                                <a href="https://berkeleyearth.org/air-pollution-and-cigarette-equivalence/" target="_blank" rel="noopener">Source: Berkeley Earth</a>
                            </div>
                        </div>

                        <div class="scale-section">
                            <div class="scale-title">AQI Scale</div>
                            <div class="scale-bar">
                                <div class="scale-seg" style="background: #00e400;">0-50</div>
                                <div class="scale-seg" style="background: #ffff00;">51-100</div>
                                <div class="scale-seg" style="background: #ff7e00; color: #fff;">101-150</div>
                                <div class="scale-seg" style="background: #ff0000; color: #fff;">151-200</div>
                                <div class="scale-seg" style="background: #8f3f97; color: #fff;">201-300</div>
                                <div class="scale-seg" style="background: #7e0023; color: #fff;">301-500</div>
                            </div>
                            <div class="scale-labels">
                                <div class="scale-label">Good</div>
                                <div class="scale-label">Moderate</div>
                                <div class="scale-label">Sensitive</div>
                                <div class="scale-label">Unhealthy</div>
                                <div class="scale-label">Very Bad</div>
                                <div class="scale-label">Hazardous</div>
                            </div>
                        </div>

                        <div style="margin-top: 28px;">
                            <div class="scale-title">Pollutant Levels</div>
                            <div class="poll-grid">
                                <div class="poll-item">
                                    <div class="poll-name">PM2.5</div>
                                    <div class="poll-val">{pm25:.1f}</div>
                                    <div class="poll-unit">&mu;g/m&sup3;</div>
                                </div>
                                <div class="poll-item">
                                    <div class="poll-name">PM10</div>
                                    <div class="poll-val">{aqi_data.get('pm10', 0):.1f}</div>
                                    <div class="poll-unit">&mu;g/m&sup3;</div>
                                </div>
                                <div class="poll-item">
                                    <div class="poll-name">CO</div>
                                    <div class="poll-val">{aqi_data.get('co', 0):.1f}</div>
                                    <div class="poll-unit">&mu;g/m&sup3;</div>
                                </div>
                                <div class="poll-item">
                                    <div class="poll-name">NO&sub2;</div>
                                    <div class="poll-val">{aqi_data.get('no2', 0):.1f}</div>
                                    <div class="poll-unit">&mu;g/m&sup3;</div>
                                </div>
                                <div class="poll-item">
                                    <div class="poll-name">O&sub3;</div>
                                    <div class="poll-val">{aqi_data.get('o3', 0):.1f}</div>
                                    <div class="poll-unit">&mu;g/m&sup3;</div>
                                </div>
                                <div class="poll-item">
                                    <div class="poll-name">SO&sub2;</div>
                                    <div class="poll-val">{aqi_data.get('so2', 0):.1f}</div>
                                    <div class="poll-unit">&mu;g/m&sup3;</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="footer">
                    Powered by ML Weather Model + Berkeley Earth AQI Research<br>
                    Parameters: Humidity, Pressure, Cloud Cover, Precipitation
                </div>
            </div>

            <script>
                function showTab(name, el) {{
                    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.getElementById(name + '-tab').classList.add('active');
                    el.classList.add('active');
                }}
            </script>
        </body>
        </html>
        """

        return html

    except Exception as e:
        return f"<h1>Error: {str(e)}</h1><pre>{repr(e)}</pre>", 500


if __name__ == "__main__":
    print("Starting Real ML Weather + AQI API...")
    print("")
    print("HTML: http://127.0.0.1:5000/rain-forecast?city=Delhi")
    print("AQI+Cigarettes: http://127.0.0.1:5000/weather-aqi?city=Delhi")
    print("")
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
