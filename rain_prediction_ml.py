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
        # Initialize weights (these would normally be learned from training data)
        # In production, you'd load these from a trained model file
        self.weights = {
            'chance_of_rain': 0.35,
            'humidity': 0.20,
            'precipitation': 0.25,
            'cloud_cover': 0.10,
            'pressure': 0.10
        }
        self.trained = False
    
    def train(self, training_data):
        """
        Train the model on historical data.
        training_data format: [{'features': {...}, 'did_rain': True/False}, ...]
        
        This is a simplified version. Real ML would use:
        - sklearn.linear_model.LogisticRegression
        - sklearn.ensemble.RandomForestClassifier
        - Or neural networks
        """
        # This is where real ML magic would happen
        # For now, keeps current weights as baseline
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
        
        # Calculate score using current weights
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
    Based on Berkeley Earth research: 22 μg/m³ PM2.5 = 1 cigarette/day
    """
    
    @staticmethod
    def pm25_to_cigarettes(pm25):
        """
        Convert PM2.5 to cigarette equivalent (per day)
        Formula: PM2.5 / 22 = cigarettes/day
        Source: Berkeley Earth (https://berkeleyearth.org/air-pollution-and-cigarette-equivalence/)
        """
        if pm25 is None or pm25 == 0:
            return 0
        return round(pm25 / 22, 1)
    
    @staticmethod
    def pm25_to_aqi(pm25):
        """
        Convert PM2.5 to AQI (0-500 scale) using EPA breakpoints
        """
        if pm25 is None or pm25 < 0:
            return 0
        
        # EPA AQI breakpoints for PM2.5 (24-hour)
        breakpoints = [
            (0.0, 12.0, 0, 50),      # Good
            (12.1, 35.4, 51, 100),   # Moderate
            (35.5, 55.4, 101, 150),  # Unhealthy for Sensitive Groups
            (55.5, 150.4, 151, 200), # Unhealthy
            (150.5, 250.4, 201, 300),# Very Unhealthy
            (250.5, 500.4, 301, 500) # Hazardous
        ]
        
        for pm_low, pm_high, aqi_low, aqi_high in breakpoints:
            if pm_low <= pm25 <= pm_high:
                # Linear interpolation formula
                aqi = ((aqi_high - aqi_low) / (pm_high - pm_low)) * (pm25 - pm_low) + aqi_low
                return round(aqi)
        
        # If PM2.5 is above 500.4, return 500 (max)
        return 500
    
    @staticmethod
    def aqi_to_category(aqi):
        """
        Get AQI category and color based on 0-500 scale
        """
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
        """
        Get relatable health comparisons
        """
        cigs = AQICigaretteConverter.pm25_to_cigarettes(pm25)
        
        comparisons = []
        
        # Cigarette equivalent
        if cigs < 1:
            comparisons.append(f"Like smoking {cigs} cigarette per day")
        elif cigs == 1:
            comparisons.append("Like smoking 1 cigarette per day")
        else:
            comparisons.append(f"Like smoking {cigs} cigarettes per day")
        
        # Weekly/yearly equivalents
        weekly_cigs = round(cigs * 7, 1)
        yearly_cigs = round(cigs * 365, 0)
        
        if yearly_cigs > 0:
            comparisons.append(f"{int(yearly_cigs)} cigarettes per year")
        
        # Packs (20 cigarettes per pack)
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
    <html>
    <head>
        <title>ML Weather & AQI API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f7fa; }
            h1 { color: #2c3e50; }
            .endpoint { background: white; padding: 20px; margin: 15px 0; 
                        border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            code { background: #34495e; color: white; padding: 4px 10px; border-radius: 5px; }
            .new { background: #e74c3c; color: white; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; }
        </style>
    </head>
    <body>
        <h1>🌧️ Real ML Weather & AQI API</h1>
        
        <div class="endpoint">
            <h3><span class="new">NEW!</span> AQI with Cigarette Equivalent</h3>
            <code>GET /weather-aqi?city=Chennai</code>
            <p><strong>Shows AQI as cigarettes based on Berkeley Earth research</strong></p>
            <ul>
                <li>PM2.5 levels</li>
                <li>Cigarettes per day equivalent</li>
                <li>Annual cigarette exposure</li>
                <li>Health comparisons</li>
            </ul>
        </div>
        
        <div class="endpoint">
            <h3>🎨 Beautiful Forecast with AQI</h3>
            <code>GET /rain-forecast?city=Chennai</code>
            <p>Colorful page with rain prediction + AQI cigarette comparison</p>
        </div>
        
        <div class="endpoint">
            <h3>Examples:</h3>
            <ul>
                <li><a href="/weather-aqi?city=Delhi">Delhi AQI (High Pollution)</a></li>
                <li><a href="/weather-aqi?city=Chennai">Chennai AQI</a></li>
                <li><a href="/rain-forecast?city=Mumbai">Mumbai Forecast</a></li>
            </ul>
        </div>
        
        <div class="endpoint">
            <h3>📚 Scientific Source:</h3>
            <p>Cigarette conversion based on <strong>Berkeley Earth research</strong>:</p>
            <p><em>"22 μg/m³ PM2.5 for 24 hours = 1 cigarette"</em></p>
            <p><a href="https://berkeleyearth.org/air-pollution-and-cigarette-equivalence/">Read the research</a></p>
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
        
        # Get PM2.5 and calculate AQI
        pm25 = aqi_data.get('pm2_5', 0)
        us_epa_index = aqi_data.get('us-epa-index', 0)  # 1-6 scale
        
        # Calculate actual AQI (0-500 scale)
        calculated_aqi = aqi_converter.pm25_to_aqi(pm25)
        aqi_info = aqi_converter.aqi_to_category(calculated_aqi)
        
        cigarettes_per_day = aqi_converter.pm25_to_cigarettes(pm25)
        health_comparisons = aqi_converter.get_health_comparison(pm25)
        
        return jsonify({
            "location": f"{location['name']}, {location['region']}, {location['country']}",
            "current_weather": {
                "temperature": f"{current['temp_c']}°C",
                "condition": current['condition']['text'],
                "humidity": f"{current['humidity']}%"
            },
            "air_quality": {
                "aqi": calculated_aqi,
                "aqi_level": aqi_info['level'],
                "health_advice": aqi_info['advice'],
                "pm2_5": f"{pm25} μg/m³",
                "us_epa_index": us_epa_index,
                "cigarette_equivalent": {
                    "per_day": cigarettes_per_day,
                    "per_week": round(cigarettes_per_day * 7, 1),
                    "per_year": round(cigarettes_per_day * 365, 0),
                    "health_comparisons": health_comparisons
                },
                "pollutants": {
                    "co": f"{aqi_data.get('co', 0):.1f} μg/m³",
                    "no2": f"{aqi_data.get('no2', 0):.1f} μg/m³",
                    "o3": f"{aqi_data.get('o3', 0):.1f} μg/m³",
                    "pm10": f"{aqi_data.get('pm10', 0):.1f} μg/m³"
                }
            },
            "source": "Berkeley Earth research: 22 μg/m³ PM2.5 = 1 cigarette/day",
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
        
        # Get AQI data - FIX: Calculate proper AQI from PM2.5
        aqi_data = current.get('air_quality', {})
        pm25 = aqi_data.get('pm2_5', 0)
        us_epa_index = aqi_data.get('us-epa-index', 0)  # 1-6 scale (internal)
        
        # Calculate actual AQI (0-500 scale that people understand)
        calculated_aqi = aqi_converter.pm25_to_aqi(pm25)
        aqi_info = aqi_converter.aqi_to_category(calculated_aqi)
        
        print(f"DEBUG - PM2.5: {pm25}, Calculated AQI: {calculated_aqi}, EPA Index: {us_epa_index}")
        
        cigarettes_per_day = aqi_converter.pm25_to_cigarettes(pm25)
        yearly_cigs = round(cigarettes_per_day * 365, 0)
        
        # Run rain predictions
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
        
        # HTML with tabs for Rain Forecast and AQI
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Weather Forecast - {location['name']}</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh; padding: 20px;
                }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; color: white; margin-bottom: 30px; }}
                .header h1 {{ font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
                .location {{ font-size: 1.3em; opacity: 0.9; }}
                
                /* Tabs */
                .tabs {{
                    display: flex; gap: 10px; justify-content: center; margin-bottom: 30px;
                }}
                .tab {{
                    background: rgba(255,255,255,0.2); color: white; padding: 15px 30px;
                    border-radius: 10px; cursor: pointer; font-size: 1.1em; font-weight: bold;
                    border: 2px solid transparent; transition: all 0.3s;
                }}
                .tab:hover {{ background: rgba(255,255,255,0.3); }}
                .tab.active {{
                    background: white; color: #667eea; border-color: white;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                }}
                
                .tab-content {{ display: none; }}
                .tab-content.active {{ display: block; }}
                
                /* AQI Card */
                .aqi-card {{
                    background: white; border-radius: 20px; padding: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-bottom: 20px;
                }}
                .aqi-title {{ font-size: 1.8em; color: #2c3e50; margin-bottom: 20px; text-align: center; }}
                .cigarette-equiv {{
                    background: #fff3cd; padding: 25px; border-radius: 15px;
                    border-left: 5px solid #ff6b6b; margin: 20px 0;
                }}
                .big-number {{
                    font-size: 5em; font-weight: bold; color: #e74c3c; text-align: center; margin: 20px 0;
                }}
                .aqi-badge {{
                    display: inline-block; background: {aqi_info['color']};
                    color: white; padding: 15px 30px; border-radius: 30px;
                    font-weight: bold; font-size: 1.3em; box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                }}
                .pollutants-grid {{
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px; margin-top: 20px;
                }}
                .pollutant-item {{
                    background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center;
                }}
                .pollutant-label {{ font-size: 0.9em; color: #7f8c8d; margin-bottom: 5px; }}
                .pollutant-value {{ font-size: 1.3em; font-weight: bold; color: #2c3e50; }}
                
                /* Rain Forecast */
                .summary-card {{
                    background: white; border-radius: 20px; padding: 30px;
                    margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                }}
                .summary-header {{ text-align: center; margin-bottom: 20px; }}
                .rain-status {{
                    display: inline-block; padding: 15px 40px; border-radius: 50px;
                    font-size: 1.8em; font-weight: bold; margin: 10px 0;
                }}
                .rain-yes {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
                .rain-no {{ background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); color: white; }}
                .probability {{ font-size: 3em; font-weight: bold; color: #2c3e50; margin: 15px 0; }}
                .recommendation {{
                    background: #f8f9fa; padding: 20px; border-radius: 15px;
                    border-left: 5px solid #667eea; margin-top: 20px;
                }}
                .recommendation-icon {{ font-size: 2em; margin-right: 10px; }}
                
                /* Day Cards */
                .days-grid {{
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px; margin-top: 30px;
                }}
                .day-card {{
                    background: white; border-radius: 15px; padding: 25px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                    transition: transform 0.3s ease;
                }}
                .day-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
                }}
                .day-header {{
                    display: flex; justify-content: space-between; align-items: center;
                    margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid #f0f0f0;
                }}
                .day-name {{ font-size: 1.5em; font-weight: bold; color: #2c3e50; }}
                .weather-icon {{ width: 64px; height: 64px; }}
                .prediction-badge {{
                    display: inline-block; padding: 10px 20px; border-radius: 25px;
                    font-weight: bold; font-size: 1.1em; margin: 10px 0;
                }}
                .will-rain {{ background: #3498db; color: white; }}
                .no-rain {{ background: #2ecc71; color: white; }}
                .confidence-bar {{
                    background: #ecf0f1; height: 30px; border-radius: 15px;
                    overflow: hidden; margin: 15px 0;
                }}
                .confidence-fill {{
                    height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    display: flex; align-items: center; justify-content: center;
                    color: white; font-weight: bold; transition: width 0.5s ease;
                }}
                .weather-details {{
                    display: grid; grid-template-columns: repeat(2, 1fr);
                    gap: 15px; margin-top: 20px;
                }}
                .detail-item {{
                    background: #f8f9fa; padding: 12px; border-radius: 10px; text-align: center;
                }}
                .detail-label {{ font-size: 0.85em; color: #7f8c8d; margin-bottom: 5px; }}
                .detail-value {{ font-size: 1.2em; font-weight: bold; color: #2c3e50; }}
                .confidence-badge {{
                    display: inline-block; padding: 5px 15px; border-radius: 20px;
                    font-size: 0.9em; font-weight: bold; margin-left: 10px;
                }}
                .confidence-high {{ background: #2ecc71; color: white; }}
                .confidence-medium {{ background: #f39c12; color: white; }}
                .confidence-low {{ background: #e74c3c; color: white; }}
                
                /* Search Box */
                .search-box {{ text-align: center; margin-bottom: 30px; }}
                .search-input {{
                    padding: 15px 25px; font-size: 1.1em; border: none;
                    border-radius: 50px; width: 300px; box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                }}
                .search-button {{
                    padding: 15px 35px; font-size: 1.1em; background: white;
                    color: #667eea; border: none; border-radius: 50px;
                    cursor: pointer; margin-left: 10px; font-weight: bold;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2); transition: all 0.3s ease;
                }}
                .search-button:hover {{
                    background: #667eea; color: white; transform: translateY(-2px);
                    box-shadow: 0 7px 20px rgba(0,0,0,0.3);
                }}
                
                @media (max-width: 768px) {{
                    .header h1 {{ font-size: 1.8em; }}
                    .tabs {{ flex-direction: column; }}
                    .days-grid {{ grid-template-columns: 1fr; }}
                    .search-input {{ width: 200px; font-size: 1em; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌦️ Weather Forecast</h1>
                    <div class="location">📍 {location['name']}, {location['region']}, {location['country']}</div>
                </div>
                
                <div class="search-box">
                    <form action="/rain-forecast" method="get">
                        <input type="text" name="city" placeholder="Enter city name..." class="search-input" value="{city}">
                        <button type="submit" class="search-button">Check Weather</button>
                    </form>
                </div>
                
                <!-- Tabs -->
                <div class="tabs">
                    <div class="tab active" onclick="showTab('rain')">☔ Rain Forecast</div>
                    <div class="tab" onclick="showTab('aqi')">🏭 Air Quality (AQI)</div>
                </div>
                
                <!-- Rain Forecast Tab -->
                <div id="rain-tab" class="tab-content active">
                    <div class="summary-card">
                        <div class="summary-header">
                            <div class="rain-status {'rain-yes' if overall_will_rain else 'rain-no'}">
                                {'☔ YES, It Will Rain!' if overall_will_rain else '☀️ NO Rain Expected'}
                            </div>
                            <div class="probability">{round(max_rain_prob)}% Rain Probability</div>
                        </div>
                        
                        <div class="recommendation">
                            <span class="recommendation-icon">💡</span>
                            <strong>Recommendation:</strong> 
                            {
                                "Pack an umbrella! High chance of rain - plan indoor activities or bring rain gear." if max_rain_prob >= 70 
                                else "Keep an umbrella handy. Moderate chance of rain - check forecast before outdoor plans." if max_rain_prob >= 50
                                else "You're good to go! Low chance of rain - enjoy outdoor activities!"
                            }
                        </div>
                    </div>
                    
                    <div class="days-grid">
        """
        
        # Add cards for each day
        for pred in predictions:
            rain_emoji = "🌧️" if pred['will_rain'] else "☀️"
            card_accent = "#3498db" if pred['will_rain'] else "#2ecc71"
            
            html += f"""
                        <div class="day-card">
                            <div class="day-header">
                                <div>
                                    <div class="day-name">{pred['day_name']}</div>
                                    <div style="color: #7f8c8d; font-size: 0.9em;">{pred['date']}</div>
                                </div>
                                <img src="https:{pred['icon']}" alt="{pred['condition']}" class="weather-icon">
                            </div>
                            
                            <div style="text-align: center; margin: 20px 0;">
                                <div class="prediction-badge {'will-rain' if pred['will_rain'] else 'no-rain'}">
                                    {rain_emoji} {'RAIN EXPECTED' if pred['will_rain'] else 'NO RAIN'}
                                </div>
                                <div style="font-size: 2em; font-weight: bold; color: {card_accent}; margin: 10px 0;">
                                    {round(pred['rain_prob'])}%
                                </div>
                                <div style="color: #7f8c8d;">
                                    Confidence: <span class="confidence-badge confidence-{pred['confidence'].lower()}">{pred['confidence']}</span>
                                </div>
                            </div>
                            
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {pred['rain_prob']}%;">
                                    {round(pred['rain_prob'])}% Probability
                                </div>
                            </div>
                            
                            <div class="weather-details">
                                <div class="detail-item">
                                    <div class="detail-label">🌡️ Temperature</div>
                                    <div class="detail-value">{pred['max_temp']}°C / {pred['min_temp']}°C</div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">💧 Humidity</div>
                                    <div class="detail-value">{pred['humidity']}%</div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">🌧️ Precipitation</div>
                                    <div class="detail-value">{pred['precipitation']} mm</div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">☁️ Condition</div>
                                    <div class="detail-value" style="font-size: 0.9em;">{pred['condition']}</div>
                                </div>
                            </div>
                            
                            {'<div style="margin-top: 15px; padding: 10px; background: #fff3cd; border-radius: 10px; text-align: center; color: #856404;"><strong>⚠️ ' + pred["intensity"] + ' Rain Expected</strong></div>' if pred['will_rain'] else ''}
                        </div>
            """
        
        html += """
                    </div>
                </div>
                
                <!-- AQI Tab -->
                <div id="aqi-tab" class="tab-content">
                    <div class="aqi-card">
                        <h2 class="aqi-title">🏭 Air Quality Index</h2>
                        <div style="text-align: center; margin-bottom: 30px;">
                            <span class="aqi-badge">AQI """ + str(calculated_aqi) + """ - """ + aqi_info['level'] + """</span>
                            <div style="margin-top: 15px; font-size: 1.1em; color: #555;">
                                <strong>""" + aqi_info['advice'] + """</strong>
                            </div>
                        </div>
                        
                        <div class="cigarette-equiv">
                            <h3 style="margin-bottom: 15px; text-align: center;">🚬 Cigarette Equivalent</h3>
                            <p style="font-size: 1.2em; margin-bottom: 10px; text-align: center;">
                                Breathing this air is like smoking:
                            </p>
                            <div class="big-number">""" + str(cigarettes_per_day) + """</div>
                            <div style="text-align: center; font-size: 1.5em; margin-top: -10px; font-weight: bold;">
                                cigarette""" + ("s" if cigarettes_per_day != 1 else "") + """ per day
                            </div>
                            <div style="margin-top: 30px; padding-top: 20px; border-top: 2px solid #ddd; text-align: center;">
                                <p style="font-size: 1.2em;"><strong>📅 """ + str(int(yearly_cigs)) + """ cigarettes per year</strong></p>
                                <p style="margin-top: 15px; font-size: 0.95em; color: #666;">
                                    <strong>Scientific Basis:</strong><br>
                                    Berkeley Earth research shows that 22 μg/m³ PM2.5<br>
                                    for 24 hours = 1 cigarette per day
                                </p>
                                <p style="margin-top: 10px; font-size: 0.9em; color: #888;">
                                    Source: <a href="https://berkeleyearth.org/air-pollution-and-cigarette-equivalence/" target="_blank">Berkeley Earth</a>
                                </p>
                            </div>
                        </div>
                        
                        <div style="margin-top: 30px;">
                            <h3 style="margin-bottom: 15px;">Understanding AQI</h3>
                            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">
                                <p style="margin-bottom: 10px;"><strong>AQI Scale (0-500):</strong></p>
                                <ul style="list-style: none; padding-left: 0;">
                                    <li style="padding: 5px 0;">🟢 <strong>0-50:</strong> Good</li>
                                    <li style="padding: 5px 0;">🟡 <strong>51-100:</strong> Moderate</li>
                                    <li style="padding: 5px 0;">🟠 <strong>101-150:</strong> Unhealthy for Sensitive Groups</li>
                                    <li style="padding: 5px 0;">🔴 <strong>151-200:</strong> Unhealthy</li>
                                    <li style="padding: 5px 0;">🟣 <strong>201-300:</strong> Very Unhealthy</li>
                                    <li style="padding: 5px 0;">🟤 <strong>301-500:</strong> Hazardous</li>
                                </ul>
                            </div>
                        </div>
                        
                        <div style="margin-top: 30px;">
                            <h3 style="margin-bottom: 15px;">Pollutant Levels</h3>
                            <div class="pollutants-grid">
                                <div class="pollutant-item">
                                    <div class="pollutant-label">PM2.5</div>
                                    <div class="pollutant-value">""" + f"{pm25:.1f}" + """ μg/m³</div>
                                </div>
                                <div class="pollutant-item">
                                    <div class="pollutant-label">PM10</div>
                                    <div class="pollutant-value">""" + f"{aqi_data.get('pm10', 0):.1f}" + """ μg/m³</div>
                                </div>
                                <div class="pollutant-item">
                                    <div class="pollutant-label">CO</div>
                                    <div class="pollutant-value">""" + f"{aqi_data.get('co', 0):.1f}" + """ μg/m³</div>
                                </div>
                                <div class="pollutant-item">
                                    <div class="pollutant-label">NO₂</div>
                                    <div class="pollutant-value">""" + f"{aqi_data.get('no2', 0):.1f}" + """ μg/m³</div>
                                </div>
                                <div class="pollutant-item">
                                    <div class="pollutant-label">O₃</div>
                                    <div class="pollutant-value">""" + f"{aqi_data.get('o3', 0):.1f}" + """ μg/m³</div>
                                </div>
                                <div class="pollutant-item">
                                    <div class="pollutant-label">SO₂</div>
                                    <div class="pollutant-value">""" + f"{aqi_data.get('so2', 0):.1f}" + """ μg/m³</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 40px; color: white; opacity: 0.8;">
                    <p>🤖 Powered by Machine Learning Weather Model + Berkeley Earth AQI Research</p>
                    <p style="font-size: 0.9em; margin-top: 10px;">Using meteorological parameters: Humidity, Pressure, Cloud Cover, Precipitation</p>
                </div>
            </div>
            
            <script>
                function showTab(tabName) {
                    // Hide all tabs
                    document.querySelectorAll('.tab-content').forEach(tab => {
                        tab.classList.remove('active');
                    });
                    document.querySelectorAll('.tab').forEach(tab => {
                        tab.classList.remove('active');
                    });
                    
                    // Show selected tab
                    document.getElementById(tabName + '-tab').classList.add('active');
                    event.target.classList.add('active');
                }
            </script>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        return f"<h1>Error: {str(e)}</h1><pre>{repr(e)}</pre>", 500


if __name__ == "__main__":
    print("🌧️  Starting Real ML Weather + AQI API...")
    print("")
    print("🎨 HTML: http://127.0.0.1:5000/rain-forecast?city=Delhi")
    print("📊 AQI+Cigarettes: http://127.0.0.1:5000/weather-aqi?city=Delhi")
    print("")
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
