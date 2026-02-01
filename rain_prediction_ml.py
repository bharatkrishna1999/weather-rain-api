from flask import Flask, request, jsonify
import requests
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

class RainPredictionModel:
    """
    Rain prediction model based on meteorological parameters.
    Uses weighted scoring system based on weather science.
    """
    
    def __init__(self):
        # Weather parameter weights (based on meteorological importance)
        self.weights = {
            'chance_of_rain': 0.35,      # API's own prediction
            'humidity': 0.20,             # High humidity increases rain chance
            'precipitation': 0.25,        # Existing precipitation
            'cloud_cover': 0.10,          # Cloud coverage
            'pressure': 0.10              # Low pressure = rain
        }
    
    def calculate_rain_score(self, weather_params):
        """
        Calculate rain probability score based on weather parameters.
        Returns score between 0-100
        """
        scores = []
        
        # 1. API's chance of rain (strongest indicator)
        if 'chance_of_rain' in weather_params:
            scores.append(weather_params['chance_of_rain'] * self.weights['chance_of_rain'])
        
        # 2. Humidity (>70% increases rain probability)
        if 'humidity' in weather_params:
            humidity = weather_params['humidity']
            humidity_score = 0
            if humidity > 80:
                humidity_score = 100
            elif humidity > 70:
                humidity_score = 75
            elif humidity > 60:
                humidity_score = 50
            else:
                humidity_score = 25
            scores.append(humidity_score * self.weights['humidity'])
        
        # 3. Current precipitation
        if 'precipitation' in weather_params:
            precip = weather_params['precipitation']
            precip_score = min(100, precip * 20)  # Scale: 5mm = 100%
            scores.append(precip_score * self.weights['precipitation'])
        
        # 4. Cloud cover (>75% increases rain)
        if 'cloud_cover' in weather_params:
            cloud = weather_params['cloud_cover']
            cloud_score = 0
            if cloud > 75:
                cloud_score = 100
            elif cloud > 50:
                cloud_score = 60
            else:
                cloud_score = 20
            scores.append(cloud_score * self.weights['cloud_cover'])
        
        # 5. Atmospheric pressure (low pressure = rain)
        if 'pressure' in weather_params:
            pressure = weather_params['pressure']
            pressure_score = 0
            if pressure < 1000:
                pressure_score = 100
            elif pressure < 1010:
                pressure_score = 70
            elif pressure < 1015:
                pressure_score = 40
            else:
                pressure_score = 10
            scores.append(pressure_score * self.weights['pressure'])
        
        # Calculate total score
        total_score = sum(scores)
        return min(100, max(0, total_score))
    
    def predict(self, day_data):
        """
        Predict rain for a given day.
        Returns: (will_rain, confidence, rain_probability)
        """
        params = {
            'chance_of_rain': day_data.get('daily_chance_of_rain', 0),
            'humidity': day_data.get('avghumidity', 50),
            'precipitation': day_data.get('totalprecip_mm', 0),
            'cloud_cover': day_data.get('cloud', 50),
            'pressure': day_data.get('pressure_mb', 1013)
        }
        
        rain_probability = self.calculate_rain_score(params)
        
        # Determine prediction
        will_rain = rain_probability >= 50
        
        # Calculate confidence based on how clear the prediction is
        if rain_probability >= 80 or rain_probability <= 20:
            confidence = "High"
            confidence_percent = 90
        elif rain_probability >= 65 or rain_probability <= 35:
            confidence = "Medium"
            confidence_percent = 70
        else:
            confidence = "Low"
            confidence_percent = 50
        
        return will_rain, confidence, round(rain_probability, 1), confidence_percent


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
            return {
                "level": "Good", 
                "color": "#00e400", 
                "bg_color": "rgba(0, 228, 0, 0.1)",
                "text_color": "#00a000",
                "advice": "Air quality is satisfactory"
            }
        elif aqi <= 100:
            return {
                "level": "Moderate", 
                "color": "#ffff00", 
                "bg_color": "rgba(255, 255, 0, 0.15)",
                "text_color": "#808000",
                "advice": "Acceptable for most people"
            }
        elif aqi <= 150:
            return {
                "level": "Unhealthy for Sensitive Groups", 
                "color": "#ff7e00",
                "bg_color": "rgba(255, 126, 0, 0.15)",
                "text_color": "#cc6400",
                "advice": "Sensitive groups should limit outdoor exposure"
            }
        elif aqi <= 200:
            return {
                "level": "Unhealthy", 
                "color": "#ff0000",
                "bg_color": "rgba(255, 0, 0, 0.15)",
                "text_color": "#cc0000",
                "advice": "Everyone should limit prolonged outdoor exposure"
            }
        elif aqi <= 300:
            return {
                "level": "Very Unhealthy", 
                "color": "#8f3f97",
                "bg_color": "rgba(143, 63, 151, 0.15)",
                "text_color": "#722c79",
                "advice": "Everyone should avoid outdoor activities"
            }
        else:
            return {
                "level": "Hazardous", 
                "color": "#7e0023",
                "bg_color": "rgba(126, 0, 35, 0.15)",
                "text_color": "#650000",
                "advice": "Everyone should remain indoors"
            }


# Initialize the models
rain_model = RainPredictionModel()
aqi_converter = AQICigaretteConverter()


@app.route("/", methods=["GET"])
def home():
    """Home endpoint"""
    return """
    <html>
    <head>
        <title>ML Rain Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f7fa; }
            h1 { color: #2c3e50; }
            .endpoint { background: white; padding: 20px; margin: 15px 0; 
                        border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            code { background: #34495e; color: white; padding: 4px 10px; 
                   border-radius: 5px; font-family: monospace; }
            .highlight { color: #e74c3c; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>🌧️ Machine Learning Rain Prediction API</h1>
        <p class="highlight">📊 Uses ML model based on meteorological parameters</p>
        
        <div class="endpoint">
            <h3>Rain Prediction Endpoint</h3>
            <code>GET /predict-rain?city=Chennai</code>
            <p><strong>Model analyzes:</strong></p>
            <ul>
                <li>Humidity levels</li>
                <li>Atmospheric pressure</li>
                <li>Cloud coverage</li>
                <li>Precipitation patterns</li>
                <li>Historical chance of rain</li>
            </ul>
        </div>
        
        <div class="endpoint">
            <h3>🎨 Beautiful HTML Forecast (NEW!)</h3>
            <code>GET /rain-forecast?city=Chennai</code>
            <p><strong>Colorful, easy-to-understand forecast page with AQI!</strong></p>
            <ul>
                <li>Visual rain probability bars</li>
                <li>Color-coded predictions</li>
                <li>Air Quality Index (AQI)</li>
                <li>Cigarette equivalent calculator</li>
                <li>3-day forecast cards</li>
            </ul>
        </div>
        
        <div class="endpoint">
            <h3>Examples:</h3>
            <ul>
                <li><a href="/rain-forecast">🎯 Auto-Detect My Location</a></li>
                <li><a href="/rain-forecast?city=Chennai">🎨 Chennai - Beautiful Forecast</a></li>
                <li><a href="/rain-forecast?city=Delhi">🎨 Delhi - Beautiful Forecast</a></li>
                <li><a href="/predict-rain?city=Chennai">📊 Chennai - JSON Data</a></li>
            </ul>
        </div>
    </body>
    </html>
    """


@app.route("/predict-rain", methods=["GET"])
def predict_rain():
    """ML-based rain prediction endpoint"""
    city = request.args.get("city")
    
    if not city:
        return jsonify({"error": "city parameter required"}), 400
    
    # Get 3-day forecast with detailed hourly data
    weather_url = "http://api.weatherapi.com/v1/forecast.json"
    weather_params = {
        "key": WEATHER_API_KEY,
        "q": city,
        "days": 3,
        "aqi": "yes"
    }
    
    try:
        print(f"📡 Fetching forecast data for {city}...")
        response = requests.get(weather_url, params=weather_params, timeout=10)
        
        if response.status_code != 200:
            return jsonify({
                "error": "Failed to fetch weather data",
                "status_code": response.status_code
            }), 400
        
        data = response.json()
        location = data['location']
        forecast_days = data['forecast']['forecastday']
        
        # Run predictions for each day
        predictions = []
        overall_will_rain = False
        max_rain_prob = 0
        
        for day in forecast_days:
            # Get average pressure from hourly data
            hourly_pressures = [hour['pressure_mb'] for hour in day['hour']]
            avg_pressure = sum(hourly_pressures) / len(hourly_pressures)
            
            # Get average cloud cover
            hourly_clouds = [hour['cloud'] for hour in day['hour']]
            avg_cloud = sum(hourly_clouds) / len(hourly_clouds)
            
            # Prepare day data for model
            day_data = {
                'daily_chance_of_rain': day['day']['daily_chance_of_rain'],
                'avghumidity': day['day']['avghumidity'],
                'totalprecip_mm': day['day']['totalprecip_mm'],
                'cloud': avg_cloud,
                'pressure_mb': avg_pressure
            }
            
            # Run prediction model
            will_rain, confidence, rain_prob, conf_percent = rain_model.predict(day_data)
            
            if will_rain:
                overall_will_rain = True
            if rain_prob > max_rain_prob:
                max_rain_prob = rain_prob
            
            # Determine rain intensity
            precip = day['day']['totalprecip_mm']
            if precip > 10:
                intensity = "Heavy"
            elif precip > 2.5:
                intensity = "Moderate"
            elif precip > 0:
                intensity = "Light"
            else:
                intensity = "None"
            
            prediction_result = {
                "date": day['date'],
                "day_name": datetime.strptime(day['date'], '%Y-%m-%d').strftime('%A'),
                "prediction": {
                    "will_rain": will_rain,
                    "rain_probability": rain_prob,
                    "confidence": confidence,
                    "confidence_percent": conf_percent
                },
                "weather_params": {
                    "max_temp": f"{day['day']['maxtemp_c']}°C",
                    "min_temp": f"{day['day']['mintemp_c']}°C",
                    "condition": day['day']['condition']['text'],
                    "humidity": f"{day['day']['avghumidity']}%",
                    "precipitation": f"{day['day']['totalprecip_mm']} mm",
                    "rain_intensity": intensity,
                    "cloud_cover": f"{round(avg_cloud)}%",
                    "pressure": f"{round(avg_pressure)} mb"
                }
            }
            predictions.append(prediction_result)
        
        # Overall summary
        days_with_rain = sum(1 for p in predictions if p['prediction']['will_rain'])
        
        summary = {
            "will_rain_in_next_2_days": overall_will_rain,
            "max_rain_probability": max_rain_prob,
            "days_with_rain": f"{days_with_rain} out of 3 days",
            "recommendation": ""
        }
        
        # Generate recommendation
        if max_rain_prob >= 70:
            summary["recommendation"] = "High chance of rain - carry umbrella and plan indoor activities"
        elif max_rain_prob >= 50:
            summary["recommendation"] = "Moderate chance of rain - keep umbrella handy"
        else:
            summary["recommendation"] = "Low chance of rain - outdoor activities should be fine"
        
        print("✅ Prediction complete!")
        
        return jsonify({
            "location": f"{location['name']}, {location['region']}, {location['country']}",
            "prediction_summary": summary,
            "daily_predictions": predictions,
            "model_info": {
                "type": "Weighted Meteorological Scoring Model",
                "parameters_used": [
                    "Chance of Rain (35% weight)",
                    "Humidity (20% weight)",
                    "Precipitation (25% weight)",
                    "Cloud Cover (10% weight)",
                    "Atmospheric Pressure (10% weight)"
                ],
                "prediction_threshold": "50% probability"
            }
        })
        
    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timeout"}), 504
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route("/rain-forecast", methods=["GET"])
def rain_forecast_html():
    """Beautiful HTML version of rain prediction with AQI tab"""
    city = request.args.get("city", None)
    lat = request.args.get("lat", None)
    lon = request.args.get("lon", None)
    
    # If no city and no coordinates, show location detection page
    if not city and not lat and not lon:
        return render_location_page()
    
    # Determine query parameter for weather API
    if lat and lon:
        query = f"{lat},{lon}"
    else:
        query = city
    
    # Get prediction data
    weather_url = "http://api.weatherapi.com/v1/forecast.json"
    weather_params = {
        "key": WEATHER_API_KEY,
        "q": query,
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
        
        # Get AQI data
        aqi_data = current.get('air_quality', {})
        pm25 = aqi_data.get('pm2_5', 0)
        pm10 = aqi_data.get('pm10', 0)
        co = aqi_data.get('co', 0)
        no2 = aqi_data.get('no2', 0)
        o3 = aqi_data.get('o3', 0)
        so2 = aqi_data.get('so2', 0)
        
        # Calculate actual AQI (0-500 scale)
        calculated_aqi = aqi_converter.pm25_to_aqi(pm25)
        aqi_info = aqi_converter.aqi_to_category(calculated_aqi)
        
        cigarettes_per_day = aqi_converter.pm25_to_cigarettes(pm25)
        yearly_cigs = round(cigarettes_per_day * 365, 0)
        cigarette_plural = "s" if cigarettes_per_day != 1 else ""
        
        # Get weather info for AQI tab
        temp = round(current['temp_c'])
        condition = current['condition']['text']
        humidity = current['humidity']
        wind_speed = current['wind_kph']
        uv_index = current['uv']
        
        # Determine weather emoji
        condition_lower = condition.lower()
        if 'sunny' in condition_lower or 'clear' in condition_lower:
            weather_emoji = '☀️'
        elif 'rain' in condition_lower:
            weather_emoji = '🌧️'
        elif 'cloud' in condition_lower:
            weather_emoji = '☁️'
        elif 'snow' in condition_lower:
            weather_emoji = '❄️'
        else:
            weather_emoji = '🌤️'
        
        # Run predictions
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
            if precip > 10:
                intensity = "Heavy"
            elif precip > 2.5:
                intensity = "Moderate"
            elif precip > 0:
                intensity = "Light"
            else:
                intensity = "None"
            
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
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rain Forecast - {location['name']}</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                
                .header {{
                    text-align: center;
                    color: white;
                    margin-bottom: 30px;
                }}
                
                .header h1 {{
                    font-size: 2.5em;
                    margin-bottom: 10px;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }}
                
                .location {{
                    font-size: 1.3em;
                    opacity: 0.9;
                }}
                
                /* Tabs */
                .tabs {{
                    display: flex;
                    gap: 10px;
                    justify-content: center;
                    margin-bottom: 30px;
                }}
                
                .tab {{
                    background: rgba(255,255,255,0.2);
                    color: white;
                    padding: 15px 30px;
                    border-radius: 10px;
                    cursor: pointer;
                    font-size: 1.1em;
                    font-weight: bold;
                    border: 2px solid transparent;
                    transition: all 0.3s;
                }}
                
                .tab:hover {{
                    background: rgba(255,255,255,0.3);
                }}
                
                .tab.active {{
                    background: white;
                    color: #667eea;
                    border-color: white;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                }}
                
                .tab-content {{
                    display: none;
                }}
                
                .tab-content.active {{
                    display: block;
                }}
                
                /* AQI Tab Redesign - Matching the Image */
                .aqi-main-card {{
                    background: transparent;
                    border-radius: 0;
                    padding: 0;
                }}

                .aqi-hero-container {{
                    background: linear-gradient(135deg, rgba(100,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
                    border-radius: 20px;
                    overflow: hidden;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                }}

                .live-indicator {{
                    position: absolute;
                    top: 20px;
                    left: 20px;
                    background: #dc3545;
                    color: white;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-size: 0.85em;
                    font-weight: bold;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}

                .live-dot {{
                    width: 8px;
                    height: 8px;
                    background: white;
                    border-radius: 50%;
                    animation: pulse 1.5s infinite;
                }}

                @keyframes pulse {{
                    0%, 100% {{ opacity: 1; }}
                    50% {{ opacity: 0.3; }}
                }}

                .aqi-display-grid {{
                    display: grid;
                    grid-template-columns: 1fr 400px;
                    gap: 0;
                    min-height: 500px;
                }}

                .aqi-left-main {{
                    position: relative;
                    padding: 60px 40px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                }}

                .aqi-giant-number {{
                    font-size: 12em;
                    font-weight: bold;
                    line-height: 1;
                    text-align: center;
                    margin: 20px 0;
                    text-shadow: 0 4px 20px rgba(0,0,0,0.1);
                }}

                .aqi-label-top {{
                    font-size: 1.1em;
                    color: rgba(0,0,0,0.6);
                    margin-bottom: -10px;
                }}

                .aqi-us-label {{
                    font-size: 0.9em;
                    color: rgba(0,0,0,0.5);
                    margin-top: -10px;
                }}

                .air-status-box {{
                    margin-top: 40px;
                    text-align: center;
                }}

                .status-prefix {{
                    font-size: 1.2em;
                    color: rgba(0,0,0,0.7);
                    margin-bottom: 5px;
                }}

                .status-main {{
                    font-size: 2em;
                    font-weight: bold;
                    color: rgba(0,0,0,0.9);
                }}

                .pm-values-row {{
                    display: flex;
                    gap: 30px;
                    margin-top: 50px;
                    justify-content: center;
                }}

                .pm-box {{
                    text-align: left;
                }}

                .pm-label {{
                    font-size: 0.9em;
                    color: rgba(0,0,0,0.6);
                    margin-bottom: 5px;
                }}

                .pm-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: rgba(0,0,0,0.9);
                }}

                .pm-unit {{
                    font-size: 0.8em;
                    color: rgba(0,0,0,0.5);
                    margin-left: 5px;
                }}

                .aqi-scale-wrapper {{
                    margin-top: 50px;
                    width: 100%;
                    max-width: 600px;
                }}

                .aqi-gradient-bar {{
                    height: 12px;
                    border-radius: 6px;
                    background: linear-gradient(
                        to right,
                        #00e400 0%,
                        #00e400 16.67%,
                        #ffff00 16.67%,
                        #ffff00 33.33%,
                        #ff7e00 33.33%,
                        #ff7e00 50%,
                        #ff0000 50%,
                        #ff0000 66.67%,
                        #8f3f97 66.67%,
                        #8f3f97 83.33%,
                        #7e0023 83.33%,
                        #7e0023 100%
                    );
                    position: relative;
                }}

                .scale-labels-row {{
                    display: flex;
                    justify-content: space-between;
                    margin-top: 8px;
                    font-size: 0.75em;
                    color: rgba(0,0,0,0.6);
                }}

                .scale-numbers-row {{
                    display: flex;
                    justify-content: space-between;
                    margin-top: 3px;
                    font-size: 0.8em;
                    font-weight: bold;
                    color: rgba(0,0,0,0.7);
                }}

                .aqi-right-weather {{
                    background: rgba(255,255,255,0.5);
                    backdrop-filter: blur(10px);
                    padding: 40px 30px;
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }}

                .weather-icon-large {{
                    font-size: 4em;
                    text-align: center;
                    margin-bottom: 10px;
                }}

                .temp-display {{
                    font-size: 4em;
                    font-weight: bold;
                    text-align: center;
                    color: #2c3e50;
                    line-height: 1;
                }}

                .temp-unit {{
                    font-size: 0.5em;
                    vertical-align: super;
                }}

                .weather-condition-text {{
                    text-align: center;
                    font-size: 1.2em;
                    color: #7f8c8d;
                    margin-bottom: 30px;
                }}

                .weather-stats-mini {{
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }}

                .stat-row {{
                    display: flex;
                    align-items: center;
                    gap: 15px;
                    padding: 15px;
                    background: rgba(255,255,255,0.7);
                    border-radius: 10px;
                }}

                .stat-emoji {{
                    font-size: 2em;
                }}

                .stat-content {{
                    flex: 1;
                }}

                .stat-name {{
                    font-size: 0.85em;
                    color: #7f8c8d;
                }}

                .stat-big-value {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                
                .summary-card {{
                    background: white;
                    border-radius: 20px;
                    padding: 30px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                }}
                
                .summary-header {{
                    text-align: center;
                    margin-bottom: 20px;
                }}
                
                .rain-status {{
                    display: inline-block;
                    padding: 15px 40px;
                    border-radius: 50px;
                    font-size: 1.8em;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                
                .rain-yes {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }}
                
                .rain-no {{
                    background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
                    color: white;
                }}
                
                .probability {{
                    font-size: 3em;
                    font-weight: bold;
                    color: #2c3e50;
                    margin: 15px 0;
                }}
                
                .recommendation {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 15px;
                    border-left: 5px solid #667eea;
                    margin-top: 20px;
                }}
                
                .recommendation-icon {{
                    font-size: 2em;
                    margin-right: 10px;
                }}
                
                .days-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-top: 30px;
                }}
                
                .day-card {{
                    background: white;
                    border-radius: 15px;
                    padding: 25px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                    transition: transform 0.3s ease;
                }}
                
                .day-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
                }}
                
                .day-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding-bottom: 15px;
                    border-bottom: 2px solid #f0f0f0;
                }}
                
                .day-name {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                
                .weather-icon {{
                    width: 64px;
                    height: 64px;
                }}
                
                .prediction-badge {{
                    display: inline-block;
                    padding: 10px 20px;
                    border-radius: 25px;
                    font-weight: bold;
                    font-size: 1.1em;
                    margin: 10px 0;
                }}
                
                .will-rain {{
                    background: #3498db;
                    color: white;
                }}
                
                .no-rain {{
                    background: #2ecc71;
                    color: white;
                }}
                
                .confidence-bar {{
                    background: #ecf0f1;
                    height: 30px;
                    border-radius: 15px;
                    overflow: hidden;
                    margin: 15px 0;
                }}
                
                .confidence-fill {{
                    height: 100%;
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    transition: width 0.5s ease;
                }}
                
                .weather-details {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 15px;
                    margin-top: 20px;
                }}
                
                .detail-item {{
                    background: #f8f9fa;
                    padding: 12px;
                    border-radius: 10px;
                    text-align: center;
                }}
                
                .detail-label {{
                    font-size: 0.85em;
                    color: #7f8c8d;
                    margin-bottom: 5px;
                }}
                
                .detail-value {{
                    font-size: 1.2em;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                
                .confidence-badge {{
                    display: inline-block;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-size: 0.9em;
                    font-weight: bold;
                    margin-left: 10px;
                }}
                
                .confidence-high {{
                    background: #2ecc71;
                    color: white;
                }}
                
                .confidence-medium {{
                    background: #f39c12;
                    color: white;
                }}
                
                .confidence-low {{
                    background: #e74c3c;
                    color: white;
                }}
                
                .search-box {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                
                .search-input {{
                    padding: 15px 25px;
                    font-size: 1.1em;
                    border: none;
                    border-radius: 50px;
                    width: 300px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                }}
                
                .search-button {{
                    padding: 15px 35px;
                    font-size: 1.1em;
                    background: white;
                    color: #667eea;
                    border: none;
                    border-radius: 50px;
                    cursor: pointer;
                    margin-left: 10px;
                    font-weight: bold;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                    transition: all 0.3s ease;
                }}
                
                .search-button:hover {{
                    background: #667eea;
                    color: white;
                    transform: translateY(-2px);
                    box-shadow: 0 7px 20px rgba(0,0,0,0.3);
                }}
                
                @media (max-width: 900px) {{
                    .aqi-display-grid {{
                        grid-template-columns: 1fr;
                    }}
                    
                    .aqi-giant-number {{
                        font-size: 8em;
                    }}
                }}
                
                @media (max-width: 768px) {{
                    .header h1 {{
                        font-size: 1.8em;
                    }}
                    
                    .days-grid {{
                        grid-template-columns: 1fr;
                    }}
                    
                    .search-input {{
                        width: 200px;
                        font-size: 1em;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌦️ Rain Forecast</h1>
                    <div class="location">📍 {location['name']}, {location['region']}, {location['country']}</div>
                </div>
                
                <div class="search-box">
                    <form action="/rain-forecast" method="get">
                        <input type="text" name="city" placeholder="Enter city name..." class="search-input" value="{location['name']}">
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
        
        html += f"""
                    </div>
                </div>
                
                <!-- AQI Tab -->
                <div id="aqi-tab" class="tab-content">
                    <div class="aqi-main-card">
                        <!-- Hero Container with AQI Display -->
                        <div class="aqi-hero-container" style="background-color: {aqi_info['bg_color']};">
                            <div class="live-indicator">
                                <div class="live-dot"></div>
                                LIVE
                            </div>
                            
                            <div class="aqi-display-grid">
                                <!-- Left: Main AQI Display -->
                                <div class="aqi-left-main">
                                    <div class="aqi-label-top">Live AQI</div>
                                    <div class="aqi-giant-number" style="color: {aqi_info['text_color']};">
                                        {calculated_aqi}
                                    </div>
                                    <div class="aqi-us-label">AQI (US)</div>
                                    
                                    <div class="air-status-box">
                                        <div class="status-prefix">Air Quality is</div>
                                        <div class="status-main" style="color: {aqi_info['color']};">{aqi_info['level']}</div>
                                    </div>
                                    
                                    <div class="pm-values-row">
                                        <div class="pm-box">
                                            <div class="pm-label">PM2.5 :</div>
                                            <div class="pm-value">
                                                {pm25:.0f}<span class="pm-unit">μg/m³</span>
                                            </div>
                                        </div>
                                        <div class="pm-box">
                                            <div class="pm-label">PM10 :</div>
                                            <div class="pm-value">
                                                {pm10:.0f}<span class="pm-unit">μg/m³</span>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="aqi-scale-wrapper">
                                        <div class="aqi-gradient-bar"></div>
                                        <div class="scale-labels-row">
                                            <span>Good</span>
                                            <span>Moderate</span>
                                            <span>Poor</span>
                                            <span>Unhealthy</span>
                                            <span>Severe</span>
                                            <span>Hazardous</span>
                                        </div>
                                        <div class="scale-numbers-row">
                                            <span>0</span>
                                            <span>50</span>
                                            <span>100</span>
                                            <span>150</span>
                                            <span>200</span>
                                            <span>300</span>
                                            <span>301+</span>
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- Right: Weather Widget -->
                                <div class="aqi-right-weather">
                                    <div class="weather-icon-large">{weather_emoji}</div>
                                    <div class="temp-display">
                                        {temp}<span class="temp-unit">°c</span>
                                    </div>
                                    <div class="weather-condition-text">{condition}</div>
                                    
                                    <div class="weather-stats-mini">
                                        <div class="stat-row">
                                            <div class="stat-emoji">💧</div>
                                            <div class="stat-content">
                                                <div class="stat-name">Humidity</div>
                                                <div class="stat-big-value">{humidity} %</div>
                                            </div>
                                        </div>
                                        <div class="stat-row">
                                            <div class="stat-emoji">💨</div>
                                            <div class="stat-content">
                                                <div class="stat-name">Wind Speed</div>
                                                <div class="stat-big-value">{wind_speed} km/h</div>
                                            </div>
                                        </div>
                                        <div class="stat-row">
                                            <div class="stat-emoji">☀️</div>
                                            <div class="stat-content">
                                                <div class="stat-name">UV Index</div>
                                                <div class="stat-big-value">{uv_index}</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Bottom Section: Cigarette + Pollutants -->
                        <div style="padding: 40px 20px;">
                            <!-- Cigarette Equivalent -->
                            <div style="background: white; border-radius: 15px; padding: 30px; margin-bottom: 30px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                                <h3 style="text-align: center; margin-bottom: 20px;">🚬 Cigarette Equivalent</h3>
                                <p style="text-align: center; color: #7f8c8d; margin-bottom: 15px;">
                                    Breathing this air is like smoking:
                                </p>
                                <div style="font-size: 4em; font-weight: bold; color: #e74c3c; text-align: center; margin: 20px 0;">
                                    {cigarettes_per_day}
                                </div>
                                <div style="font-size: 1.3em; text-align: center; margin-bottom: 20px;">
                                    cigarette{cigarette_plural} per day
                                </div>
                                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center;">
                                    📅 <strong>{int(yearly_cigs)} cigarettes per year</strong>
                                </div>
                                <p style="text-align: center; margin-top: 15px; font-size: 0.85em; color: #95a5a6;">
                                    Source: Berkeley Earth (22 μg/m³ PM2.5 = 1 cigarette/day)
                                </p>
                            </div>
                            
                            <!-- All Pollutants Grid -->
                            <div style="background: white; border-radius: 15px; padding: 30px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                                <h3 style="margin-bottom: 25px;">📊 Detailed Pollutant Levels</h3>
                                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px;">
                                    <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
                                        <div style="font-size: 2em; margin-bottom: 10px;">🔴</div>
                                        <div style="font-size: 0.9em; color: #7f8c8d;">PM2.5</div>
                                        <div style="font-size: 2em; font-weight: bold; color: #2c3e50; margin: 5px 0;">{pm25:.1f}</div>
                                        <div style="font-size: 0.8em; color: #95a5a6;">μg/m³</div>
                                    </div>
                                    <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
                                        <div style="font-size: 2em; margin-bottom: 10px;">🟠</div>
                                        <div style="font-size: 0.9em; color: #7f8c8d;">PM10</div>
                                        <div style="font-size: 2em; font-weight: bold; color: #2c3e50; margin: 5px 0;">{pm10:.1f}</div>
                                        <div style="font-size: 0.8em; color: #95a5a6;">μg/m³</div>
                                    </div>
                                    <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
                                        <div style="font-size: 2em; margin-bottom: 10px;">⚫</div>
                                        <div style="font-size: 0.9em; color: #7f8c8d;">CO</div>
                                        <div style="font-size: 2em; font-weight: bold; color: #2c3e50; margin: 5px 0;">{co:.1f}</div>
                                        <div style="font-size: 0.8em; color: #95a5a6;">μg/m³</div>
                                    </div>
                                    <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
                                        <div style="font-size: 2em; margin-bottom: 10px;">🟡</div>
                                        <div style="font-size: 0.9em; color: #7f8c8d;">NO₂</div>
                                        <div style="font-size: 2em; font-weight: bold; color: #2c3e50; margin: 5px 0;">{no2:.1f}</div>
                                        <div style="font-size: 0.8em; color: #95a5a6;">μg/m³</div>
                                    </div>
                                    <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
                                        <div style="font-size: 2em; margin-bottom: 10px;">🔵</div>
                                        <div style="font-size: 0.9em; color: #7f8c8d;">O₃</div>
                                        <div style="font-size: 2em; font-weight: bold; color: #2c3e50; margin: 5px 0;">{o3:.1f}</div>
                                        <div style="font-size: 0.8em; color: #95a5a6;">μg/m³</div>
                                    </div>
                                    <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
                                        <div style="font-size: 2em; margin-bottom: 10px;">🟤</div>
                                        <div style="font-size: 0.9em; color: #7f8c8d;">SO₂</div>
                                        <div style="font-size: 2em; font-weight: bold; color: #2c3e50; margin: 5px 0;">{so2:.1f}</div>
                                        <div style="font-size: 0.8em; color: #95a5a6;">μg/m³</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 40px; color: white; opacity: 0.8;">
                    <p>🤖 Powered by Machine Learning Weather Prediction Model</p>
                    <p style="font-size: 0.9em; margin-top: 10px;">Using meteorological parameters: Humidity, Pressure, Cloud Cover, Precipitation</p>
                </div>
            </div>
            
            <script>
                function showTab(tabName) {{
                    // Hide all tabs
                    document.querySelectorAll('.tab-content').forEach(tab => {{
                        tab.classList.remove('active');
                    }});
                    document.querySelectorAll('.tab').forEach(tab => {{
                        tab.classList.remove('active');
                    }});
                    
                    // Show selected tab
                    document.getElementById(tabName + '-tab').classList.add('active');
                    event.target.classList.add('active');
                }}
            </script>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        return f"<h1>Error: {str(e)}</h1>", 500




def render_location_page():
    """Render page that asks for location permission and auto-detects city"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Detecting Your Location...</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .location-container {
                background: white;
                border-radius: 20px;
                padding: 50px;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                max-width: 500px;
            }
            h1 { color: #2c3e50; margin-bottom: 20px; font-size: 2em; }
            .loader {
                border: 5px solid #f3f3f3;
                border-top: 5px solid #667eea;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                animation: spin 1s linear infinite;
                margin: 30px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .status { color: #7f8c8d; margin: 20px 0; font-size: 1.1em; }
            .manual-search {
                margin-top: 30px;
                padding-top: 30px;
                border-top: 2px solid #ecf0f1;
            }
            .search-input {
                padding: 15px 25px;
                font-size: 1.1em;
                border: 2px solid #ecf0f1;
                border-radius: 50px;
                width: 100%;
                margin-bottom: 15px;
            }
            .search-button {
                padding: 15px 35px;
                font-size: 1.1em;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 50px;
                cursor: pointer;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            .search-button:hover {
                background: #5568d3;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102,126,234,0.4);
            }
            .error { color: #e74c3c; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="location-container">
            <h1>🌍 Detecting Your Location</h1>
            <div class="loader" id="loader"></div>
            <div class="status" id="status">Getting your location...</div>
            
            <div class="manual-search">
                <h3 style="color: #2c3e50; margin-bottom: 15px;">Or Search Manually</h3>
                <form action="/rain-forecast" method="get">
                    <input type="text" name="city" placeholder="Enter city name..." class="search-input" required>
                    <button type="submit" class="search-button">Check Weather</button>
                </form>
            </div>
        </div>
        
        <script>
            // Automatically detect location on page load
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    // Success callback
                    function(position) {
                        const lat = position.coords.latitude;
                        const lon = position.coords.longitude;
                        
                        document.getElementById('status').textContent = 'Location found! Loading your weather...';
                        
                        // Redirect to weather page with coordinates
                        window.location.href = '/rain-forecast?lat=' + lat + '&lon=' + lon;
                    },
                    // Error callback
                    function(error) {
                        document.getElementById('loader').style.display = 'none';
                        let errorMsg = '';
                        
                        switch(error.code) {
                            case error.PERMISSION_DENIED:
                                errorMsg = "Location permission denied. Please search manually or allow location access.";
                                break;
                            case error.POSITION_UNAVAILABLE:
                                errorMsg = "Location unavailable. Please search manually.";
                                break;
                            case error.TIMEOUT:
                                errorMsg = "Location request timeout. Please search manually.";
                                break;
                            default:
                                errorMsg = "An error occurred. Please search manually.";
                        }
                        
                        document.getElementById('status').innerHTML = '<div class="error">⚠️ ' + errorMsg + '</div>';
                    },
                    // Options
                    {
                        enableHighAccuracy: true,
                        timeout: 5000,
                        maximumAge: 0
                    }
                );
            } else {
                document.getElementById('loader').style.display = 'none';
                document.getElementById('status').innerHTML = '<div class="error">⚠️ Geolocation is not supported by your browser. Please search manually.</div>';
            }
        </script>
    </body>
    </html>
    """


@app.route("/model-info", methods=["GET"])
def model_info():
    """Get information about the prediction model"""
    return jsonify({
        "model_name": "Rain Prediction Model v1.0",
        "model_type": "Weighted Meteorological Scoring System",
        "description": "Predicts rain probability based on multiple weather parameters",
        "input_parameters": {
            "chance_of_rain": {
                "weight": "35%",
                "description": "Weather API's forecast probability"
            },
            "humidity": {
                "weight": "20%",
                "description": "Average humidity percentage",
                "thresholds": ">80% = high rain probability"
            },
            "precipitation": {
                "weight": "25%",
                "description": "Total precipitation in mm",
                "scale": "5mm or more = 100% score"
            },
            "cloud_cover": {
                "weight": "10%",
                "description": "Cloud coverage percentage",
                "thresholds": ">75% = high rain probability"
            },
            "pressure": {
                "weight": "10%",
                "description": "Atmospheric pressure in mb",
                "thresholds": "<1000mb = high rain probability"
            }
        },
        "output": {
            "will_rain": "Boolean prediction (True/False)",
            "rain_probability": "0-100% score",
            "confidence": "Low/Medium/High based on prediction clarity"
        },
        "confidence_calculation": {
            "High": "Probability >80% or <20%",
            "Medium": "Probability 65-80% or 20-35%",
            "Low": "Probability 35-65% (uncertain zone)"
        }
    })


if __name__ == "__main__":
    print("🌧️  Starting ML Rain Prediction Server...")
    print("📊 Model: Weighted Meteorological Scoring System")
    print("")
    print("🎨 Beautiful HTML: http://127.0.0.1:5000/rain-forecast?city=Chennai")
    print("📊 JSON Endpoint: http://127.0.0.1:5000/predict-rain?city=Chennai")
    print("ℹ️  Model Info: http://127.0.0.1:5000/model-info")
    print("")

    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
